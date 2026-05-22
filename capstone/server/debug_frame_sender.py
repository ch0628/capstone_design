"""
Non-blocking debug frame streamer (Seraph Server → Local PC).

- Main thread only calls enqueue() (queue put, ~microseconds).
- Background worker encodes JPEG and sends over TCP.
- Separate accept thread waits for the PC client (SSH port-forward friendly).
- Send failures are logged and ignored; the main pipeline is never blocked.
"""

import os
import socket
import struct
import threading
import time
from collections import deque

import cv2


# Protocol: 8-byte double (server timestamp) + 4-byte uint32 (jpeg length) + jpeg bytes
_HEADER_FMT = "<dI"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

DEFAULT_HOST = os.getenv("DEBUG_STREAM_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("DEBUG_STREAM_PORT", "10000"))
DEFAULT_JPEG_QUALITY = int(os.getenv("DEBUG_JPEG_QUALITY", "80"))
DEFAULT_MAX_QUEUE = int(os.getenv("DEBUG_MAX_QUEUE", "2"))


class DebugFrameSender:
    """
    Push annotated BGR frames to a connected Local PC client.

    Usage:
        sender = DebugFrameSender()
        sender.start()
        ...
        sender.enqueue(annotated_bgr)  # non-blocking
    """

    def __init__(
        self,
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        jpeg_quality=DEFAULT_JPEG_QUALITY,
        max_queue=DEFAULT_MAX_QUEUE,
    ):
        self.host = host
        self.port = port
        self.jpeg_quality = jpeg_quality
        self.max_queue = max(1, max_queue)

        self._queue = deque(maxlen=self.max_queue)
        self._queue_lock = threading.Lock()
        self._queue_cv = threading.Condition(self._queue_lock)

        self._client_sock = None
        self._client_lock = threading.Lock()

        self._listen_sock = None
        self._stop = threading.Event()
        self._accept_thread = None
        self._worker_thread = None

        self._dropped = 0
        self._sent = 0
        self._send_errors = 0

    def start(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        self._stop.clear()
        self._listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                self._listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass
        self._listen_sock.bind((self.host, self.port))
        self._listen_sock.listen(1)
        self._listen_sock.settimeout(1.0)

        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="debug-accept", daemon=True
        )
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="debug-sender", daemon=True
        )
        self._accept_thread.start()
        self._worker_thread.start()

        print(
            f"📡 [DebugStream] listen {self.host}:{self.port} "
            f"(queue={self.max_queue}, jpeg_q={self.jpeg_quality})"
        )

    def stop(self):
        self._stop.set()
        with self._queue_cv:
            self._queue_cv.notify_all()

        if self._listen_sock:
            try:
                self._listen_sock.close()
            except OSError:
                pass

        self._close_client()

        for th in (self._accept_thread, self._worker_thread):
            if th and th.is_alive():
                th.join(timeout=2.0)

    def enqueue(self, frame_bgr):
        """
        Offer one BGR frame (numpy HxWx3). Never blocks the caller.
        If the queue is full, the oldest frame is dropped.
        """
        if frame_bgr is None:
            return

        with self._queue_cv:
            if len(self._queue) >= self.max_queue:
                self._queue.popleft()
                self._dropped += 1
            # Copy so main thread can reuse its buffer safely.
            self._queue.append(frame_bgr.copy())
            self._queue_cv.notify()

    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                conn, addr = self._listen_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    break
                continue

            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._client_lock:
                self._close_client_unlocked()
                self._client_sock = conn
            print(f"🖥️  [DebugStream] PC connected: {addr}")

    def _worker_loop(self):
        while not self._stop.is_set():
            with self._queue_cv:
                while not self._queue and not self._stop.is_set():
                    self._queue_cv.wait(timeout=0.5)
                if self._stop.is_set():
                    break
                frame = self._queue.popleft()

            sock = self._get_client()
            if sock is None:
                continue

            ok, payload = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            )
            if not ok:
                continue

            jpeg = payload.tobytes()
            packet = struct.pack(_HEADER_FMT, time.time(), len(jpeg)) + jpeg

            try:
                sock.sendall(packet)
                self._sent += 1
            except OSError as e:
                self._send_errors += 1
                print(f"⚠️  [DebugStream] send failed: {e}")
                self._close_client()

    def _get_client(self):
        with self._client_lock:
            return self._client_sock

    def _close_client(self):
        with self._client_lock:
            self._close_client_unlocked()

    def _close_client_unlocked(self):
        if self._client_sock is not None:
            try:
                self._client_sock.close()
            except OSError:
                pass
            self._client_sock = None


# Module-level singleton for convenience from main.py
_sender = None
_disabled = False


def get_debug_sender():
    """Returns DebugFrameSender or None if port bind failed."""
    global _sender, _disabled
    if _disabled:
        return None
    if _sender is None:
        try:
            _sender = DebugFrameSender()
            _sender.start()
        except OSError as e:
            _disabled = True
            print(
                f"⚠️  [DebugStream] 포트 {DEFAULT_PORT} 바인드 실패 — 디버그 전송 비활성화: {e}"
            )
            return None
    return _sender


def send_to_pc(frame_bgr):
    """Non-blocking enqueue (alias). No-op if debug stream unavailable."""
    sender = get_debug_sender()
    if sender is not None:
        sender.enqueue(frame_bgr)

