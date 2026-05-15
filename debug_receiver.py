"""
Local PC — debug frame receiver (save-only).

Seraph Server가 보낸 YOLO annotated JPEG를 로컬 폴더에만 저장합니다.
화면 표시(cv2.imshow) 없음 — SSH 터널 + 디스크 저장만.

Protocol (debug_frame_sender.py):
  8 bytes: double  — server timestamp
  4 bytes: uint32 — JPEG length
  N bytes: JPEG
"""

import argparse
import os
import socket
import struct
import time
from datetime import datetime


_HEADER_FMT = "<dI"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 10000
DEFAULT_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_frames")


def recv_exact(sock, nbytes):
    buf = b""
    while len(buf) < nbytes:
        chunk = sock.recv(nbytes - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def connect_with_retry(host, port, retry_interval=2.0):
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
            sock.connect((host, port))
            print(f"✅ Connected to {host}:{port}")
            return sock
        except OSError as e:
            print(f"⏳ Waiting for server ({host}:{port})... {e}")
            time.sleep(retry_interval)


def save_jpeg(jpeg_bytes, save_dir, server_ts, frame_index):
    os.makedirs(save_dir, exist_ok=True)
    local_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    server_part = f"{server_ts:.3f}".replace(".", "_")
    filename = f"frame_{frame_index:06d}_{local_ts}_srv{server_part}.jpg"
    path = os.path.join(save_dir, filename)
    with open(path, "wb") as f:
        f.write(jpeg_bytes)
    return path


def run(host, port, save_dir, print_every):
    print("=" * 60)
    print("💾 Seraph Debug Frame Receiver (save-only)")
    print(f"   connect: {host}:{port}")
    print(f"   save to: {save_dir}")
    print("   Ctrl+C to quit")
    print("=" * 60)

    frame_count = 0

    try:
        while True:
            sock = connect_with_retry(host, port)
            try:
                while True:
                    header = recv_exact(sock, _HEADER_SIZE)
                    if header is None:
                        raise ConnectionError("connection closed while reading header")

                    server_ts, jpeg_len = struct.unpack(_HEADER_FMT, header)
                    if jpeg_len == 0 or jpeg_len > 50 * 1024 * 1024:
                        raise ConnectionError(f"invalid jpeg length: {jpeg_len}")

                    jpeg_bytes = recv_exact(sock, jpeg_len)
                    if jpeg_bytes is None:
                        raise ConnectionError("connection closed while reading jpeg")

                    frame_count += 1
                    path = save_jpeg(jpeg_bytes, save_dir, server_ts, frame_count)

                    if print_every <= 1 or frame_count % print_every == 0:
                        print(f"💾 #{frame_count} ({len(jpeg_bytes) // 1024} KB) → {path}")

            except (ConnectionError, OSError) as e:
                print(f"⚠️  Disconnected: {e}. Reconnecting...")
            finally:
                try:
                    sock.close()
                except OSError:
                    pass

    except KeyboardInterrupt:
        print(f"\n🛑 Stopped. Total saved: {frame_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Receive YOLO annotated JPEG from Seraph Server and save locally"
    )
    parser.add_argument("--host", default=os.getenv("DEBUG_STREAM_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.getenv("DEBUG_STREAM_PORT", str(DEFAULT_PORT))))
    parser.add_argument("--save-dir", default=os.getenv("DEBUG_SAVE_DIR", DEFAULT_SAVE_DIR))
    parser.add_argument(
        "--print-every",
        type=int,
        default=int(os.getenv("DEBUG_PRINT_EVERY", "1")),
        help="Log every N-th frame (default: 1 = every frame)",
    )
    args = parser.parse_args()

    run(args.host, args.port, args.save_dir, max(1, args.print_every))


if __name__ == "__main__":
    main()

