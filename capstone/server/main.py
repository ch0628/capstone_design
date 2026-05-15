"""
main.py — GPU 서버 메인 진입점

RPi의 image_sender.py와 짝꿍이 되는 서버.
- 9999 포트 listen
- RPi가 보낸 (명령? + RGB + Depth + avoidance_attempts) 받음
- system2.plan() + system1.execute() 호출
- 결과 JSON으로 응답

(단발 회귀 테스트는 test_runner.py 사용)

Protocol (RPi 측 정의):
  RPi → Server (한 프레임 송신):
    헤더 32 bytes: struct '<IQQId'
      - cmd_size (uint32, 명령 본문 바이트 수. 0이면 명령 없음 → 직전 명령 재사용)
      - rgb_size (uint64)
      - depth_size (uint64)
      - avoidance_attempts (uint32)
      - t_pi_start (float64, RPi 송신 시점 timestamp)
    body: cmd_utf8 (cmd_size bytes) + rgb_jpg + depth_raw (uint16 mm)

  Server → RPi (응답):
    헤더 4 bytes: '<I' (json_size, uint32)
    body: JSON utf-8

  JSON 구조 (RPi가 사용하는 키):
    {
      "status": "success" | "abort" | "error",
      "command": str | None,
      "action_command": {...},   # System 2 출력
      "execution": {...},         # System 1 출력
      "timings": {
        "pi_to_server_ms": float,
        "vlm_inference_ms": float,
        "yolo_inference_ms": float,
        "post_processing_ms": float,
        "plan_total_ms": float,
        "action_gen_ms": float,
        "server_sent_ts": float,
      }
    }

명령 관리:
  - 서버는 마지막으로 받은 명령을 CommandState에 보관 (task 단위 유지)
  - cmd_size>0이면 새 명령으로 갱신
  - cmd_size=0이면 보관된 명령 재사용
  - 보관된 명령도 없으면 status="error", reason="no_command_set" 응답
"""

import socket
import struct
import json
import time
import io
import numpy as np
from PIL import Image

from system2 import VisionPlanner
from system1 import MotionExecutor


# ── 시연 설정 ─────────────────────────────────────────────────
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 9999

# RPi 헤더 크기 [개선] avoidance_attempts(uint32) 추가 → 총 32 bytes
# cmd_size: I(4), rgb_size: Q(8), depth_size: Q(8), avoidance_attempts: I(4), t_pi_start: d(8)
HEADER_SIZE = 32
HEADER_FMT = "<IQQId"

# Depth 해상도 (RPi와 합의된 값. 640x480 기본)
EXPECTED_H = 480
EXPECTED_W = 640


class CommandState:
    def __init__(self):
        self.current = None

    def update(self, new_cmd):
        if new_cmd is not None:
            self.current = new_cmd

    def is_set(self):
        return self.current is not None


def recv_all(sock, count):
    buf = b""
    while count > 0:
        chunk = sock.recv(count)
        if not chunk:
            return None
        buf += chunk
        count -= len(chunk)
    return buf


def receive_one_frame(conn):
    """
    RPi에서 한 사이클 데이터 받기.
    """
    header = recv_all(conn, HEADER_SIZE)
    if header is None:
        return None, None, None, 0, None

    cmd_size, rgb_size, depth_size, avoidance_attempts, t_pi_start = struct.unpack(HEADER_FMT, header)

    cmd_str = None
    if cmd_size > 0:
        cmd_bytes = recv_all(conn, cmd_size)
        if cmd_bytes is None:
            return None, None, None, 0, None
        cmd_str = cmd_bytes.decode("utf-8")

    rgb_bytes = recv_all(conn, rgb_size)
    if rgb_bytes is None:
        return None, None, None, 0, None

    depth_bytes = recv_all(conn, depth_size)
    if depth_bytes is None:
        return None, None, None, 0, None

    return cmd_str, rgb_bytes, depth_bytes, avoidance_attempts, t_pi_start


def decode_rgb(rgb_bytes):
    return Image.open(io.BytesIO(rgb_bytes)).convert("RGB")


def decode_depth(depth_bytes, h=EXPECTED_H, w=EXPECTED_W):
    arr = np.frombuffer(depth_bytes, dtype=np.uint16)
    expected_size = h * w
    if arr.size != expected_size:
        raise ValueError(f"depth 크기 불일치: 받은 {arr.size}개, 기대 {expected_size}개")
    return arr.reshape(h, w)


def send_response(conn, response_dict):
    body = json.dumps(response_dict, ensure_ascii=False).encode("utf-8")
    header = struct.pack("<I", len(body))
    conn.sendall(header + body)


def handle_one_request(conn, planner, executor, cmd_state):
    cmd_str, rgb_bytes, depth_bytes, avoidance_attempts, t_pi_start = receive_one_frame(conn)
    if rgb_bytes is None:
        return False

    if cmd_str is not None:
        cmd_state.update(cmd_str)
        planner.reset_state()
        print(f"📝 [명령 갱신] '{cmd_str}'")

    if not cmd_state.is_set():
        send_response(conn, {"status": "error", "reason": "no_command_set"})
        return True

    command = cmd_state.current
    t_server_recv_done = time.time()
    pi_to_server_ms = (t_server_recv_done - t_pi_start) * 1000

    pil_img = decode_rgb(rgb_bytes)
    try:
        depth_map = decode_depth(depth_bytes)
    except ValueError as e:
        send_response(conn, {"status": "error", "reason": str(e)})
        return True

    # 3. plan 호출 (System 2)
    t_plan_start = time.time()
    action_command = planner.plan(
        image=pil_img,
        command=command,
        depth_map=depth_map,
        avoidance_attempts=avoidance_attempts,
    )
    t_plan_done = time.time()

    # 4. execute 호출 (System 1)
    t_exec_start = time.time()
    execution = executor.execute(action_command)
    t_exec_done = time.time()

    # 4-1. [Memory Update] 실행된 움직임을 메모리에 반영
    executed_motions = execution.get("executed_motions", [])
    if executed_motions:
        planner.update_memory_on_motion(executed_motions)
        # [추가] 움직임이 있었다면 다음 프레임에서 바로 '놓침'으로 판단하지 않도록 유예
        planner.target_lost_count = 0

    # Response
    t_server_sent = time.time()
    s2_timings = action_command.get("timings", {})
    response = {
        "status": action_command.get("status", "unknown"),
        "command": command,
        "action_command": _strip_internal(action_command),
        "execution": execution,
        "timings": {
            "pi_to_server_ms": round(pi_to_server_ms, 2),
            "plan_total_ms": s2_timings.get("total_ms", 0.0),
            "action_gen_ms": round((t_server_sent - t_plan_done) * 1000, 2),
            "server_sent_ts": t_server_sent,
        },
    }

    send_response(conn, response)
    return True


def _strip_internal(action_command):
    if not isinstance(action_command, dict): return action_command
    if "context" in action_command:
        ctx = action_command["context"]
        ctx.pop("image", None)
        ctx.pop("config", None)
    return action_command


def main():
    planner = VisionPlanner()
    executor = MotionExecutor()
    cmd_state = CommandState()

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((LISTEN_HOST, LISTEN_PORT))
    server_sock.listen(5)
    print(f"🚀 Seraph 서버 대기 중... ({LISTEN_HOST}:{LISTEN_PORT})")

    try:
        while True:
            conn, addr = server_sock.accept()
            try:
                handle_one_request(conn, planner, executor, cmd_state)
            except Exception as e:
                import traceback; traceback.print_exc()
            finally:
                conn.close()
    finally:
        server_sock.close()


if __name__ == "__main__":
    main()
