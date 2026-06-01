"""
Step 1 — GPU 서버 메인 진입점

Configuration : VLM + RGB (Depth 없음)
Capability    : VLM이 명령 해석, 이미지만으로 거리 추정, VLM이 모터 명령 직접 생성
Failure Case  : VLM 시각적 거리 추정 부정확 → 잘못된 정지 거리
"""

import os
import socket
import struct
import json
import time
import io
import sys
from PIL import Image

from system2 import VisionPlanner


LISTEN_HOST = os.getenv("SERAPH_HOST", "0.0.0.0")
LISTEN_PORT = int(os.getenv("SERAPH_PORT", "9999"))

HEADER_SIZE = 28
HEADER_FMT = "<IQQd"

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
    header = recv_all(conn, HEADER_SIZE)
    if header is None:
        return None, None, None, None

    cmd_size, rgb_size, depth_size, t_pi_start = struct.unpack(HEADER_FMT, header)

    cmd_str = None
    if cmd_size > 0:
        cmd_bytes = recv_all(conn, cmd_size)
        if cmd_bytes is None:
            return None, None, None, None
        cmd_str = cmd_bytes.decode("utf-8")

    rgb_bytes = recv_all(conn, rgb_size)
    if rgb_bytes is None:
        return None, None, None, None

    # depth_size만큼 수신하지만 사용하지 않음
    if depth_size > 0:
        depth_bytes = recv_all(conn, depth_size)
        if depth_bytes is None:
            return None, None, None, None

    return cmd_str, rgb_bytes, None, t_pi_start


def decode_rgb(rgb_bytes):
    return Image.open(io.BytesIO(rgb_bytes)).convert("RGB")


def send_response(conn, response_dict):
    body = json.dumps(response_dict, ensure_ascii=False).encode("utf-8")
    header = struct.pack("<I", len(body))
    conn.sendall(header + body)


def _print_plan_diagnostics(action_command):
    plan_data = action_command.get("plan", {})
    if plan_data:
        vlm_target = plan_data.get("target_object")
        est_dist = plan_data.get("estimated_distance_m")
        print(f"🧠 [VLM 요약] target={vlm_target}, estimated_distance={est_dist}m")

    motor_cmds = action_command.get("motor_commands", [])
    print(f"🤖 [Step1] action={action_command.get('action')}, motor={motor_cmds}")

    status = action_command.get("status")
    if status in ("abort", "retry"):
        print(f"⚠️  [{status}] 사유: {action_command.get('reason', 'unknown')}")


def handle_one_request(conn, planner, cmd_state):
    cmd_str, rgb_bytes, _, t_pi_start = receive_one_frame(conn)
    if rgb_bytes is None:
        print("⚠️ 연결 끊김 또는 데이터 없음")
        return False

    if cmd_str is not None:
        cmd_state.update(cmd_str)
        print(f"📝 [명령 갱신] '{cmd_str}'")

    if not cmd_state.is_set():
        print("❌ 명령이 아직 세팅 안 됨. RPi가 먼저 명령을 보내야 함.")
        send_response(conn, {
            "status": "error",
            "reason": "no_command_set",
            "command": None,
        })
        return True

    command = cmd_state.current

    t_server_recv_done = time.time()
    pi_to_server_ms = (t_server_recv_done - t_pi_start) * 1000
    print(f"\n📥 수신 완료 (RPi→Server: {pi_to_server_ms:.1f}ms, 명령: '{command}')")

    pil_img = decode_rgb(rgb_bytes)
    print(f"🖼  RGB: PIL Image {pil_img.size}, mode={pil_img.mode}")

    t_plan_start = time.time()
    action_command = planner.plan(
        image=pil_img,
        command=command,
        depth_map=None,
    )
    t_plan_done = time.time()

    _print_plan_diagnostics(action_command)

    t_server_sent = time.time()
    s2_timings = action_command.get("timings", {})
    response = {
        "status": action_command.get("status", "unknown"),
        "command": command,
        "action": action_command.get("action"),
        "motor_commands": action_command.get("motor_commands", []),
        "next_hint": action_command.get("next_hint", "continue"),
        "timings": {
            "pi_to_server_ms": round(pi_to_server_ms, 2),
            "vlm_inference_ms": s2_timings.get("vlm_ms", 0.0),
            "post_processing_ms": s2_timings.get("post_ms", 0.0),
            "plan_total_ms": s2_timings.get("total_ms",
                round((t_plan_done - t_plan_start) * 1000, 2)),
            "server_sent_ts": t_server_sent,
        },
    }

    send_response(conn, response)
    print(f"📤 응답 전송 완료 "
          f"(VLM: {s2_timings.get('vlm_ms', 0):.1f}ms, "
          f"post: {s2_timings.get('post_ms', 0):.1f}ms)")

    return True


def create_listen_socket(host, port, label="RPi"):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, "SO_REUSEPORT"):
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except OSError:
            pass
    try:
        sock.bind((host, port))
    except OSError as e:
        errno = getattr(e, "errno", None)
        in_use = errno in (98, 48, 10048) or "Address already in use" in str(e)
        if in_use:
            print(
                f"\n❌ 포트 {port} ({label})이(가) 이미 사용 중입니다.\n"
                f"   → 이전 main.py 또는 팀원 프로세스가 점유 중일 수 있습니다.\n"
                f"   확인: ss -tlnp | grep ':{port}'\n"
                f"         fuser -v {port}/tcp\n"
                f"   종료: kill $(fuser {port}/tcp 2>/dev/null)\n",
                file=sys.stderr,
            )
            sys.exit(1)
        raise
    sock.listen(5)
    return sock


def main():
    print("=" * 60)
    print("🚀 Seraph 서버 시작 (Step 1)")
    print(f"   RPi listen: {LISTEN_HOST}:{LISTEN_PORT}")
    print(f"   기대 해상도: {EXPECTED_W}x{EXPECTED_H}")
    print(f"   Depth 미사용 — VLM 시각 추정 모드")
    print(f"   명령은 RPi가 첫 송신 시 동봉 (cmd_size>0)")
    print("=" * 60)

    server_sock = create_listen_socket(LISTEN_HOST, LISTEN_PORT, label="RPi")
    print(f"✅ RPi 포트 확보: {LISTEN_HOST}:{LISTEN_PORT}")

    planner = VisionPlanner()
    cmd_state = CommandState()

    print(f"\n👂 RPi 연결 대기 중...")

    try:
        while True:
            conn, addr = server_sock.accept()
            print(f"\n🔌 RPi 연결됨: {addr}")
            try:
                handle_one_request(conn, planner, cmd_state)
            except Exception as e:
                print(f"❌ 요청 처리 중 에러: {e}")
                import traceback
                traceback.print_exc()
            finally:
                conn.close()
                print("🔚 연결 종료. 다음 RPi 연결 대기...")

    except KeyboardInterrupt:
        print("\n\n🛑 서버 종료")
    finally:
        server_sock.close()


if __name__ == "__main__":
    main()
