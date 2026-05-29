"""
main.py — GPU 서버 메인 진입점

RPi의 sender.py와 짝꿍이 되는 서버.
- LISTEN_PORT (기본 9999): RPi → RGB + Depth + 명령, JSON 응답
- DEBUG_STREAM_PORT (기본 10000, debug_frame_sender): Local PC ← YOLO annotated JPEG
"""

import os
import socket
import struct
import json
import time
import io
import sys
import numpy as np
from PIL import Image

from system import SeraphSystem
from debug_frame_sender import get_debug_sender, send_to_pc


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

    depth_bytes = recv_all(conn, depth_size)
    if depth_bytes is None:
        return None, None, None, None

    return cmd_str, rgb_bytes, depth_bytes, t_pi_start


def decode_rgb(rgb_bytes):
    return Image.open(io.BytesIO(rgb_bytes)).convert("RGB")


def decode_depth(depth_bytes, h=EXPECTED_H, w=EXPECTED_W):
    arr = np.frombuffer(depth_bytes, dtype=np.uint16)
    expected_size = h * w
    if arr.size != expected_size:
        raise ValueError(
            f"depth 크기 불일치: 받은 {arr.size}개, 기대 {expected_size}개 "
            f"({h}x{w}). RPi 해상도 확인 필요."
        )
    return arr.reshape(h, w)


def send_response(conn, response_dict):
    body = json.dumps(response_dict, ensure_ascii=False).encode("utf-8")
    header = struct.pack("<I", len(body))
    conn.sendall(header + body)


def _print_plan_diagnostics(action_command):
    """장애물 인식 디버깅용 진단 출력 (system2에 이미 들어가있지만 main에서 한 번 더 요약)."""
    status = action_command.get("status")
    action = action_command.get("action")

    plan_data = action_command.get("plan", {})
    if plan_data:
        vlm_target = plan_data.get("target_object")
        vlm_obstacles = plan_data.get("obstacle_classes", [])
        print(f"🧠 [VLM 요약] target={vlm_target}, obstacle_classes={vlm_obstacles}")

    if status == "success":
        context = action_command.get("context", {})
        target = context.get("target", {})
        blocking = context.get("blocking_obstacles", [])

        target_dist = target.get("distance", 0)
        target_yaw = target.get("yaw_deg", 0)
        target_aligned = target.get("aligned", False)
        print(f"🎯 [Target] {target.get('class')} "
              f"dist={target_dist:.2f}m yaw={target_yaw:+.1f}° aligned={target_aligned}")

        print(f"🚧 [Blocking] {len(blocking)}건 (action={action})")
        for b in blocking:
            print(f"   - {b.get('class')} "
                  f"dist={b.get('distance', 0):.2f}m "
                  f"yaw={b.get('yaw_deg', 0):+.1f}° "
                  f"conf={b.get('conf', 0):.2f}")

    if status in ("abort", "retry"):
        print(f"⚠️  [{status}] 사유: {action_command.get('reason', 'unknown')}")


def handle_one_request(conn, system, cmd_state):
    cmd_str, rgb_bytes, depth_bytes, t_pi_start = receive_one_frame(conn)
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
    try:
        depth_map = decode_depth(depth_bytes)
    except ValueError as e:
        print(f"❌ depth 디코딩 실패: {e}")
        send_response(conn, {
            "status": "error",
            "reason": str(e),
            "command": command,
        })
        return True

    print(f"🖼  RGB: PIL Image {pil_img.size}, mode={pil_img.mode}")
    print(f"📊 Depth: shape={depth_map.shape}, dtype={depth_map.dtype}, "
          f"min={depth_map.min()}, max={depth_map.max()}, "
          f"valid_pct={(depth_map > 0).sum() / depth_map.size * 100:.1f}%")

    result = system.run(
        image=pil_img,
        command=command,
        depth_map=depth_map,
        avoidance_attempts=0,
    )

    _print_plan_diagnostics(result)

    # YOLO annotated frame → Local PC (non-blocking, 전송 실패해도 파이프라인 계속)
    debug_frame = result.pop("debug_frame", None)
    if debug_frame is not None:
        send_to_pc(debug_frame)

    execution = result.pop("execution", {})

    r_timings = result.get("timings", {})
    t_server_sent = time.time()
    response = {
        "status": result.get("status", "unknown"),
        "command": command,
        "action_command": _strip_internal(result),
        "execution": execution,
        "timings": {
            "pi_to_server_ms": round(pi_to_server_ms, 2),
            "vlm_inference_ms": r_timings.get("vlm_ms", 0.0),
            "yolo_inference_ms": r_timings.get("yolo_ms", 0.0),
            "post_processing_ms": r_timings.get("post_ms", 0.0),
            "plan_total_ms": r_timings.get("plan_ms", 0.0),
            "action_gen_ms": r_timings.get("exec_ms", 0.0),
            "server_sent_ts": t_server_sent,
        },
    }

    send_response(conn, response)
    print(f"📤 응답 전송 완료 "
          f"(VLM: {r_timings.get('vlm_ms', 0):.1f}ms, "
          f"YOLO: {r_timings.get('yolo_ms', 0):.1f}ms, "
          f"post: {r_timings.get('post_ms', 0):.1f}ms, "
          f"execute: {r_timings.get('exec_ms', 0):.1f}ms)")

    return True


def _strip_internal(action_command):
    if not isinstance(action_command, dict):
        return action_command
    action_command.pop("debug_frame", None)
    if "context" in action_command:
        ctx = action_command["context"]
        if "target" in ctx and isinstance(ctx["target"].get("bbox"), list):
            ctx["target"]["bbox"] = [round(v, 1) for v in ctx["target"]["bbox"]]
        for o in ctx.get("obstacles", []):
            if isinstance(o.get("bbox"), list):
                o["bbox"] = [round(v, 1) for v in o["bbox"]]
        for o in ctx.get("blocking_obstacles", []):
            if isinstance(o.get("bbox"), list):
                o["bbox"] = [round(v, 1) for v in o["bbox"]]
        ctx.pop("image", None)
        ctx.pop("config", None)
    return action_command


def create_listen_socket(host, port, label="RPi"):
    """모델 로딩 전 포트 선점 — 충돌 시 VLM 로딩 시간 낭비 방지."""
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
    print("🚀 Seraph 서버 시작")
    print(f"   RPi listen: {LISTEN_HOST}:{LISTEN_PORT}")
    print(f"   Debug stream: 0.0.0.0:{os.getenv('DEBUG_STREAM_PORT', '10000')} (Local PC)")
    print(f"   기대 해상도: {EXPECTED_W}x{EXPECTED_H}")
    print(f"   명령은 RPi가 첫 송신 시 동봉 (cmd_size>0)")
    print("=" * 60)

    server_sock = create_listen_socket(LISTEN_HOST, LISTEN_PORT, label="RPi")
    print(f"✅ RPi 포트 확보: {LISTEN_HOST}:{LISTEN_PORT}")

    system = SeraphSystem()
    cmd_state = CommandState()

    if get_debug_sender() is None:
        print("⚠️  DebugStream 미사용 — YOLO 프레임 Local 저장만 비활성화됩니다.")
    print(f"\n👂 RPi 연결 대기 중...")

    try:
        while True:
            conn, addr = server_sock.accept()
            print(f"\n🔌 RPi 연결됨: {addr}")
            try:
                handle_one_request(conn, system, cmd_state)
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