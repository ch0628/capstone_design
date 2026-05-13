"""
main.py — GPU 서버 메인 진입점

RPi의 image_sender.py와 짝꿍이 되는 서버.
- 9999 포트 listen
- RPi가 보낸 (명령? + RGB + Depth) 받음
- system2.plan() + system1.execute() 호출
- 결과 JSON으로 응답

(단발 회귀 테스트는 test_runner.py 사용)

Protocol (RPi 측 정의):
  RPi → Server (한 프레임 송신):
    헤더 28 bytes: struct '<IQQd'
      - cmd_size (uint32, 명령 본문 바이트 수. 0이면 명령 없음 → 직전 명령 재사용)
      - rgb_size (uint64)
      - depth_size (uint64)
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

# RPi 헤더 크기 (cmd_size: 4, rgb_size: 8, depth_size: 8, t_pi_start: 8 → 총 28)
HEADER_SIZE = 28
HEADER_FMT = "<IQQd"

# Depth 해상도 (RPi와 합의된 값. 640x480 기본)
EXPECTED_H = 480
EXPECTED_W = 640


class CommandState:
    """
    서버 전체에서 '현재 활성 명령'을 유지.
    RPi는 매 connection마다 새 socket을 맺지만, 명령은 task 단위로 유지된다.
    cmd_size>0인 송신이 오면 갱신, cmd_size=0이면 직전 명령 재사용.
    """
    def __init__(self):
        self.current = None

    def update(self, new_cmd):
        if new_cmd is not None:
            self.current = new_cmd

    def is_set(self):
        return self.current is not None


def recv_all(sock, count):
    """지정한 바이트 수만큼 받을 때까지 반복."""
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
    RPi에서 한 사이클(명령? + RGB + Depth + timestamp) 받기.

    Returns:
        cmd_str (str | None), rgb_bytes, depth_bytes, t_pi_start
        cmd_size=0이면 cmd_str=None.
        연결 끊기면 (None, None, None, None).
    """
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
    """jpg bytes → PIL Image 객체 (메모리 내 처리, 디스크 안 거침)."""
    return Image.open(io.BytesIO(rgb_bytes)).convert("RGB")


def decode_depth(depth_bytes, h=EXPECTED_H, w=EXPECTED_W):
    """raw bytes (uint16 mm) → numpy (H, W) 배열."""
    arr = np.frombuffer(depth_bytes, dtype=np.uint16)
    expected_size = h * w
    if arr.size != expected_size:
        raise ValueError(
            f"depth 크기 불일치: 받은 {arr.size}개, 기대 {expected_size}개 "
            f"({h}x{w}). RPi 해상도 확인 필요."
        )
    return arr.reshape(h, w)


def send_response(conn, response_dict):
    """JSON 응답 송신."""
    body = json.dumps(response_dict, ensure_ascii=False).encode("utf-8")
    header = struct.pack("<I", len(body))
    conn.sendall(header + body)


def _print_plan_diagnostics(action_command):
    """장애물 인식 디버깅용 진단 출력."""
    status = action_command.get("status")
    action = action_command.get("action")
    
    # VLM 출력 확인
    plan_data = action_command.get("plan", {})
    if plan_data:
        vlm_target = plan_data.get("target_object")
        vlm_obstacles = plan_data.get("obstacle_classes", [])
        vlm_reasoning = plan_data.get("reasoning", "")
        print(f"🧠 [VLM] target={vlm_target}, obstacle_classes={vlm_obstacles}")
        if vlm_reasoning:
            print(f"   reasoning: {vlm_reasoning}")
    
    # 장애물 상세 (success 케이스)
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
    
    # abort/retry 사유
    if status in ("abort", "retry"):
        print(f"⚠️  [{status}] 사유: {action_command.get('reason', 'unknown')}")


def handle_one_request(conn, planner, executor, cmd_state):
    """
    한 사이클 처리:
      RPi 데이터 받기 → (명령 갱신) → plan → execute → 응답 송신.

    Returns:
        True if 처리 성공, False if 연결 끊김
    """
    t_server_recv_start = time.time()

    # 1. 데이터 수신
    cmd_str, rgb_bytes, depth_bytes, t_pi_start = receive_one_frame(conn)
    if rgb_bytes is None:
        print("⚠️ 연결 끊김 또는 데이터 없음")
        return False

    # 1-1. 명령 갱신 (RPi가 새 명령 동봉했을 때만)
    if cmd_str is not None:
        cmd_state.update(cmd_str)
        planner.reset_state() # 새 작업 시작 시 상태 초기화
        print(f"📝 [명령 갱신] '{cmd_str}' (상태 초기화)")

    # 1-2. 명령 보장 - 한 번도 명령 받은 적 없으면 에러 응답
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

    # 2. 디코딩 (메모리 내, 디스크 안 거침)
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

    # 3. plan 호출 (System 2)
    t_plan_start = time.time()
    action_command = planner.plan(
        image=pil_img,
        command=command,
        depth_map=depth_map,
        avoidance_attempts=0,   # 단발이라 항상 0 (TODO: 추후 세션화 시 갱신)
    )
    t_plan_done = time.time()

    # 3-1. 진단 출력 (VLM + 장애물 상세)
    _print_plan_diagnostics(action_command)

    # 4. execute 호출 (System 1)
    t_exec_start = time.time()
    execution = executor.execute(action_command)
    t_exec_done = time.time()

    # 5. 응답 패키징 (system2의 정밀 timings 활용)
    t_server_sent = time.time()
    s2_timings = action_command.get("timings", {})
    response = {
        "status": action_command.get("status", "unknown"),
        "command": command,
        "action_command": _strip_internal(action_command),
        "execution": execution,
        "timings": {
            "pi_to_server_ms": round(pi_to_server_ms, 2),
            "vlm_inference_ms": s2_timings.get("vlm_ms", 0.0),
            "yolo_inference_ms": s2_timings.get("yolo_ms", 0.0),
            "post_processing_ms": s2_timings.get("post_ms", 0.0),
            "plan_total_ms": s2_timings.get("total_ms",
                round((t_plan_done - t_plan_start) * 1000, 2)),
            "action_gen_ms": round((t_exec_done - t_exec_start) * 1000, 2),
            "server_sent_ts": t_server_sent,
        },
    }

    # 6. 송신
    send_response(conn, response)
    print(f"📤 응답 전송 완료 "
          f"(VLM: {s2_timings.get('vlm_ms', 0):.1f}ms, "
          f"YOLO: {s2_timings.get('yolo_ms', 0):.1f}ms, "
          f"post: {s2_timings.get('post_ms', 0):.1f}ms, "
          f"execute: {(t_exec_done-t_exec_start)*1000:.1f}ms)")

    return True


def _strip_internal(action_command):
    """출력 정리: 큰 raw bbox round, 내부 필드 제거."""
    if not isinstance(action_command, dict):
        return action_command
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


def main():
    print("=" * 60)
    print("🚀 Seraph 서버 시작")
    print(f"   listen: {LISTEN_HOST}:{LISTEN_PORT}")
    print(f"   기대 해상도: {EXPECTED_W}x{EXPECTED_H}")
    print(f"   명령은 RPi가 첫 송신 시 동봉 (cmd_size>0)")
    print("=" * 60)

    # 모델 로딩 (오래 걸림)
    planner = VisionPlanner()
    executor = MotionExecutor()
    cmd_state = CommandState()

    # 서버 소켓
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((LISTEN_HOST, LISTEN_PORT))
    server_sock.listen(5)
    print(f"\n👂 RPi 연결 대기 중...")

    try:
        while True:
            conn, addr = server_sock.accept()
            print(f"\n🔌 RPi 연결됨: {addr}")
            try:
                handle_one_request(conn, planner, executor, cmd_state)
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