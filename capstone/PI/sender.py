"""
sender.py — RPi 메인 (모터 통합 버전)

구조:
  프레임 송신 → 응답 수신 → 모터 실행(블로킹) → 자동으로 다음 프레임 송신 → ...
  next_action_hint이 done/wait_user/abort/proximity일 때만 사용자 입력 대기.
  retry hint는 자동으로 다음 사이클 진행, MAX_RETRY 초과 시 케이스별 처리.

서버는 그대로(MotionExecutor mock 유지). 서버가 보낸 executed_motions 문자열을
RPi의 MotorDriver(PCA9685)가 받아서 실제 모터를 돌린다.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import threading
import cv2
import socket
import struct
import time
import json

from motor import MotorDriver


SERVER_HOST = 'localhost'
SERVER_PORT = 9999

# task가 끝났음을 의미하는 hint들 → 사용자 입력 대기
TASK_END_HINTS = {'done', 'wait_user', 'abort', 'proximity'}

# 첫 프레임 들어올 때까지 최대 대기 시간
INITIAL_FRAME_TIMEOUT_S = 10.0

# retry 한도 (이 횟수 초과 시 케이스별 처리)
MAX_RETRY = 3


class ImageSender(Node):
    """RealSense 토픽 백그라운드 구독, 최신 프레임 보관."""
    def __init__(self):
        super().__init__('image_sender')
        self.bridge = CvBridge()

        self._frame_lock = threading.Lock()
        self._latest_color = None
        self._latest_depth = None
        self._latest_stamp = None
        self._frame_count = 0

        self.color_sub = message_filters.Subscriber(
            self, Image, '/camera/color/image_raw')
        self.depth_sub = message_filters.Subscriber(
            self, Image, '/camera/aligned_depth_to_color/image_raw')
        ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], 10, 0.1)
        ts.registerCallback(self.image_callback)

        self.get_logger().info('📷 카메라 토픽 구독 시작 (백그라운드 갱신)')

    def image_callback(self, color_msg, depth_msg):
        try:
            color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except Exception as e:
            self.get_logger().error(f'이미지 변환 실패: {e}')
            return

        with self._frame_lock:
            self._latest_color = color_img
            self._latest_depth = depth_img
            self._latest_stamp = time.monotonic()
            self._frame_count += 1

    def get_latest_frame(self):
        with self._frame_lock:
            if self._latest_color is None or self._latest_depth is None:
                return None
            return (
                self._latest_color.copy(),
                self._latest_depth.copy(),
                self._latest_stamp,
                self._frame_count,
            )

    def wait_for_first_frame(self, timeout_s=INITIAL_FRAME_TIMEOUT_S):
        start = time.time()
        while time.time() - start < timeout_s:
            if self.get_latest_frame() is not None:
                return True
            time.sleep(0.1)
        return False


def recv_all(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def send_one_shot(color_img, depth_img, command_to_send):
    """
    매 호출마다 새 소켓 연결 → 송신 → 응답 수신 → 닫기.
    """
    t_pi_start = time.time()

    _, color_encoded = cv2.imencode('.jpg', color_img)
    color_data = color_encoded.tobytes()
    depth_data = depth_img.tobytes()

    if command_to_send is not None:
        cmd_data = command_to_send.encode('utf-8')
    else:
        cmd_data = b''

    header = struct.pack(
        '<IQQd',
        len(cmd_data),
        len(color_data),
        len(depth_data),
        t_pi_start,
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((SERVER_HOST, SERVER_PORT))
        sock.sendall(header + cmd_data + color_data + depth_data)

        res_size_data = recv_all(sock, 4)
        if not res_size_data:
            raise ConnectionError('서버가 응답 헤더를 보내지 않고 끊었음')
        res_size = struct.unpack('<I', res_size_data)[0]

        res_json_data = recv_all(sock, res_size)
        if res_json_data is None:
            raise ConnectionError('서버가 응답 본문을 보내지 않고 끊었음')

    t_pi_end = time.time()
    result = json.loads(res_json_data.decode('utf-8'))
    return result, t_pi_start, t_pi_end


def print_stats(res, t_start, t_end):
    total_rtt = (t_end - t_start) * 1000
    timings = res.get('timings', {})

    pi_to_server = timings.get('pi_to_server_ms')
    vlm_time = timings.get('vlm_inference_ms')
    yolo_time = timings.get('yolo_inference_ms')
    post_time = timings.get('post_processing_ms')
    plan_total = timings.get('plan_total_ms')
    action_time = timings.get('action_gen_ms')
    server_sent_ts = timings.get('server_sent_ts')
    server_to_pi = (
        (t_end - server_sent_ts) * 1000 if server_sent_ts is not None else None
    )

    def fmt(v):
        return f'{v:>7.2f} ms' if isinstance(v, (int, float)) else '    N/A'

    status = res.get('status', 'unknown')
    action_cmd = res.get('action_command') or {}
    action = action_cmd.get('action', 'N/A')
    context = action_cmd.get('context') or {}
    target = context.get('target') or {}
    execution = res.get('execution') or {}
    motions = execution.get('executed_motions', [])
    next_hint = execution.get('next_action_hint', 'N/A')
    reason = action_cmd.get('reason') or res.get('reason')

    print('\n' + '=' * 60)
    print(f"[명령]  {res.get('command', 'N/A')}")
    print(f"[결과]  status={status}, action={action}, next={next_hint}")

    if status in ('abort', 'error', 'retry'):
        print(f"[사유]  reason={reason}")
    else:
        if target:
            dist = target.get('distance')
            yaw = target.get('yaw_deg')
            dist_str = f"{dist:.2f}m" if isinstance(dist, (int, float)) else "N/A"
            yaw_str = f"{yaw:.2f}°" if isinstance(yaw, (int, float)) else "N/A"
            print(f"[타겟]  {target.get('class')} "
                  f"(conf={target.get('conf')}, "
                  f"dist={dist_str}, "
                  f"yaw={yaw_str}, "
                  f"aligned={target.get('aligned')})")
        if motions:
            print(f"[모터]  {' → '.join(motions)}")

    print('-' * 60)
    print(f"  1. Pi → Server     {fmt(pi_to_server)}")
    print(f"  2. VLM 추론        {fmt(vlm_time)}")
    print(f"  3. YOLO 추론       {fmt(yolo_time)}")
    print(f"  4. Post-processing {fmt(post_time)}")
    print(f"     (Plan 합계      {fmt(plan_total)})")
    print(f"  5. Action 생성     {fmt(action_time)}")
    print(f"  6. Server → Pi    {fmt(server_to_pi)}")
    print('-' * 60)
    print(f"  7. 전체 RTT       {total_rtt:>7.2f} ms")
    print('=' * 60)


def prompt_initial_command():
    print('\n[SERAPH RPi Sender]')
    print('  로봇에게 시킬 일을 자연어로 입력하세요.')
    print('  예시: "목 마르네", "리모컨 가져와", "책 좀"')
    print('  (빈 줄 또는 q: 종료)\n')
    cmd = input('명령 > ').strip()
    if not cmd or cmd.lower() == 'q':
        return None
    return cmd


def prompt_after_task_end(current_command, hint):
    if hint == 'done':
        print('\n✅ 도착 완료. 다음에 뭘 할까요?')
    elif hint == 'wait_user':
        print('\n⚠️  로봇이 막혔습니다. 장애물을 치워주세요.')
    elif hint == 'abort':
        print('\n❌ 작업이 중단되었습니다.')
    elif hint == 'proximity':
        print('\n🎯 타겟에 너무 가까이 접근했습니다. (근접 로직은 팀원 작업 대기 중)')

    print(f"  Enter      : 같은 명령으로 재시도 ('{current_command}')")
    print(f"  <텍스트>   : 새 명령으로 변경")
    print(f"  q          : 종료")

    user_input = input('>>> ').strip()
    if user_input.lower() == 'q':
        return 'quit', None
    if user_input == '':
        return 'continue', None
    return 'new_command', user_input


def run_one_cycle(node, motor, current_command, command_dirty):
    snapshot = node.get_latest_frame()
    if snapshot is None:
        print('⚠ 카메라 프레임이 아직 없음. 0.2초 후 재시도.')
        time.sleep(0.2)
        return False, None, command_dirty

    color_img, depth_img, stamp, count = snapshot
    cmd_to_send = current_command if command_dirty else None
    cmd_indicator = "+ 명령" if command_dirty else "(명령 생략)"
    print(f'\n→ 프레임 #{count} 송신 중 {cmd_indicator}...')

    try:
        result, t_start, t_end = send_one_shot(color_img, depth_img, cmd_to_send)
    except Exception as e:
        node.get_logger().error(f'송신/수신 실패: {e}')
        return False, None, command_dirty

    new_dirty = False

    print_stats(result, t_start, t_end)

    if result.get('status') == 'error':
        reason = result.get('reason', 'unknown')
        print(f'⚠ 서버 에러: {reason}')
        if reason == 'no_command_set':
            new_dirty = True
        return True, result, new_dirty

    execution = result.get('execution') or {}
    motions = execution.get('executed_motions', [])
    if motions:
        motor.execute_motions(motions)
    else:
        print('   (실행할 모터 명령 없음)')

    return True, result, new_dirty


def main():
    rclpy.init()
    node = ImageSender()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    motor = MotorDriver()

    current_command = prompt_initial_command()
    if current_command is None:
        print('명령 없이 종료합니다.')
        motor.cleanup()
        node.destroy_node()
        rclpy.shutdown()
        return

    print('\n📷 첫 카메라 프레임 대기 중...')
    if not node.wait_for_first_frame():
        print('❌ 카메라 프레임이 안 들어옵니다. RealSense 노드 확인 필요.')
        motor.cleanup()
        node.destroy_node()
        rclpy.shutdown()
        return
    print('✅ 카메라 OK. 작업 시작.')

    command_dirty = True
    last_hint = None
    retry_count = 0
    last_retry_reason = None

    try:
        while True:
            if last_hint in TASK_END_HINTS:
                action, new_cmd = prompt_after_task_end(current_command, last_hint)
                if action == 'quit':
                    break
                if action == 'new_command':
                    current_command = new_cmd
                    command_dirty = True
                last_hint = None
                retry_count = 0
                last_retry_reason = None

            success, result, command_dirty = run_one_cycle(
                node, motor, current_command, command_dirty
            )

            if not success:
                continue

            execution = result.get('execution') or {}
            last_hint = execution.get('next_action_hint')
            retry_reason = execution.get('next_action_reason')

            if last_hint == 'retry':
                retry_count += 1
                last_retry_reason = retry_reason
                print(f"   🔁 retry {retry_count}/{MAX_RETRY} (reason: {retry_reason})")

                if retry_count >= MAX_RETRY:
                    if retry_reason == 'target_depth_unavailable':
                        print(f"\n🎯 타겟에 너무 가까움 → proximity 로직 진입 예정")
                        last_hint = 'proximity'
                    else:
                        print(f"\n⚠️  {MAX_RETRY}회 연속 retry 실패 ({retry_reason}) → 사용자 대기")
                        last_hint = 'wait_user'
                    retry_count = 0
                    last_retry_reason = None
                else:
                    last_hint = None
            else:
                retry_count = 0
                last_retry_reason = None

    except KeyboardInterrupt:
        print('\n중단됨')
    finally:
        motor.cleanup()
        node.destroy_node()
        rclpy.shutdown()
        print('정리 완료. 종료.')


if __name__ == '__main__':
    main()