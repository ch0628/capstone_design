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


SERVER_HOST = 'localhost'
SERVER_PORT = 9999

# task가 끝났음을 의미하는 hint들 → 새 명령 받을지 물어봐야 함
TASK_END_HINTS = {'done', 'wait_user', 'abort'}


class ImageSender(Node):
    """
    백그라운드에서 RealSense 토픽을 계속 구독하면서 최신 프레임을 보관.
    실제 송신은 외부에서 send_one_shot()을 호출할 때만 일어남.
    """
    def __init__(self):
        super().__init__('image_sender')
        self.bridge = CvBridge()

        # 최신 프레임 보관소 (콜백이 갱신, 메인 루프가 읽음)
        self._frame_lock = threading.Lock()
        self._latest_color = None   # numpy bgr8
        self._latest_depth = None   # numpy uint16 (mm)
        self._latest_stamp = None   # 프레임 수신 시각 (monotonic, 디버그용)
        self._frame_count = 0       # 누적 수신 프레임 수

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
        """메인 루프가 송신할 때 호출. 최신 프레임 스냅샷 반환."""
        with self._frame_lock:
            if self._latest_color is None or self._latest_depth is None:
                return None
            # copy: 콜백이 덮어써도 송신 중인 데이터가 안 망가지게
            return (
                self._latest_color.copy(),
                self._latest_depth.copy(),
                self._latest_stamp,
                self._frame_count,
            )


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
    서버는 무한 루프로 다음 연결을 기다리는 구조라 이게 깔끔함.

    command_to_send: str | None
        - 문자열이면 cmd_size>0으로 동봉 (서버가 명령 갱신)
        - None이면 cmd_size=0 (서버가 마지막 명령 재사용)
    """
    t_pi_start = time.time()  # 서버랑 시간 비교해야 하니 wall clock 유지

    _, color_encoded = cv2.imencode('.jpg', color_img)
    color_data = color_encoded.tobytes()
    depth_data = depth_img.tobytes()

    if command_to_send is not None:
        cmd_data = command_to_send.encode('utf-8')
    else:
        cmd_data = b''

    # 헤더: <IQQd>
    #   cmd_size, rgb_size, depth_size, t_pi_start
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

    if status in ('abort', 'error'):
        print(f"[중단]  reason={reason}")
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
    """
    프로그램 시작 시 명령 강제 입력. 빈 입력이나 q면 None 반환 (종료).
    """
    print('\n[SERAPH RPi Sender]')
    print('  로봇에게 시킬 일을 자연어로 입력하세요.')
    print('  예시: "목 마르네", "리모컨 가져와", "책 좀"')
    print('  (빈 줄 또는 q: 종료)\n')
    cmd = input('명령 > ').strip()
    if not cmd or cmd.lower() == 'q':
        return None
    return cmd


def prompt_after_task_end(current_command, hint):
    """
    task가 끝난 상태(done/wait_user/abort)에서의 입력 분기.

    Returns:
        ('continue', None)        : 같은 명령 재시도
        ('new_command', new_cmd)  : 새 명령으로 교체
        ('quit', None)            : 종료
    """
    if hint == 'done':
        print('\n✅ 도착 완료. 다음에 뭘 할까요?')
    elif hint == 'wait_user':
        print('\n⚠️  로봇이 막혔습니다. 장애물을 치워주세요.')
    elif hint == 'abort':
        print('\n❌ 작업이 중단되었습니다.')

    print(f"  Enter      : 같은 명령으로 재시도 ('{current_command}')")
    print(f"  <텍스트>   : 새 명령으로 변경")
    print(f"  q          : 종료")

    user_input = input('>>> ').strip()
    if user_input.lower() == 'q':
        return 'quit', None
    if user_input == '':
        return 'continue', None
    return 'new_command', user_input


def prompt_continue(current_command):
    """
    task 진행 중(continue/reevaluate)일 때 단순 Enter 트리거.
    Returns: True if 계속, False if 종료
    """
    user_input = input(
        f">>> Enter로 다음 프레임 송신 [명령: '{current_command}'] (q: 종료): "
    ).strip().lower()
    return user_input != 'q'


def main():
    rclpy.init()
    node = ImageSender()

    # rclpy.spin을 백그라운드 스레드로 돌려서 콜백이 계속 latest_frame을 갱신
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # 1) 시작 시 명령 강제 입력
    current_command = prompt_initial_command()
    if current_command is None:
        print('명령 없이 종료합니다.')
        node.destroy_node()
        rclpy.shutdown()
        return

    command_dirty = True   # 첫 송신엔 무조건 명령 동봉
    last_hint = None       # 직전 응답의 next_action_hint

    try:
        while True:
            # 2) hint에 따라 프롬프트 분기
            if last_hint is None or last_hint not in TASK_END_HINTS:
                # 첫 진입 또는 진행 중(continue/reevaluate) → 단순 Enter
                if not prompt_continue(current_command):
                    break
            else:
                # task 종료 → 재시도/새명령/종료 선택
                # (운영 시엔 이 자리를 아두이노 시그널 + UI로 대체)
                action, new_cmd = prompt_after_task_end(current_command, last_hint)
                if action == 'quit':
                    break
                if action == 'new_command':
                    current_command = new_cmd
                    command_dirty = True
                # 'continue'면 그대로 진행 (command_dirty 유지)

            # 3) 최신 프레임 가져오기
            snapshot = node.get_latest_frame()
            if snapshot is None:
                print('⚠ 아직 카메라 프레임이 안 들어왔어. 잠시 후 다시 시도.')
                continue

            color_img, depth_img, stamp, count = snapshot
            cmd_to_send = current_command if command_dirty else None
            cmd_indicator = "+ 명령" if command_dirty else "(명령 생략)"
            print(f'  → 프레임 #{count} 송신 중 {cmd_indicator}...')

            # 4) 송신
            try:
                result, t_start, t_end = send_one_shot(
                    color_img, depth_img, cmd_to_send
                )
                # 송신 성공 → 서버가 명령을 알고 있음
                if command_dirty:
                    command_dirty = False

                print_stats(result, t_start, t_end)

                # next_hint 갱신
                execution = result.get('execution') or {}
                last_hint = execution.get('next_action_hint')

                # 에러 응답 처리
                if result.get('status') == 'error':
                    reason = result.get('reason', 'unknown')
                    print(f'⚠ 서버 에러: {reason}')
                    if reason == 'no_command_set':
                        # 서버가 명령을 잃어버림 → 다음 송신에 재동봉
                        command_dirty = True

            except Exception as e:
                node.get_logger().error(f'송신/수신 실패: {e}')
                # 송신 실패면 dirty 유지 (다음 시도에 명령 다시 보냄)

    except KeyboardInterrupt:
        print('\n중단됨')
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print('정리 완료. 종료.')


if __name__ == '__main__':
    main()