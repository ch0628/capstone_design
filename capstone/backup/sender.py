import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import cv2
import socket
import struct
import time
import json

class ImageSender(Node):
    def __init__(self):
        super().__init__('image_sender')
        self.bridge = CvBridge()
        
        # 소켓 연결
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect(('localhost', 9999))
        self.get_logger().info('✅ 세라프 서버 터널 연결 완료')

        self.color_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], 10, 0.1)
        ts.registerCallback(self.image_callback)

    def image_callback(self, color_msg, depth_msg):
        try:
            t_pi_start = time.time() # [측정 시작] 6번 전체 시간 시작점

            color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

            _, color_encoded = cv2.imencode('.jpg', color_img)
            color_data = color_encoded.tobytes()
            depth_data = depth_img.tobytes()

            # 헤더: [RGB크기(Q)] + [Depth크기(Q)] + [파이송신시간(d)]
            header = struct.pack('<QQd', len(color_data), len(depth_data), t_pi_start)
            self.client_socket.sendall(header + color_data + depth_data)

            # --- 서버로부터 결과 수신 대기 ---
            # 1. 결과 데이터 크기 받기 (4바이트)
            res_size_data = self.recv_all(self.client_socket, 4)
            if not res_size_data: return
            res_size = struct.unpack('<I', res_size_data)[0]

            # 2. 결과 JSON 받기
            res_json_data = self.recv_all(self.client_socket, res_size)
            result = json.loads(res_json_data.decode('utf-8'))

            t_pi_end = time.time() # [측정 종료] 6번 전체 시간 종료점

            # --- 시간 통계 출력 ---
            self.print_stats(result, t_pi_start, t_pi_end)

        except Exception as e:
            self.get_logger().error(f'에러: {e}')

    def recv_all(self, sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def print_stats(self, res, t_start, t_end):
        total_rtt = (t_end - t_start) * 1000
        # 서버에서 계산해서 보낸 값들
        pi_to_server = res['timings']['pi_to_server_ms']
        vlm_time = res['timings']['vlm_inference_ms']
        yolo_time = res['timings']['yolo_inference_ms']
        action_time = res['timings']['action_gen_ms']
        server_to_pi = (t_end - res['timings']['server_sent_ts']) * 1000

        print("\n" + "="*50)
        print(f"[성능 측정 결과] 명령: {res['command']}")
        print("-"*50)
        print(f"1. 네트워크 (Pi -> Server): {pi_to_server:.2f} ms")
        print(f"2. VLM 추론 시간:         {vlm_time:.2f} ms")
        print(f"3. YOLO 추론 시간:        {yolo_time:.2f} ms")
        print(f"4. Action 생성 시간:      {action_time:.2f} ms")
        print(f"5. 네트워크 (Server -> Pi): {server_to_pi:.2f} ms")
        print("-"*50)
        print(f"6. 전체 소요 시간 (RTT):   {total_rtt:.2f} ms")
        print("="*50)

def main():
    rclpy.init()
    node = ImageSender()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
