"""
Seraph Integrated Test Script (통합 테스트 스크립트)

로봇의 전체 시스템(카메라-서버-모터)이 정상적으로 연결되어 작동하는지 검증

1. 카메라 연동 확인: RGB 및 Depth 영상 데이터를 수집
2. 서버 통신 및 인식 검증: 서버로 영상을 보내 YOLO 객체 인식 및 VLM 상황 추론 결과를 수신
3. 상황 판단 검증: 사용자의 맥락(예: 더워, 목말라)에 따라 적절한 물체를 선택했는지 확인
4. 안전 기능 확인: 장애물이 가까이 있을 때 긴급 제동(Emergency Stop)이 작동하는지 체크
5. 모터 구동 확인: 서버의 명령을 받아 실제 모터가 동작하는지 사용자의 승인 후 실행

"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import cv2
import socket
import struct
import json
import time
from motor import MotorDriver

class SeraphIntegratedTest(Node):
    def __init__(self):
        super().__init__('seraph_integrated_test')
        self.bridge = CvBridge()
        self.motor = MotorDriver(use_lcd=True)
        
        # 서버 설정 (사용자의 환경에 맞게 수정 필요)
        self.server_ip = 'localhost' 
        self.server_port = 9999
        
        self.connect_server()

        # 카메라 구독
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        self.ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], 10, 0.1)
        self.ts.registerCallback(self.callback)

        print("\n" + "="*50)
        print("[Seraph 통합 테스트 시스템]")
        print("1. 객체 인식 (YOLO/VLM)")
        print("2. 상황 추론 (Context Reasoning)")
        print("3. 로봇 구동 (Motor Action)")
        print("="*50)
        
        self.test_command = input("로봇에게 시킬 명령을 입력하세요 (예: 목이 말라..): ")
        self.is_done = False

    def connect_server(self):
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.server_ip, self.server_port))
            print(f"[OK] 서버({self.server_ip}) 연결 완료")
        except Exception as e:
            print(f"[Error] 서버 연결 실패: {e}. 서버(main.py)가 실행 중인지 확인하세요.")
            exit()

    def callback(self, color_msg, depth_msg):
        if self.is_done: return
        self.is_done = True # 한 프레임만 테스트

        try:
            print(f"\n[Info] 영상을 캡처하여 서버로 전송 중... (명령: {self.test_command})")
            
            # 이미지 처리
            color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            _, color_encoded = cv2.imencode('.jpg', color_img)
            color_data = color_encoded.tobytes()
            depth_data = depth_img.tobytes()
            cmd_bytes = self.test_command.encode('utf-8')

            # 패킷 송신 (Header 28bytes)
            header = struct.pack('<IQQd', len(cmd_bytes), len(color_data), len(depth_data), time.time())
            self.client_socket.sendall(header + cmd_bytes + color_data + depth_data)

            # 결과 수신
            res_size_data = self.recv_all(self.client_socket, 4)
            if not res_size_data: return
            res_size = struct.unpack('<I', res_size_data)[0]
            res_json_data = self.recv_all(self.client_socket, res_size)
            res = json.loads(res_json_data.decode('utf-8'))

            # --- 검증 파트 ---
            print("\n[1. 객체 인식 및 추론 검증]")
            action_cmd = res.get('action_command', {})
            plan = res.get('plan', {}) # VLM plan 데이터
            context = action_cmd.get('context', {})
            target = context.get('target', {})
            reasoning = plan.get('reasoning', '정보 없음')
            
            if target:
                print(f"[VLM] 인식된 물체: {target.get('class')} (신뢰도: {target.get('conf')})")
                print(f"[YOLO] 위치(화면중앙기준): {target.get('pixel_offset')}px 이동 필요")
                print(f"[Depth] 거리: {target.get('distance'):.2f}m")
                print(f"[Reasoning] 판단 근거: {reasoning}")
            else:
                print("[Fail] 물체를 찾지 못했습니다.")

            print("\n[2. 로봇 작동 검증]")
            motions = res.get('execution', {}).get('executed_motions', [])
            if res.get('status') == 'emergency_stop' or res.get('action') == 'emergency_stop':
                print("[Warning] 긴급 제동! 바로 앞에 장애물이 있어 멈췄습니다.")
            
            if motions:
                print(f"[Plan] 실행될 명령: {motions}")
                ans = input("이대로 로봇을 움직일까요? (y/n): ")
                if ans.lower() == 'y':
                    print("[Action] 모터 가동!")
                    self.motor.execute_motions(motions)
                else:
                    print("[Stop] 작동 취소됨")
            else:
                print("[Info] 수행할 모터 명령이 없습니다.")

        except Exception as e:
            print(f"[Error] 에러 발생: {e}")
        finally:
            print("\n테스트가 끝났습니다. 프로그램을 종료합니다.")
            rclpy.shutdown()

    def recv_all(self, sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

def main():
    rclpy.init()
    node = SeraphIntegratedTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"에러: {e}")

if __name__ == '__main__':
    main()
