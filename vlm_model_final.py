import socket
import json
import struct
import io
import re
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from ultralytics import YOLO

class SeraphVLAServer:
    def __init__(self, host='0.0.0.0', port=9999):
        self.host = host
        self.port = port
        
        # 1. 모델 로딩 (한 번만 실행)
        print("🚀 [1/2] VLM(Qwen3.5-4B) 로딩 중...")
        vlm_id = "Qwen/Qwen3.5-4B"
        self.processor = AutoProcessor.from_pretrained(vlm_id)
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            vlm_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        
        print("🚀 [2/2] YOLOv11 엔진 로딩 중...")
        self.yolo = YOLO('yolo11n.pt') 
        print("✅ 모든 모델 준비 완료!")

    def ask_vlm(self, image, system_prompt, user_text):
        """VLM 추론 공통 함수"""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}]}
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to(self.vlm.device)
        
        outputs = self.vlm.generate(**inputs, max_new_tokens=256, temperature=0.2)
        return self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    def process_logic(self, pil_img, command):
        """VLM -> YOLO -> VLM 통합 파이프라인"""
        
        # [STEP 1] 1차 VLM: 의도 및 타겟 파악
        sys_1 = "You are a robot vision system. Extract 'intent' and 'target_object' in JSON format."
        res_1 = self.ask_vlm(pil_img, sys_1, command)
        
        target_obj = "cup" # 기본값 (파싱 실패 대비)
        try:
            json_match = re.search(r'\{.*\}', res_1, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                target_obj = data.get("target_object", "cup")
                print(f"🎯 1차 판단 결과: {data}")
        except: pass

        # [STEP 2] YOLO: 정밀 좌표 추출
        yolo_results = self.yolo.predict(pil_img, conf=0.25, verbose=False)
        bbox = None
        for r in yolo_results:
            for box in r.boxes:
                if r.names[int(box.cls[0])].lower() == target_obj.lower():
                    bbox = box.xyxy[0].tolist() # [x1, y1, x2, y2]
                    break
        
        # [STEP 3] 2차 VLM: 액션 생성 (Decision Making)
        # BBox와 거리 정보를 프롬프트에 녹여 넣습니다.
        dist_dummy = "1.2m" # 나중에 Depth 데이터로 치환될 부분
        sys_2 = "You are a robot driver. Generate motion JSON: {'linear': v, 'angular': w, 'status': '...'}"
        user_2 = f"Target: {target_obj}, BBox: {bbox}, Distance: {dist_dummy}. What is the next action?"
        
        res_2 = self.ask_vlm(pil_img, sys_2, user_2)
        
        # 최종 결과 JSON만 추출
        try:
            json_match = re.search(r'\{.*\}', res_2, re.DOTALL)
            return json.loads(json_match.group()) if json_match else {"status": "error"}
        except:
            return {"status": "parse_error"}

    def start(self):
        """TCP 서버 시작 및 대기"""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.bind((self.host, self.port))
        server_sock.listen(1)
        print(f"📡 서버 대기 중... ({self.host}:{self.port})")

        while True:
            conn, addr = server_sock.accept()
            print(f"🤝 파이 연결됨: {addr}")
            
            try:
                # 1. 데이터 수신 (텍스트 길이 -> 텍스트 -> 이미지 길이 -> 이미지 바이트)
                header = conn.recv(8)
                text_len, img_len = struct.unpack('II', header)
                
                command = conn.recv(text_len).decode('utf-8')
                img_bytes = b''
                while len(img_bytes) < img_len:
                    chunk = conn.recv(4096)
                    if not chunk: break
                    img_bytes += chunk
                
                # 2. 이미지 복원 및 처리
                pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                print(f"📥 명령 수신: '{command}' / 이미지 분석 시작...")
                
                # 3. VLA 파이프라인 가동
                action_result = self.process_logic(pil_img, command)
                
                # 4. 결과 전송
                conn.sendall(json.dumps(action_result).encode('utf-8'))
                print(f"📤 액션 전송: {action_result}")

            except Exception as e:
                print(f"❌ 에러 발생: {e}")
            finally:
                conn.close()

if __name__ == "__main__":
    server = SeraphVLABrainServer()
    server.start()