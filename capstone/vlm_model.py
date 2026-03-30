import json
import io
import re
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from ultralytics import YOLO

class SeraphVLMTest:
    def __init__(self):
        # 1. 모델 로딩
        print("🚀 [1/2] VLM(Qwen3-VL-4B) 로딩 중...")
        vlm_id = "Qwen/Qwen3-VL-4B-Instruct"
        self.processor = AutoProcessor.from_pretrained(vlm_id)
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            vlm_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        
        print("🚀 [2/2] YOLOv11 엔진 로딩 중...")
        self.yolo = YOLO('yolo11n.pt') 
        print("✅ 모델 준비 완료! 테스트를 시작합니다.")

    def ask_vlm(self, image, system_prompt, user_text):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}]}
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to(self.vlm.device)
        
        outputs = self.vlm.generate(**inputs, max_new_tokens=256, temperature=0.2)
        return self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    def run_test(self, image_path, command):
        """로컬 이미지를 읽어서 VLM -> YOLO -> VLM 실행"""
        try:
            # 이미지 로드
            pil_img = Image.open(image_path).convert('RGB')
            print(f"\n📸 이미지 '{image_path}' 분석 시작 (명령: {command})")

            # [STEP 1] VLM 1차: 의도 파악
            sys_1 = """
            You are a robot vision system. 
            Extract 'intent' and 'target_object' using ONLY stardard COCO datasent classes in JSON format.
            (e.g., cup, bottle, chair, person, cell phone)."""
            res_1 = self.ask_vlm(pil_img, sys_1, command)
            print(f"📦 [VLM 1차 응답]: {res_1}")

            # 1. 처음엔 아무것도 모르는 상태(None)로 시작합니다.
            target_obj = None 

            json_match = re.search(r'\{.*\}', res_1, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    # AI가 말한 타겟을 가져옵니다.
                    target_obj = data.get("target_object") 
                except json.JSONDecodeError:
                    print("⚠️ JSON 파싱 실패")

            # 2. 만약 AI가 타겟을 못 찾았다면?
            if not target_obj:
                print("❌ AI가 명령에서 타겟 물체를 식별하지 못했습니다.")
                return {"status": "fail", "reason": "No target object found"}
            # [STEP 2] YOLO: 좌표 추출
            print(f"🔍 [YOLO] '{target_obj}' 찾는 중...")
            yolo_results = self.yolo.predict(pil_img, conf=0.25, verbose=False)
            bbox = None
            for r in yolo_results:
                for box in r.boxes:
                    label = r.names[int(box.cls[0])].lower()
                    if label == target_obj.lower():
                        bbox = box.xyxy[0].tolist()
                        break
            print(f"🎯 [YOLO 결과]: {bbox}")
            if bbox is None:
                print(f"❌ [YOLO] 이미지 내에서 '{target_obj}'를 물리적으로 특정할 수 없습니다.")
                return {
                    "linear": 0.0, 
                    "angular": 0.0, 
                    "status": f"Target '{target_obj}' not found. Please rotate the robot to scan."
                }
            # [STEP 3] VLM 2차: 최종 액션 결정
            sys_2 = "You are a robot driver. Generate motion JSON: {'linear': v, 'angular': w, 'status': '...'}"
            user_2 = f"Target: {target_obj}, BBox: {bbox}, Distance: 1.2m. What is the next action?"
            res_2 = self.ask_vlm(pil_img, sys_2, user_2)
            
            print(f"🏁 [최종 VLA 결과]: {res_2}")

        except Exception as e:
            print(f"❌ 테스트 중 에러 발생: {e}")

if __name__ == "__main__":
    tester = SeraphVLMTest()
    # 파일명이 test.jpg가 맞는지 확인하세요!
    tester.run_test("test.jpg", "나 목말라.")
