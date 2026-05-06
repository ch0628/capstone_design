import json
import re
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from ultralytics import YOLO
import os
from dotenv import load_dotenv

class SeraphVLMTest:
    def __init__(self):
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")

        if not hf_token:
            raise ValueError("HF_TOKEN이 .env 파일에 없습니다.")
        
        print("🚀 [1/2] VLM(Qwen3.5-4B) 로딩 중...")
        vlm_id = "Qwen/Qwen3.5-4B"

        self.processor = AutoProcessor.from_pretrained(vlm_id, token=hf_token)
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            vlm_id,
            token=hf_token,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

        print("🚀 [2/2] YOLOv11 엔진 로딩 중...")
        self.yolo = YOLO("yolo11n.pt")
        print("✅ 모델 준비 완료! 테스트를 시작합니다.")

    def ask_vlm(self, image, system_prompt, user_text, max_new_tokens=128):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

        inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
        ).to(self.vlm.device)

        outputs = self.vlm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )

        return self.processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )

    def ask_llm(self, system_prompt, user_text, max_new_tokens=128):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

        inputs = self.processor(
            text=[prompt],
            return_tensors="pt",
        ).to(self.vlm.device)

        outputs = self.vlm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )

        return self.processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )

    def run_test(self, image_path, command, distance=1.2):
        try:
            pil_img = Image.open(image_path).convert("RGB")
            print(f"\n📸 이미지 '{image_path}' 분석 시작 (명령: {command}, distance={distance:.2f}m)")

            # STEP 1: 명령에서 우선순위 후보군(target_candidates) 추출
            sys_1 = """
Return only one JSON object.
No explanation.

The "target_candidates" must be a list of COCO classes in order of priority based on the command.
Example: "I want to sit and watch TV" -> ["chair", "couch", "tv"]

Format:
{"intent":"...","target_candidates":["obj1", "obj2"]}
"""
            res_1 = self.ask_vlm(pil_img, sys_1, command, max_new_tokens=192)
            print(f"📦 [VLM 1차 응답]: {res_1}")

            target_candidates = []
            json_matches = re.findall(r"\{.*?\}", res_1, re.DOTALL)
            if json_matches:
                try:
                    data = json.loads(json_matches[-1])
                    target_candidates = data.get("target_candidates", [])
                except json.JSONDecodeError:
                    print("⚠️ JSON 파싱 실패")

            if not target_candidates:
                print("❌ AI가 명령에서 타겟 후보를 식별하지 못했습니다.")
                return {"status": "fail", "reason": "No target candidates found"}

            # STEP 2: YOLO로 후보군 중 탐지되는 첫 번째 물체 탐지
            yolo_results = self.yolo.predict(pil_img, conf=0.25, verbose=False)
            target_obj = None
            best_box = None

            for cand in target_candidates:
                print(f"🔍 [YOLO] '{cand}' 검사 중...")
                for r in yolo_results:
                    for box in r.boxes:
                        label = r.names[int(box.cls[0])].lower()
                        if label == cand.lower():
                            best_box = box
                            target_obj = cand
                            break
                    if best_box: break
                if best_box: break

            if best_box is None:
                print(f"❌ [YOLO] 후보군 {target_candidates} 중 사진에서 탐지된 물체가 없습니다.")
                return {"move": "stop", "status": f"No candidates found in image"}

            bbox = best_box.xyxy[0].tolist()
            print(f"🎯 [최종 타겟 확정]: {target_obj} at {bbox}")
            
            # 결과 이미지 저장 로직
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join("test result", f"{base_name}_result.jpg")
            target_cls_id = int(best_box.cls[0])
            # 해당 클래스 박스만 남겨서 저장
            yolo_results[0].boxes = yolo_results[0].boxes[yolo_results[0].boxes.cls == target_cls_id]
            yolo_results[0].save(filename=save_path)
            print(f"[ '{save_path}'에 저장 완료 ]")

            # 좌표 계산
            img_w, img_h = pil_img.size
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cx_norm = cx / img_w
            cy_norm = (y1 + y2) / 2 / img_h
            w_norm = (x2 - x1) / img_w
            h_norm = (y2 - y1) / img_h

            print(f"cx_norm={cx_norm:.3f}, distance={distance:.2f}")

            # STEP 3: 숫자 정보만으로 move 판단
            sys_2 = """
Return only one JSON object.
No explanation.
Format: {"move":"front","status":"..."}
Allowed: "front", "back", "left", "right", "stop"

Strict rules:
1. If cx_norm < 0.45, return "left".
2. If cx_norm > 0.55, return "right".
3. If 0.45 <= cx_norm <= 0.55 (centered):
   - distance < 0.8: "back"
   - 0.8 <= distance <= 1.2: "stop"
   - distance > 1.2: "front"
"""
            user_2 = (
                f"Target={target_obj}. cx_norm={cx_norm:.3f}, "
                f"w_norm={w_norm:.3f}, h_norm={h_norm:.3f}, distance={distance:.2f}."
            )

            res_2 = self.ask_llm(sys_2, user_2, max_new_tokens=128)
            print(f"🏁 [최종 VLM 결과]: {res_2}")

            json_matches = re.findall(r"\{.*?\}", res_2, re.DOTALL)
            if json_matches:
                try:
                    action = json.loads(json_matches[-1])
                    action['target'] = target_obj
                    sys_3 = """
Return only one JSON object.
No explanation.

Context: The robot has ALREADY found the target object and moved right in front of it.
CRITICAL RULE 1: DO NOT output any tasks related to moving, navigating, or searching (e.g., "이동", "이동하기", "찾기", "다가가기").
CRITICAL RULE 2: The task MUST be highly concise, using only 1~3 words representing the core action (e.g., "옮기기", "가져오기", "열기"). Do not use long sentences.
CRITICAL RULE 3: If there is no logical physical interaction to perform with the target, or if the main goal is simply to show the user where the object is, the task MUST be "주위 돌며 알리기".

Examples:
- Command: "목말라", Target: "cup" -> {"task": "가져오기"}
- Command: "약 먹을 시간이야", Target: "bottle" -> {"task": "전달하기"}
- Command: "이제 잘래", Target: "bed" -> {"task": "조명 제어"}
- Command: "밥 먹을 준비 하자", Target: "chair" -> {"task": "주위 돌며 알리기"}
- Command: "여기 청소해야지", Target: "trash can" -> {"task": "열기"}
- Command: "내 가방 어딨지?", Target: "backpack" -> {"task": "주위 돌며 알리기"}

Format:
{"task": "..."}
"""
                    user_3 = f"Original Command: {command}\nConfirmed Target Object: {target_obj}"
                    res_3 = self.ask_llm(sys_3, user_3, max_new_tokens=64)
                    print(f"🛠️  [최종 VLM 결과(Task)]: {res_3}")
                    
                    task_matches = re.findall(r"\{.*?\}", res_3, re.DOTALL)
                    if task_matches:
                        try:
                            task_data = json.loads(task_matches[-1])
                            action['task'] = task_data.get("task", "대기")
                        except json.JSONDecodeError:
                            action['task'] = "..."
                    else:
                        action['task'] = "..."

                    return action
                except json.JSONDecodeError:
                    pass

            return {"status": "fail", "reason": "Final action parsing failed"}

        except Exception as e:
            print(f"❌ 에러 발생: {e}")
            return {"status": "error", "reason": str(e)}

if __name__ == "__main__":
    file_path = "test_cases.json"
    if not os.path.exists(file_path):
        print(f"❌ {file_path} 파일이 없습니다.")
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            all_cases = json.load(f)

        tester = SeraphVLMTest()

        for case_id, case_data in all_cases.items():
            print(f"\n" + "="*50)
            print(f"🔎 테스트 ID: {case_id}")
            
            dist = case_data.get("distance", 1.7)
            result = tester.run_test(case_data["image_path"], case_data["command"], distance=dist)
            
            all_cases[case_id]["actual"] = result
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(all_cases, f, indent=2, ensure_ascii=False)
                
            print(f"✅ 케이스 {case_id} 완료 및 저장됨")