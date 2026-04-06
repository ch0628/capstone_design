import json
import re
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from ultralytics import YOLO
import os


class SeraphVLMTest:
    def __init__(self):
        print("🚀 [1/2] VLM(Qwen3.5-4B) 로딩 중...")
        vlm_id = "Qwen/Qwen3.5-4B"

        self.processor = AutoProcessor.from_pretrained(vlm_id)
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            vlm_id,
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

            # STEP 1: 명령에서 target object 추출
            sys_1 = """
Return only one JSON object.
No explanation.

Format:
{"intent":"drink","target_object":"cup"}

Use only COCO classes for target_object.
"""
            res_1 = self.ask_vlm(pil_img, sys_1, command, max_new_tokens=192)
            print(f"📦 [VLM 1차 응답]: {res_1}")

            target_obj = None
            json_matches = re.findall(r"\{.*?\}", res_1, re.DOTALL)
            if json_matches:
                try:
                    data = json.loads(json_matches[-1])
                    target_obj = data.get("target_object")
                except json.JSONDecodeError:
                    print("⚠️ JSON 파싱 실패")

            if not target_obj:
                print("❌ AI가 명령에서 타겟 물체를 식별하지 못했습니다.")
                return {"status": "fail", "reason": "No target object found"}

            # STEP 2: YOLO로 target object 탐지
            print(f"🔍 [YOLO] '{target_obj}' 찾는 중...")
            yolo_results = self.yolo.predict(pil_img, conf=0.25, verbose=False)

            best_box = None
            best_conf = -1.0

            for r in yolo_results:
                for box in r.boxes:
                    label = r.names[int(box.cls[0])].lower()
                    conf = float(box.conf[0])
                    if label == target_obj.lower() and conf > best_conf:
                        best_conf = conf
                        best_box = box

            bbox = best_box.xyxy[0].tolist() if best_box is not None else None
            print(f"🎯 [YOLO 결과]: {bbox}")

            if bbox is None:
                print(f"❌ [YOLO] 이미지 내에서 '{target_obj}'를 물리적으로 특정할 수 없습니다.")
                return {"move": "stop", "status": f"Target '{target_obj}' not found"}
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join("test_result", f"{base_name}_result.jpg")
            yolo_results[0].save(filename=save_path)
            print(f"[ '{save_path}' 파일로 저장 완료 ]")

            img_w, img_h = pil_img.size
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            cx_norm = cx / img_w
            cy_norm = cy / img_h
            w_norm = (x2 - x1) / img_w
            h_norm = (y2 - y1) / img_h

            print(
                f"cx_norm={cx_norm:.3f}, cy_norm={cy_norm:.3f}, "
                f"w_norm={w_norm:.3f}, h_norm={h_norm:.3f}, distance={distance:.2f}"
            )

            # STEP 3: 이미지 없이 숫자 정보만으로 move 판단
            sys_2 = """
Return only one JSON object.
No explanation.

Format:
{"move":"front","status":"..."}

Allowed values for "move":
- "front"
- "back"
- "left"
- "right"
- "stop"

You must decide using ONLY the numeric values provided by the user.
Do NOT use visual impression, object pose, handle direction, or background context.

Strict rules:
1. Use cx_norm as the primary rule for horizontal movement.
2. If cx_norm < 0.45, return "left".
3. If cx_norm > 0.55, return "right".
4. If 0.45 <= cx_norm <= 0.55, treat the target as horizontally centered.
5. When the target is horizontally centered, use ONLY distance:
   - if distance < 0.8, return "back"
   - if 0.8 <= distance <= 1.2, return "stop"
   - if distance > 1.2, return "front"

These rules are mandatory.
Do not override them.
"""

            user_2 = (
                f"Target={target_obj}. "
                f"cx_norm={cx_norm:.3f}, cy_norm={cy_norm:.3f}, "
                f"w_norm={w_norm:.3f}, h_norm={h_norm:.3f}, "
                f"distance={distance:.2f}. "
                f"Apply the strict rules exactly."
            )

            res_2 = self.ask_llm(sys_2, user_2, max_new_tokens=128)
            print(f"🏁 [최종 VLM 결과]: {res_2}")

            action = None
            json_matches = re.findall(r"\{.*?\}", res_2, re.DOTALL)
            if json_matches:
                try:
                    action = json.loads(json_matches[-1])
                    print(f"✅ [파싱된 최종 액션]: {action}")
                    return action
                except json.JSONDecodeError:
                    print("⚠️ 최종 액션 JSON 파싱 실패")

            return {"status": "fail", "reason": "No valid action JSON found"}

        except Exception as e:
            print(f"❌ 테스트 중 에러 발생: {e}")
            return {"status": "error", "reason": str(e)}


if __name__ == "__main__":
    file_path = "test_cases.json"
    with open(file_path, "r", encoding="utf-8") as f:
        all_cases = json.load(f)
    
    case_id = "01"
    case = all_cases[case_id]

    tester = SeraphVLMTest()
    result = tester.run_test(case["image_path"], case["command"], distance=1.7)

    all_cases[case_id]["actual"] = result

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(all_cases, f, indent=2, ensure_ascii=False)

    print(f"📌 [최종 반환값 (result)]: {result}")
    print(f"🎯 [예상 결과값 (Expected)]: {case['expected']}")
