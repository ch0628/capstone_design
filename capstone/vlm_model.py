import json
import re
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from ultralytics import YOLO


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

    def run_test(self, image_path, command):
        try:
            pil_img = Image.open(image_path).convert("RGB")
            print(f"\n📸 이미지 '{image_path}' 분석 시작 (명령: {command})")

            # STEP 1
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

            # STEP 2
            print(f"🔍 [YOLO] '{target_obj}' 찾는 중...")
            yolo_results = self.yolo.predict(pil_img, conf=0.25, verbose=False)

            bbox = None
            for r in yolo_results:
                for box in r.boxes:
                    label = r.names[int(box.cls[0])].lower()
                    if label == target_obj.lower():
                        bbox = box.xyxy[0].tolist()
                        break
                if bbox is not None:
                    break

            print(f"🎯 [YOLO 결과]: {bbox}")

            if bbox is None:
                print(f"❌ [YOLO] 이미지 내에서 '{target_obj}'를 물리적으로 특정할 수 없습니다.")
                return {"move": "stop", "status": f"Target '{target_obj}' not found"}

            img_w, img_h = pil_img.size
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            cx_norm = cx / img_w
            cy_norm = cy / img_h
            w_norm = (x2 - x1) / img_w
            h_norm = (y2 - y1) / img_h

            # STEP 3
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

Decision rules:
- If the target is left of center, choose "left".
- If the target is right of center, choose "right".
- If the target is near the horizontal center:
  - choose "front" if the target is far.
  - choose "back" if the target is too close.
  - choose "stop" if the target is centered and at a good distance.
"""
            user_2 = (
                f"Target: {target_obj}. "
                f"Center(normalized): ({cx_norm:.3f}, {cy_norm:.3f}). "
                f"Size(normalized): ({w_norm:.3f}, {h_norm:.3f}). "
                f"Distance: 1.2m. "
                f"Decide only one move from [front, back, left, right, stop]."
            )

            res_2 = self.ask_vlm(pil_img, sys_2, user_2, max_new_tokens=128)
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
    tester = SeraphVLMTest()
    result = tester.run_test("test.jpg", "나 목말라.")
    print(f"📌 [최종 반환값]: {result}")