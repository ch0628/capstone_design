import json
import re
import math
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from ultralytics import YOLO
from dotenv import load_dotenv
import os


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

        # Intel RealSense RGB horizontal FOV
        self.hfov_deg = 70.0

        # 허용 오차
        self.center_tolerance_px = 40
        self.distance_tolerance_m = 0.10

        # 테스트용 상수값
        self.target_distance_m = 1.0   # 도달해야 하는 목표 거리
        self.current_distance_m = 1.7  # 현재 거리(임시 테스트용, 나중에 depth값으로 대체)

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

    def compute_yaw_to_center(self, cx, img_w):
        img_center_x = img_w / 2
        pixel_offset = cx - img_center_x

        if abs(pixel_offset) <= self.center_tolerance_px:
            return 0.0, pixel_offset, True

        hfov_rad = math.radians(self.hfov_deg)
        fx = (img_w / 2) / math.tan(hfov_rad / 2)

        yaw_rad = math.atan(pixel_offset / fx)
        yaw_deg = math.degrees(yaw_rad)

        return yaw_deg, pixel_offset, False

    def decide_linear_action(self, distance):
        distance_error = distance - self.target_distance_m

        if distance_error > self.distance_tolerance_m:
            return "front", distance_error
        elif distance_error < -self.distance_tolerance_m:
            return "back", abs(distance_error)
        else:
            return "stop", 0.0

    def run_test(self, image_path, command):
        try:
            distance = self.current_distance_m

            pil_img = Image.open(image_path).convert("RGB")
            print(
                f"\n📸 이미지 '{image_path}' 분석 시작 "
                f"(명령: {command}, distance={distance:.2f}m, target_distance={self.target_distance_m:.2f}m)"
            )

            sys_1 = """
Return only one JSON object.
No explanation.

The chosen target_object MUST be clearly visible in the provided image.
Do not guess or assume objects that are outside the camera view.

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
                return {"status": "fail", "reason": f"Target '{target_obj}' not found"}

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

            yaw_deg, pixel_offset, aligned = self.compute_yaw_to_center(cx, img_w)

            if yaw_deg < 0:
                turn_direction = "left"
            elif yaw_deg > 0:
                turn_direction = "right"
            else:
                turn_direction = "center"

            if aligned:
                move, linear_distance = self.decide_linear_action(distance)
            else:
                move, linear_distance = "hold", 0.0

            result = {
                "status": "success",
                "target": target_obj,
                "bbox_center_x": round(cx, 2), # 바운딩박스 중심 x좌표
                "image_center_x": round(img_w / 2, 2), # 이미지 자체의 중앙 좌표
                "pixel_offset": round(pixel_offset, 2), # 객체와 화면 중심 간의 차이
                "yaw_deg": round(yaw_deg, 2), # 로봇이 회전해야 하는 각도
                "turn_direction": turn_direction, # 회전 방향
                "aligned": aligned, # 정렬이 되었는지
                "distance": round(distance, 3), # 객체까지 거리값
                "target_distance": round(self.target_distance_m, 3), # 목표 거리
                "distance_error": round(distance - self.target_distance_m, 3), # 거리 차이
                "move": move, # front/back -> 앞뒤로 이동, stop -> 적절한 거리, hold -> 정렬이 안 끝나서 이동 보류
                "linear_distance": round(linear_distance, 3), # 실제 이동해야하는 거리
            }

            #print(f"✅ [최종 제어 결과]: {result}")
            return result

        except Exception as e:
            print(f"❌ 테스트 중 에러 발생: {e}")
            return {"status": "error", "reason": str(e)}


if __name__ == "__main__":
    file_path = "test_cases.json"
    with open(file_path, "r", encoding="utf-8") as f:
        all_cases = json.load(f)

    case_id = "02"
    case = all_cases[case_id]

    tester = SeraphVLMTest()

    result = tester.run_test(
        case["image_path"],
        case["command"],
    )

    all_cases[case_id]["actual"] = result

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(all_cases, f, indent=2, ensure_ascii=False)

    print(f"📌 [최종 반환값 (result)]: {result}")
    print(f"🎯 [예상 결과값 (Expected)]: {case['expected']}")