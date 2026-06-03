"""
Step 1 — System 2 (VisionPlanner)

Configuration : VLM + RGB (Depth 없음)
Capability    : VLM이 명령 해석, 이미지만으로 거리 추정, VLM이 모터 명령 직접 생성
Failure Case  : VLM 시각적 거리 추정 부정확 → 잘못된 정지 거리
"""

import json
import re
import math
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from dotenv import load_dotenv
import os
import time


OBJECT_POOL = [
    "bottle",
    "cup",
    "clock",
    "sports ball",
    "car",
    "handbag",
]


class VisionPlanner:
    def __init__(self):
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN이 .env 파일에 없습니다.")

        print("🚀 [System2] VLM(Qwen3.5-4B) 로딩 중...")
        vlm_id = "Qwen/Qwen3.5-4B"
        self.processor = AutoProcessor.from_pretrained(vlm_id, token=hf_token)
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            vlm_id,
            token=hf_token,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            attn_implementation="sdpa",
        )

        self.hfov_deg = 70.0
        self.center_tolerance_px = 70
        self.target_distance_m = 0.3

        print("✅ System2(VisionPlanner) 준비 완료")

    # ── VLM ────────────────────────────────────────────────────

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
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
        inputs = self.processor(
            text=[prompt], images=[image], return_tensors="pt",
        ).to(self.vlm.device)
        outputs = self.vlm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        return self.processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True,
        )

    def build_planning_prompt(self):
        pool_str = ", ".join(OBJECT_POOL)
        return f"""You are a planning and control module for a home service robot.
Given a user command and the camera image, decide:
1) Which object the user wants the robot to approach (target_object).
2) The bounding box of that object in the image (bbox_norm).
3) Your visual estimate of the robot's distance to the target (estimated_distance_m).
4) Which objects may block the path to the target (obstacles).
5) The motor commands the robot should execute (motor_commands).

[CRITICAL] DO NOT copy the exact numbers from the examples. Calculate dynamic values for distance and angle based on the current scene.

ALLOWED CLASSES (use these only):
[{pool_str}]

STRICT RULES:
- target_object MUST be clearly visible in the image. If no suitable object is visible, set target_object to null.
- bbox_norm: normalized bounding box of target (0.0 ~ 1.0). null if not visible.
- estimated_distance_m: your visual estimate (in meters) of robot's distance to target.
  Use object size, perspective, and position in frame as cues. null if target not visible.
- obstacles: list of objects blocking the path between robot and target.
  Each entry: "class" (from allowed list), "estimated_distance_m" (visual estimate), "yaw_deg" (positive=right, negative=left).
  Do NOT include the target's own class. Empty list if none or target is null.
- motor_commands: list of motor commands to move robot toward target while avoiding obstacles.
  GOAL: Approach the target until 0.3m from it. Avoid hitting any obstacles on the way.
  Available commands: "STOP", "TURN LEFT Xdeg", "TURN RIGHT Xdeg" (X=5~30), "MOVE FRONT Xm" (X=0.05~0.40)
  You may output a SEQUENCE when needed (e.g. avoidance: [turn aside, move forward, turn back]).
  Simple actions use one command. null if target not visible.
- Pick classes ONLY from the allowed list above.
- Respond with ONE JSON object only. No explanation, no markdown.

Output format:
{{"intent": "<short verb phrase>", "target_object": "<class or null>",
 "bbox_norm": [x1, y1, x2, y2],
 "estimated_distance_m": 1.2,
 "obstacles": [{{"class": "<class>", "estimated_distance_m": 0.4, "yaw_deg": -10.0}}],
 "motor_commands": ["<cmd1>", "<cmd2>"]}}

Examples:
User: "I'm thirsty"
-> {{"intent": "drink", "target_object": "bottle", "bbox_norm": [0.3, 0.2, 0.6, 0.8], "estimated_distance_m": 0.8, "obstacles": [], "motor_commands": ["MOVE FRONT 0.50m"]}}

User: "What time is it?"
(handbag visible on left side, blocking path to clock)
-> {{"intent": "check_time", "target_object": "clock", "bbox_norm": [0.6, 0.0, 0.9, 0.5], "estimated_distance_m": 1.5, "obstacles": [{{"class": "handbag", "estimated_distance_m": 0.4, "yaw_deg": -8.0}}], "motor_commands": ["TURN RIGHT 15deg", "MOVE FRONT 0.40m", "TURN LEFT 15deg"]}}

User: "Help me find my phone"
(no phone visible in scene, and phone is not in allowed classes)
-> {{"intent": "locate_phone", "target_object": null, "bbox_norm": null, "estimated_distance_m": null, "obstacles": [], "motor_commands": null}}
"""

    def parse_vlm_plan(self, raw_text):
        json_matches = re.findall(r"\{.*?\}", raw_text, re.DOTALL)
        if not json_matches:
            return None

        try:
            data = json.loads(json_matches[-1])
        except json.JSONDecodeError:
            print("⚠️ JSON 파싱 실패")
            return None

        target = data.get("target_object")
        if isinstance(target, str) and target.lower() in ("null", "none", ""):
            target = None
        if isinstance(target, str):
            target = target.lower()
            if target not in [c.lower() for c in OBJECT_POOL]:
                print(f"⚠️ target '{target}'이 OBJECT_POOL에 없음 → null 처리")
                target = None

        bbox_norm = data.get("bbox_norm")
        if isinstance(bbox_norm, list) and len(bbox_norm) == 4:
            if all(isinstance(v, (int, float)) and 0.0 <= v <= 1.0 for v in bbox_norm):
                parsed_bbox_norm = bbox_norm
            else:
                parsed_bbox_norm = None
        else:
            parsed_bbox_norm = None

        estimated_distance_m = data.get("estimated_distance_m")
        if isinstance(estimated_distance_m, (int, float)) and estimated_distance_m > 0:
            parsed_distance = float(estimated_distance_m)
        else:
            parsed_distance = None

        raw_motor_commands = data.get("motor_commands")
        if isinstance(raw_motor_commands, list) and raw_motor_commands:
            motor_commands = [c for c in raw_motor_commands if isinstance(c, str)]
            if not motor_commands:
                motor_commands = None
        else:
            motor_commands = None

        raw_obstacles = data.get("obstacles", [])
        parsed_obstacles = []
        if isinstance(raw_obstacles, list):
            pool_lower = [c.lower() for c in OBJECT_POOL]
            for item in raw_obstacles:
                if not isinstance(item, dict):
                    continue
                cls = item.get("class", "")
                if isinstance(cls, str):
                    cls = cls.lower()
                if cls not in pool_lower or cls == target:
                    continue
                dist = item.get("estimated_distance_m")
                yaw = item.get("yaw_deg")
                if isinstance(dist, (int, float)) and dist > 0 and isinstance(yaw, (int, float)):
                    parsed_obstacles.append({
                        "class": cls,
                        "estimated_distance_m": float(dist),
                        "yaw_deg": float(yaw),
                    })

        return {
            "intent": data.get("intent"),
            "target_object": target,
            "bbox_norm": parsed_bbox_norm,
            "estimated_distance_m": parsed_distance,
            "obstacles": parsed_obstacles,
            "motor_commands": motor_commands,
        }

    # ── 기하 ───────────────────────────────────────────────────

    def compute_geometry(self, bbox, img_w, img_h):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        yaw_deg, pixel_offset, aligned = self.compute_yaw_to_center(cx, img_w)
        area_ratio = ((x2 - x1) * (y2 - y1)) / (img_w * img_h)

        return {
            "cx": cx, "cy": cy,
            "yaw_deg": yaw_deg,
            "pixel_offset": pixel_offset,
            "aligned": aligned,
            "area_ratio": area_ratio,
        }

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

    # ── 메인 진입점 ────────────────────────────────────────────

    def plan(self, image, command, depth_map=None):
        t_total_start = time.time()
        timings = {}

        try:
            if isinstance(image, str):
                pil_img = Image.open(image).convert("RGB")
                src_label = f"'{image}'"
            else:
                pil_img = image.convert("RGB") if image.mode != "RGB" else image
                src_label = "<PIL Image>"

            img_w, img_h = pil_img.size
            print(f"\n📸 [System2] {src_label} (명령: {command})")

            t_vlm = time.time()
            sys_prompt = self.build_planning_prompt()
            res_vlm = self.ask_vlm(pil_img, sys_prompt, command)
            plan = self.parse_vlm_plan(res_vlm)
            timings["vlm_ms"] = round((time.time() - t_vlm) * 1000, 2)
            print(f"📦 [VLM] {res_vlm}")

            if plan is None:
                timings["total_ms"] = round((time.time() - t_total_start) * 1000, 2)
                return {
                    "status": "retry", "action": "retry",
                    "reason": "vlm_parse_error",
                    "timings": timings,
                }

            target_obj = plan["target_object"]
            bbox_norm = plan["bbox_norm"]
            target_distance = plan["estimated_distance_m"]
            motor_commands = plan["motor_commands"]
            obstacles = plan.get("obstacles", [])
            print(f"🧠 [Plan] target={target_obj}, bbox={bbox_norm}, dist={target_distance}m, cmds={motor_commands}")
            print(f"   🚧 [obstacles] {len(obstacles)}건")
            for o in obstacles:
                print(f"      - {o['class']}: dist={o['estimated_distance_m']}m, yaw={o['yaw_deg']}°")

            if not target_obj:
                timings["total_ms"] = round((time.time() - t_total_start) * 1000, 2)
                return {
                    "status": "retry", "action": "retry",
                    "reason": "no_target",
                    "plan": plan, "timings": timings,
                }

            if bbox_norm is None or target_distance is None or motor_commands is None:
                timings["total_ms"] = round((time.time() - t_total_start) * 1000, 2)
                return {
                    "status": "retry", "action": "retry",
                    "reason": "vlm_estimation_unavailable",
                    "plan": plan, "timings": timings,
                }

            t_post = time.time()

            first_cmd = motor_commands[0].upper()
            if first_cmd.startswith("STOP"):
                action = "stop"
                next_hint = "done"
            elif first_cmd.startswith("TURN"):
                action = "turn"
                next_hint = "continue"
            else:
                action = "move"
                next_hint = "continue"

            timings["post_ms"] = round((time.time() - t_post) * 1000, 2)
            timings["total_ms"] = round((time.time() - t_total_start) * 1000, 2)

            return {
                "status": "success",
                "action": action,
                "motor_commands": motor_commands,
                "next_hint": next_hint,
                "plan": plan,
                "timings": timings,
            }

        except Exception as e:
            print(f"❌ [System2] 에러: {e}")
            timings["total_ms"] = round((time.time() - t_total_start) * 1000, 2)
            return {
                "status": "abort", "action": "abort", "reason": str(e),
                "timings": timings,
            }
