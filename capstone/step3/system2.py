"""
Step 3 — System 2 (VisionPlanner)

Configuration : VLM + YOLO + RGB-D (System 1/2 계층 없음)
Capability    : YOLO 정확한 BBox + Depth 기반 거리, VLM이 모터 명령 생성
Failure Case  : VLM flat 구조 명령 생성 — 장애물 회피·긴급 제동·상태 관리 불가
"""

import json
import re
import math
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from ultralytics import YOLO
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

        print("🚀 [System2 1/2] VLM(Qwen3.5-4B) 로딩 중...")
        vlm_id = "Qwen/Qwen3.5-4B"
        self.processor = AutoProcessor.from_pretrained(vlm_id, token=hf_token)
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            vlm_id,
            token=hf_token,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            attn_implementation="sdpa",
        )

        print("🚀 [System2 2/2] YOLOv11 엔진 로딩 중...")
        self.yolo = YOLO("yolo11n.pt")

        self.hfov_deg = 70.0
        self.center_tolerance_px = 70
        self.distance_tolerance_m = 0.10
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

    def build_motor_prompt(self, target_class, target_distance_m, target_yaw_deg,
                           target_aligned, obstacles_info):
        if obstacles_info:
            obs_lines = [
                f"- {o['class']}: distance={o['distance']:.2f}m, yaw={o['yaw_deg']:.2f}deg"
                for o in obstacles_info
            ]
            obs_str = "OBSTACLES IN SCENE:\n" + "\n".join(obs_lines)
        else:
            obs_str = "OBSTACLES: none"

        return f"""You are controlling a home service robot.

[CRITICAL] DO NOT copy the exact numbers from the examples. Calculate dynamic values for distance and angle based on the current scene.

GOAL: Approach the {target_class} until 0.3m from it. Avoid hitting any obstacles on the way.

CURRENT STATE:
- Target distance: {target_distance_m:.2f}m
- Target yaw: {target_yaw_deg:.2f}deg (negative=left, positive=right)
- Target aligned: {target_aligned}
{obs_str}

MOTOR COMMANDS:
- "STOP"
- "TURN LEFT Xdeg" / "TURN RIGHT Xdeg" (X = 5 to 30)
- "MOVE FRONT Xm" (X = 0.05 to 0.40)

You may output a SEQUENCE of commands when needed.
- Simple actions (align toward target, move forward, stop): one command
- Avoidance: a sequence like [turn aside, move forward, turn back]

After your commands execute, the robot will take a new image and ask you again.

Respond with JSON only:
{{"motor_commands": ["<cmd1>", "<cmd2>", ...]}}

Examples:
{{"motor_commands": ["STOP"]}}
{{"motor_commands": ["MOVE FRONT 0.30m"]}}
{{"motor_commands": ["TURN LEFT 12deg"]}}
{{"motor_commands": ["TURN RIGHT 20deg", "MOVE FRONT 0.30m", "TURN LEFT 20deg"]}}
"""

    def build_planning_prompt(self):
        pool_str = ", ".join(OBJECT_POOL)
        return f"""You are a planning module for a home service robot.
Given a user command and the camera image, decide:
1) Which object the user wants the robot to approach (target_object).
2) Which other objects in the scene could block the robot's path on the way to the target (obstacle_classes).

ALLOWED CLASSES (use these only, both for target and obstacles):
[{pool_str}]

KEY IDEA:
The same class can be a target in one command and an obstacle in another, depending on user intent.
The user's intent decides the role.

STRICT RULES:
- target_object MUST be clearly visible in the image. If no suitable object is visible, set target_object to null.
- obstacle_classes lists class names that appear in the image AND are physically positioned between the robot and the target.
  - Do NOT include the target's own class in obstacle_classes.
  - If target_object is null, also return [] for obstacle_classes.
- Pick classes ONLY from the allowed list above.
- Respond with ONE JSON object only. No explanation, no markdown.

Output format:
{{"intent": "<short verb phrase>", "target_object": "<class or null>", "obstacle_classes": ["<class>", ...]}}

Examples:
User: "I'm thirsty"
-> {{"intent": "drink", "target_object": "bottle", "obstacle_classes": ["handbag"]}}

User: "What time is it?"
-> {{"intent": "check_time", "target_object": "clock", "obstacle_classes": []}}

User: "Let's play"
-> {{"intent": "play", "target_object": "sports ball", "obstacle_classes": ["cup"]}}

User: "Bring me my bag"
-> {{"intent": "fetch_bag", "target_object": "handbag", "obstacle_classes": []}}

User: "Help me find my phone"
(no phone visible in scene, and phone is not in allowed classes)
-> {{"intent": "locate_phone", "target_object": null, "obstacle_classes": []}}
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

        obstacles = data.get("obstacle_classes", [])
        if not isinstance(obstacles, list):
            obstacles = []
        pool_lower = [c.lower() for c in OBJECT_POOL]
        obstacles = list({
            o.lower() for o in obstacles
            if isinstance(o, str) and o.lower() in pool_lower
        })
        if target and target in obstacles:
            obstacles.remove(target)
        if not target:
            obstacles = []

        return {
            "intent": data.get("intent"),
            "target_object": target,
            "obstacle_classes": obstacles,
        }

    # ── YOLO ───────────────────────────────────────────────────

    def detect_objects(self, pil_img, target_class, obstacle_classes=None):
        yolo_results = self.yolo.predict(pil_img, conf=0.05, verbose=False)

        target_class_lower = target_class.lower() if target_class else None
        obstacle_classes_lower = [c.lower() for c in (obstacle_classes or [])]

        best_target = None
        best_target_conf = -1.0
        obstacle_dets = []

        for r in yolo_results:
            for box in r.boxes:
                label = r.names[int(box.cls[0])].lower()
                conf = float(box.conf[0])
                bbox = box.xyxy[0].tolist()

                if target_class_lower and label == target_class_lower:
                    if conf > best_target_conf:
                        best_target_conf = conf
                        best_target = {"class": label, "conf": conf, "bbox": bbox}
                elif label in obstacle_classes_lower:
                    obstacle_dets.append({"class": label, "conf": conf, "bbox": bbox})

        return best_target, obstacle_dets

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

    def _read_depth_at_bbox(self, depth_map, bbox):
        if depth_map is None:
            return None

        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = depth_map.shape[:2]

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half_w = max(1, (x2 - x1) // 4)
        half_h = max(1, (y2 - y1) // 4)

        cx1 = cx - half_w
        cx2 = cx + half_w
        cy1 = cy - half_h
        cy2 = cy + half_h

        cx1 = max(0, min(cx1, w - 1))
        cx2 = max(0, min(cx2, w))
        cy1 = max(0, min(cy1, h - 1))
        cy2 = max(0, min(cy2, h))

        if cx2 <= cx1 or cy2 <= cy1:
            return None

        roi = depth_map[cy1:cy2, cx1:cx2]
        valid = roi[roi > 0]

        if valid.size == 0:
            return None

        return float(np.median(valid)) / 1000.0

    # ── 메인 진입점 ────────────────────────────────────────────

    def plan(self, image, command, depth_map):
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
            print(f"🧠 [Plan] target={target_obj}")

            if not target_obj:
                timings["total_ms"] = round((time.time() - t_total_start) * 1000, 2)
                return {
                    "status": "retry", "action": "retry",
                    "reason": "no_target",
                    "plan": plan, "timings": timings,
                }

            obstacle_classes = plan.get("obstacle_classes", [])

            t_yolo = time.time()
            target_det, obstacle_dets = self.detect_objects(pil_img, target_obj, obstacle_classes)
            timings["yolo_ms"] = round((time.time() - t_yolo) * 1000, 2)

            print(f"   🔍 [YOLO] target_det={'O' if target_det else 'X'}, obstacles={len(obstacle_dets)}건")
            if target_det:
                print(f"   🎯 [YOLO target] bbox={[round(v,1) for v in target_det['bbox']]} "
                      f"conf={target_det['conf']:.3f}")

            if target_det is None:
                timings["total_ms"] = round((time.time() - t_total_start) * 1000, 2)
                return {
                    "status": "retry", "action": "retry",
                    "reason": "target_not_found",
                    "plan": plan, "timings": timings,
                }

            t_post = time.time()
            target_geom = self.compute_geometry(target_det["bbox"], img_w, img_h)
            target_distance = self._read_depth_at_bbox(depth_map, target_det["bbox"])

            if target_distance is None:
                target_distance = 0.3
                print(f"📏 [Depth] target=N/A (depth 측정 실패 → 0.3m 가정)")
            else:
                print(f"📏 [Depth] target={target_distance:.2f}m")

            obstacles_info = []
            for od in obstacle_dets:
                og = self.compute_geometry(od["bbox"], img_w, img_h)
                obs_dist = self._read_depth_at_bbox(depth_map, od["bbox"])
                if obs_dist is None:
                    continue
                x1, _, x2, _ = od["bbox"]
                obstacles_info.append({
                    "class": od["class"],
                    "distance": round(obs_dist, 2),
                    "yaw_deg": round(og["yaw_deg"], 2),
                    "bbox_width_px": int(x2 - x1),
                })

            print(f"   🚧 [obstacles] {len(obstacles_info)}건")
            for o in obstacles_info:
                print(f"      - {o['class']}: dist={o['distance']}m, yaw={o['yaw_deg']}°, w={o['bbox_width_px']}px")

            # VLM이 모터 명령 직접 생성
            motor_prompt = self.build_motor_prompt(
                target_obj, target_distance,
                target_geom["yaw_deg"], target_geom["aligned"],
                obstacles_info,
            )
            raw_motor = self.ask_vlm(pil_img, motor_prompt, "", max_new_tokens=128)
            print(f"🤖 [VLM motor] {raw_motor}")

            motor_commands = None
            motor_matches = re.findall(r"\{.*?\}", raw_motor, re.DOTALL)
            if motor_matches:
                try:
                    motor_data = json.loads(motor_matches[-1])
                    motor_commands = motor_data.get("motor_commands")
                    if not isinstance(motor_commands, list) or not motor_commands:
                        motor_commands = None
                    else:
                        motor_commands = [c for c in motor_commands if isinstance(c, str)]
                        if not motor_commands:
                            motor_commands = None
                except json.JSONDecodeError:
                    pass

            if motor_commands is None:
                timings["post_ms"] = round((time.time() - t_post) * 1000, 2)
                timings["total_ms"] = round((time.time() - t_total_start) * 1000, 2)
                return {
                    "status": "retry", "action": "retry",
                    "reason": "vlm_motor_parse_error",
                    "plan": plan, "timings": timings,
                }

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
