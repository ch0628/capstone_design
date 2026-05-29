"""
system.py — SeraphSystem

VLM + Depth만 사용. YOLO 없이 VLM이 의미 이해와 위치 탐지를 한 번에 처리한다.

run(image, command, depth_map, avoidance_attempts)
    -> {
        "status", "action", "context", "plan",
        "timings",    # vlm_ms, post_ms, plan_ms, exec_ms, total_ms
        "execution",  # executed_motions, next_action_hint, next_avoidance_attempts
        "debug_frame",# BBox 그려진 BGR numpy (main.py에서 send_to_pc로 전송)
    }
"""

import math
import json
import re
import cv2
import numpy as np
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


class SeraphSystem:
    def __init__(self):
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN이 .env 파일에 없습니다.")

        print("🚀 VLM(Qwen3.5-4B) 로딩 중...")
        vlm_id = "Qwen/Qwen3.5-4B"
        self.processor = AutoProcessor.from_pretrained(vlm_id, token=hf_token)
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            vlm_id,
            token=hf_token,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            attn_implementation="sdpa",
        )

        # ── 시각·계획 파라미터 ──────────────────────────────────
        self.hfov_deg = 70.0
        self.center_tolerance_px = 70
        self.distance_tolerance_m = 0.10
        self.obstacle_angle_corridor_deg = 20.0
        self.max_avoidance_attempts = 3
        self.target_distance_m = 0.3
        self.emergency_brake_distance_m = 0.3

        # ── 동작 파라미터 ───────────────────────────────────────
        self.avoid_safety_margin_deg = 15.0
        self.avoid_turn_min_deg = 10.0
        self.avoid_turn_max_deg = 30.0
        self.avoid_pass_buffer_m = 0.1
        self.max_forward_per_cycle_m = 0.4
        self.max_turn_per_cycle_deg = 20.0

        print("✅ SeraphSystem 준비 완료")
        print(f"   max_forward_per_cycle = {self.max_forward_per_cycle_m}m")
        print(f"   max_turn_per_cycle = {self.max_turn_per_cycle_deg}deg")

    # ── VLM ────────────────────────────────────────────────────

    def ask_vlm(self, image, system_prompt, user_text, max_new_tokens=256):
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

    def check_emergency_brake(self, depth_map):
        if depth_map is None:
            return False, 0.0

        h, w = depth_map.shape[:2]
        y1, y2 = int(h * 0.7), int(h * 0.95)
        x1, x2 = int(w * 0.35), int(w * 0.65)

        roi = depth_map[y1:y2, x1:x2]
        valid = roi[roi > 0]

        if valid.size == 0:
            return False, 0.0

        min_dist_m = float(np.min(valid)) / 1000.0

        if min_dist_m < self.emergency_brake_distance_m:
            return True, min_dist_m

        return False, min_dist_m

    def build_planning_prompt(self):
        pool_str = ", ".join(OBJECT_POOL)
        return f"""You are a perception and planning module for a home service robot.
Given a user command and the camera image, identify:
1) The object the user wants the robot to approach (target).
2) Any objects that could block the robot's path to the target (obstacles).
For each detected object, provide its bounding box in normalized coordinates [x1, y1, x2, y2] (values 0.0–1.0, origin at top-left).

ALLOWED CLASSES (use these only):
[{pool_str}]

KEY IDEA:
The same class can be a target in one command and an obstacle in another.
The user's intent decides the role.

STRICT RULES:
- target MUST be clearly visible in the image. If not visible, set target to null.
- obstacles lists objects physically between the robot and the target.
  - Do NOT include the target's class in obstacles.
  - If target is null, return [] for obstacles.
- bbox must be [x1, y1, x2, y2] in normalized coordinates (0.0–1.0).
- Pick classes ONLY from the allowed list above.
- Respond with ONE JSON object only. No explanation, no markdown.

Output format:
{{"intent": "<verb phrase>", "target": {{"class": "<class>", "bbox": [x1, y1, x2, y2]}}, "obstacles": [{{"class": "<class>", "bbox": [x1, y1, x2, y2]}}, ...]}}
If target not visible: {{"intent": "<verb phrase>", "target": null, "obstacles": []}}

Examples:
User: "I'm thirsty"
-> {{"intent": "drink", "target": {{"class": "bottle", "bbox": [0.35, 0.10, 0.55, 0.85]}}, "obstacles": [{{"class": "handbag", "bbox": [0.60, 0.20, 0.80, 0.75]}}]}}

User: "What time is it?"
-> {{"intent": "check_time", "target": {{"class": "clock", "bbox": [0.40, 0.15, 0.70, 0.60]}}, "obstacles": []}}

User: "Let's play"
-> {{"intent": "play", "target": {{"class": "sports ball", "bbox": [0.45, 0.50, 0.65, 0.90]}}, "obstacles": [{{"class": "cup", "bbox": [0.20, 0.30, 0.35, 0.70]}}]}}

User: "Help me find my phone"
-> {{"intent": "locate_phone", "target": null, "obstacles": []}}
"""

    def _extract_json(self, raw_text):
        """중첩 JSON을 올바르게 추출한다."""
        start = raw_text.find('{')
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False

        for i, ch in enumerate(raw_text[start:], start):
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw_text[start:i + 1])
                    except json.JSONDecodeError:
                        return None
        return None

    def _validate_bbox(self, bbox):
        if not (isinstance(bbox, list) and len(bbox) == 4):
            return False
        if not all(isinstance(v, (int, float)) for v in bbox):
            return False
        x1, y1, x2, y2 = bbox
        return 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0

    def parse_vlm_plan(self, raw_text):
        data = self._extract_json(raw_text)
        if data is None:
            print("⚠️ JSON 파싱 실패")
            return None

        pool_lower = [c.lower() for c in OBJECT_POOL]

        # target 파싱
        target_data = data.get("target")
        target = None
        if isinstance(target_data, dict):
            cls = target_data.get("class")
            bbox = target_data.get("bbox")
            if isinstance(cls, str) and cls.lower() not in ("null", "none", ""):
                cls = cls.lower()
                if cls in pool_lower and self._validate_bbox(bbox):
                    target = {"class": cls, "bbox": bbox}
                else:
                    if cls not in pool_lower:
                        print(f"⚠️ target '{cls}'이 OBJECT_POOL에 없음 → null 처리")
                    elif not self._validate_bbox(bbox):
                        print(f"⚠️ target bbox 유효하지 않음: {bbox} → null 처리")

        # obstacles 파싱
        obstacles = []
        for obs in data.get("obstacles", []):
            if not isinstance(obs, dict):
                continue
            cls = obs.get("class")
            bbox = obs.get("bbox")
            if not (isinstance(cls, str) and cls.lower() not in ("null", "none", "")):
                continue
            cls = cls.lower()
            if cls not in pool_lower:
                continue
            if target and cls == target["class"]:
                continue
            if self._validate_bbox(bbox):
                obstacles.append({"class": cls, "bbox": bbox})

        if not target:
            obstacles = []

        return {
            "intent": data.get("intent"),
            "target": target,
            "obstacles": obstacles,
        }

    # ── 시각화 ─────────────────────────────────────────────────

    def build_annotated_frame(self, pil_img, target_det, obstacle_dets, plan=None, action=None):
        """BBox가 그려진 BGR 프레임 (Local PC debug_frame_sender용)."""
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        for od in obstacle_dets:
            x1, y1, x2, y2 = [int(v) for v in od["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                od["class"],
                (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        if target_det:
            x1, y1, x2, y2 = [int(v) for v in target_det["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"TARGET {target_det['class']}",
                (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        if plan:
            target_info = plan.get("target")
            label = target_info["class"] if target_info else "no_target"
            intent = plan.get("intent") or ""
            cv2.putText(
                frame,
                f"VLM: {label} ({intent})",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
        if action:
            cv2.putText(
                frame,
                f"action: {action}",
                (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )

        return frame

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

        cx1 = max(0, min(cx - half_w, w - 1))
        cx2 = max(0, min(cx + half_w, w))
        cy1 = max(0, min(cy - half_h, h - 1))
        cy2 = max(0, min(cy + half_h, h))

        if cx2 <= cx1 or cy2 <= cy1:
            return None

        roi = depth_map[cy1:cy2, cx1:cx2]
        valid = roi[roi > 0]

        if valid.size == 0:
            return None

        return float(np.median(valid)) / 1000.0

    # ── 장애물 판정 ────────────────────────────────────────────

    def is_blocking_obstacle(self, obs_geom, obs_bbox, obs_distance,
                             target_geom, target_bbox, target_distance):
        if obs_distance >= target_distance:
            return False, "behind_target"

        ox1, _, ox2, _ = obs_bbox
        tx1, _, tx2, _ = target_bbox
        tw = tx2 - tx1
        bbox_x_overlap = (ox2 >= (tx1 - tw)) and (ox1 <= (tx2 + tw))

        angle_diff = abs(obs_geom["yaw_deg"] - target_geom["yaw_deg"])
        angle_in_corridor = angle_diff <= self.obstacle_angle_corridor_deg

        if not (bbox_x_overlap or angle_in_corridor):
            return False, "out_of_corridor"

        return True, "blocking"

    # ── 행동 결정 ──────────────────────────────────────────────

    def select_action(self, target_info, blocking_obstacles, avoidance_attempts, is_emergency=False):
        if is_emergency:
            return "emergency_stop"

        if not target_info["aligned"]:
            return "track"

        if blocking_obstacles and avoidance_attempts >= self.max_avoidance_attempts:
            return "wait_user"

        if blocking_obstacles:
            return "avoid_obstacle"

        distance_error = target_info["distance"] - self.target_distance_m
        if abs(distance_error) <= self.distance_tolerance_m:
            return "stop_at_target"

        return "track"

    # ── 모터 명령 (mock) ───────────────────────────────────────

    def _motor_turn(self, direction, amount_deg):
        cmd = f"TURN {direction.upper()} {amount_deg:.2f}deg"
        print(f"   🔧 [모터] {cmd}")
        return cmd

    def _motor_move(self, direction, distance_m):
        cmd = f"MOVE {direction.upper()} {distance_m:.3f}m"
        print(f"   🔧 [모터] {cmd}")
        return cmd

    def _motor_stop(self):
        cmd = "STOP"
        print(f"   🔧 [모터] {cmd}")
        return cmd

    # ── 행동별 실행 ────────────────────────────────────────────

    def _execute_track(self, context):
        target = context["target"]
        config = context["config"]
        executed = []

        if not target["aligned"]:
            yaw = target["yaw_deg"]
            direction = "left" if yaw < 0 else "right"
            turn_amount = min(abs(yaw), self.max_turn_per_cycle_deg)
            executed.append(self._motor_turn(direction, turn_amount))
            return executed, "continue"

        distance_error = target["distance_error"]
        tolerance = config["distance_tolerance_m"]

        if distance_error > tolerance:
            move_dist = min(distance_error, self.max_forward_per_cycle_m)
            executed.append(self._motor_move("front", move_dist))
            return executed, "continue"
        elif distance_error < -tolerance:
            move_dist = min(abs(distance_error), self.max_forward_per_cycle_m)
            executed.append(self._motor_move("back", move_dist))
            return executed, "continue"
        else:
            executed.append(self._motor_stop())
            return executed, "done"

    def _execute_avoid_obstacle(self, context):
        target = context["target"]
        blocking = context["blocking_obstacles"]
        img_w = context["image"]["width"]
        hfov_deg = context["image"]["hfov_deg"]
        executed = []

        most_blocking = min(blocking, key=lambda o: o["distance"])

        if most_blocking.get("depth_measurement") == "estimated_failed":
            obstacle_distance = 0.3
            print(
                f"   ↪ 회피 대상: {most_blocking['class']} "
                f"(depth 측정 실패 → 0.3m로 가정하고 회피)"
            )
        else:
            obstacle_distance = most_blocking["distance"]

            if obstacle_distance >= 0.3:
                move_dist = min(obstacle_distance - 0.3, self.max_forward_per_cycle_m)
                print(
                    f"   ↪ 회피 대상: {most_blocking['class']} "
                    f"(obstacle={obstacle_distance:.2f}m ≥ 0.3m → 직진 {move_dist:.2f}m로 거리 좁힘)"
                )
                executed.append(self._motor_move("front", move_dist))
                return executed, "reevaluate"

        if most_blocking["yaw_deg"] > target["yaw_deg"]:
            avoid_dir = "left"
        else:
            avoid_dir = "right"

        x1, _, x2, _ = most_blocking["bbox"]
        half_width_px = (x2 - x1) / 2
        hfov_rad = math.radians(hfov_deg)
        fx = (img_w / 2) / math.tan(hfov_rad / 2)
        half_angle_deg = math.degrees(math.atan(half_width_px / fx))

        raw_turn = half_angle_deg + self.avoid_safety_margin_deg
        clamped_turn = max(self.avoid_turn_min_deg, raw_turn)

        avoid_forward = min(
            obstacle_distance + self.avoid_pass_buffer_m,
            self.max_forward_per_cycle_m,
        )

        return_dir = "right" if avoid_dir == "left" else "left"

        print(
            f"   ↪ 회피 대상: {most_blocking['class']} "
            f"(obstacle={obstacle_distance:.2f}m, "
            f"turn raw={raw_turn:.2f}°, clamped={clamped_turn:.2f}°, "
            f"forward={avoid_forward:.2f}m, return turn {clamped_turn:.2f}°)"
        )

        executed.append(self._motor_turn(avoid_dir, clamped_turn))
        executed.append(self._motor_move("front", avoid_forward))
        executed.append(self._motor_turn(return_dir, clamped_turn))

        return executed, "reevaluate"

    def _execute_stop_at_target(self, context):
        executed = [self._motor_stop()]
        target = context["target"]
        distance = target.get("distance")
        if distance is None:
            print(f"   🎯 도착! target={target['class']}, distance=N/A (depth 측정 불가, 너무 가까움)")
        else:
            print(f"   🎯 도착! target={target['class']}, distance={distance:.2f}m")
        return executed, "done"

    def _execute_wait_user(self, context):
        executed = [self._motor_stop()]
        blocking = context.get("blocking_obstacles", [])
        if blocking:
            classes = list({o["class"] for o in blocking})
            msg = f"길에 {', '.join(classes)}이(가) 계속 막고 있어요. 치워주세요."
        else:
            msg = "사용자 도움이 필요합니다."
        print(f"   👤 [사용자 알림] {msg}")
        return executed, "wait_user"

    def _execute(self, action_command):
        """계획 결과를 받아 모터 실행."""
        action = action_command.get("action")
        status = action_command.get("status", "success")

        print(f"\n🤖 [실행] action='{action}'")

        if status == "abort" or action == "abort":
            self._motor_stop()
            return {
                "executed_motions": ["STOP"],
                "next_action_hint": "abort",
                "next_avoidance_attempts": 0,
            }

        if action == "retry":
            reason = action_command.get("reason", "unknown")
            print(f"   🔁 [retry] reason={reason}")
            return {
                "executed_motions": [],
                "next_action_hint": "retry",
                "next_action_reason": reason,
                "next_avoidance_attempts": 0,
            }

        context = action_command.get("context", {})
        current_attempts = context.get("avoidance_attempts", 0)

        if action == "emergency_stop":
            executed = [self._motor_stop()]
            print("🚨 [긴급 제동] 로봇 바로 앞에 장애물이 감지되어 정지합니다.")
            hint = "reevaluate"
            next_attempts = current_attempts
        elif action == "track":
            executed, hint = self._execute_track(context)
            next_attempts = 0
        elif action == "avoid_obstacle":
            executed, hint = self._execute_avoid_obstacle(context)
            next_attempts = current_attempts + 1
        elif action == "stop_at_target":
            executed, hint = self._execute_stop_at_target(context)
            next_attempts = 0
        elif action == "wait_user":
            executed, hint = self._execute_wait_user(context)
            next_attempts = current_attempts
        else:
            print(f"⚠️ 알 수 없는 action: {action}")
            self._motor_stop()
            executed = ["STOP"]
            hint = "abort"
            next_attempts = 0

        return {
            "executed_motions": executed,
            "next_action_hint": hint,
            "next_avoidance_attempts": next_attempts,
        }

    # ── 메인 진입점 ────────────────────────────────────────────

    def run(self, image, command, depth_map, avoidance_attempts=0):
        t_total_start = time.time()
        timings = {}
        action_command = {"status": "abort", "action": "abort", "reason": "unknown", "timings": timings}

        try:
            if isinstance(image, str):
                pil_img = Image.open(image).convert("RGB")
                src_label = f"'{image}'"
            else:
                pil_img = image.convert("RGB") if image.mode != "RGB" else image
                src_label = "<PIL Image>"

            img_w, img_h = pil_img.size
            print(
                f"\n📸 {src_label} "
                f"(명령: {command}, attempts={avoidance_attempts})"
            )

            is_emergency, min_dist = self.check_emergency_brake(depth_map)
            if is_emergency:
                print(f"🚨 [Emergency] 전방 {min_dist:.2f}m에 장애물 → emergency_stop")

            # ── VLM: 의미 이해 + 위치 탐지 ──────────────────────
            t_vlm = time.time()
            sys_prompt = self.build_planning_prompt()
            res_vlm = self.ask_vlm(pil_img, sys_prompt, command)
            plan = self.parse_vlm_plan(res_vlm)
            timings["vlm_ms"] = round((time.time() - t_vlm) * 1000, 2)
            print(f"📦 [VLM] {res_vlm}")

            if plan is None:
                action_command = {
                    "status": "retry", "action": "retry",
                    "reason": "vlm_parse_error",
                    "timings": timings,
                }
            else:
                target_plan = plan["target"]
                print(f"🧠 [Plan] target={target_plan}, obstacles={plan['obstacles']}")

                if not target_plan:
                    action_command = {
                        "status": "retry", "action": "retry",
                        "reason": "no_target",
                        "plan": plan, "timings": timings,
                    }
                else:
                    # ── normalized bbox → pixel bbox ──────────────
                    target_det = {
                        "class": target_plan["class"],
                        "bbox": [
                            target_plan["bbox"][0] * img_w,
                            target_plan["bbox"][1] * img_h,
                            target_plan["bbox"][2] * img_w,
                            target_plan["bbox"][3] * img_h,
                        ],
                    }
                    obstacle_dets = []
                    for obs in plan["obstacles"]:
                        obstacle_dets.append({
                            "class": obs["class"],
                            "bbox": [
                                obs["bbox"][0] * img_w,
                                obs["bbox"][1] * img_h,
                                obs["bbox"][2] * img_w,
                                obs["bbox"][3] * img_h,
                            ],
                        })

                    print(f"   🔍 [VLM detect] target=O, obstacles={len(obstacle_dets)}건")
                    print(f"   🎯 [VLM target] {target_det['class']} "
                          f"bbox={[round(v,1) for v in target_det['bbox']]}")
                    for od in obstacle_dets:
                        print(f"   🚧 [VLM obstacle] {od['class']} "
                              f"bbox={[round(v,1) for v in od['bbox']]}")

                    annotated_frame = self.build_annotated_frame(
                        pil_img, target_det, obstacle_dets, plan=plan
                    )

                    t_post = time.time()
                    target_geom = self.compute_geometry(target_det["bbox"], img_w, img_h)
                    target_distance = self._read_depth_at_bbox(depth_map, target_det["bbox"])

                    if target_distance is None:
                        aligned = target_geom["aligned"]
                        target_info = {
                            "class": target_det["class"],
                            "bbox": target_det["bbox"],
                            "cx": target_geom["cx"],
                            "yaw_deg": target_geom["yaw_deg"],
                            "aligned": aligned,
                            "distance": None,
                            "distance_error": None,
                        }
                        context = {
                            "target": target_info,
                            "obstacles": [],
                            "blocking_obstacles": [],
                            "image": {
                                "width": img_w,
                                "height": img_h,
                                "hfov_deg": self.hfov_deg,
                            },
                            "config": {
                                "target_distance_m": self.target_distance_m,
                                "distance_tolerance_m": self.distance_tolerance_m,
                                "center_tolerance_px": self.center_tolerance_px,
                            },
                            "avoidance_attempts": avoidance_attempts,
                        }
                        if aligned:
                            action = "stop_at_target"
                            print(f"📏 [Depth] target=N/A (정렬됨 → stop_at_target, 너무 가까움)")
                        else:
                            action = "track"
                            print(f"📏 [Depth] target=N/A (정렬 우선)")
                        timings["post_ms"] = round((time.time() - t_post) * 1000, 2)
                        action_command = {
                            "status": "success",
                            "action": action,
                            "context": context,
                            "plan": plan,
                            "timings": timings,
                            "debug_frame": annotated_frame,
                        }
                    else:
                        print(f"📏 [Depth] target={target_distance:.2f}m")

                        obstacles_info = []
                        blocking_obstacles = []

                        for od in obstacle_dets:
                            og = self.compute_geometry(od["bbox"], img_w, img_h)

                            obs_distance = self._read_depth_at_bbox(depth_map, od["bbox"])
                            depth_failed = obs_distance is None
                            if depth_failed:
                                obs_distance = max(0.01, target_distance - 0.01)

                            blocking, reason = self.is_blocking_obstacle(
                                og, od["bbox"], obs_distance,
                                target_geom, target_det["bbox"], target_distance,
                            )

                            ox1, _, ox2, _ = od["bbox"]
                            tx1, _, tx2, _ = target_det["bbox"]
                            tw = tx2 - tx1
                            x_overlap = (ox2 >= (tx1 - tw)) and (ox1 <= (tx2 + tw))

                            print(f"   🔍 [obstacle filter] {od['class']} "
                                  f"dist={obs_distance:.2f}m yaw={og['yaw_deg']:+.1f}° "
                                  f"x_overlap={'O' if x_overlap else 'X'} "
                                  f"→ {'BLOCKING' if blocking else f'skip({reason})'}")

                            entry = {
                                "class": od["class"],
                                "bbox": od["bbox"],
                                "yaw_deg": og["yaw_deg"],
                                "area_ratio": round(og["area_ratio"], 4),
                                "distance": round(obs_distance, 3),
                                "depth_measurement": "estimated_failed" if depth_failed else "measured",
                                "blocking": blocking,
                                "skip_reason": None if blocking else reason,
                            }
                            obstacles_info.append(entry)
                            if blocking:
                                blocking_obstacles.append(entry)

                        target_info = {
                            "class": target_det["class"],
                            "bbox": target_det["bbox"],
                            "cx": target_geom["cx"],
                            "yaw_deg": target_geom["yaw_deg"],
                            "aligned": target_geom["aligned"],
                            "distance": target_distance,
                            "distance_error": target_distance - self.target_distance_m,
                        }

                        action = self.select_action(
                            target_info, blocking_obstacles, avoidance_attempts, is_emergency
                        )

                        annotated_frame = self.build_annotated_frame(
                            pil_img, target_det, obstacle_dets, plan=plan, action=action
                        )

                        context = {
                            "target": target_info,
                            "obstacles": obstacles_info,
                            "blocking_obstacles": blocking_obstacles,
                            "image": {
                                "width": img_w,
                                "height": img_h,
                                "hfov_deg": self.hfov_deg,
                            },
                            "config": {
                                "target_distance_m": self.target_distance_m,
                                "distance_tolerance_m": self.distance_tolerance_m,
                                "center_tolerance_px": self.center_tolerance_px,
                            },
                            "avoidance_attempts": avoidance_attempts,
                        }

                        timings["post_ms"] = round((time.time() - t_post) * 1000, 2)
                        action_command = {
                            "status": "success",
                            "action": action,
                            "context": context,
                            "plan": plan,
                            "timings": timings,
                            "debug_frame": annotated_frame,
                        }

        except Exception as e:
            print(f"❌ 에러: {e}")
            action_command = {
                "status": "abort", "action": "abort", "reason": str(e),
                "timings": timings,
            }

        # ── 실행 단계 ──────────────────────────────────────────
        timings["plan_ms"] = round((time.time() - t_total_start) * 1000, 2)

        t_exec = time.time()
        execution = self._execute(action_command)
        timings["exec_ms"] = round((time.time() - t_exec) * 1000, 2)
        timings["total_ms"] = round((time.time() - t_total_start) * 1000, 2)

        return {
            **action_command,
            "execution": execution,
        }
