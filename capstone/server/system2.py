"""
System 2 — VisionPlanner

상위 의사결정 모듈.
이미지 + 자연어 명령을 받아서 "어떤 행동을 할지"만 결정한다.
구체적인 동작(회전 각도, 모터 제어 등)은 System 1의 책임.

행동 종류 (action_type):
  - "track"           : target을 향해 추적 (정렬+전진을 묶은 추상 행동)
  - "avoid_obstacle"  : 회피 동작 필요
  - "stop_at_target"  : 목표 거리 도달, 정지
  - "wait_user"       : 사용자 개입 대기 (회피 한도 초과 등)
  - "abort"           : 실패 (target 못 찾음, VLM 파싱 실패 등)

핵심 인터페이스:
    planner.plan(image_path, command, current_distance, avoidance_attempts)
        -> action_command dict {"status", "action", "context", ...}
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


# 가정환경에서 다룰 수 있는 COCO 클래스 풀.
# 같은 클래스가 명령에 따라 target도 되고 obstacle도 됨.
#
# [임시 제외] depth 카메라 연결 전까지 표면 역할 케이스 구분 불가:
#   - "dining table": target이 위에 있을 때 표면이고, 너머에 있을 때 장애물.
#     2D bbox만으론 이 둘을 원리적으로 구분 못 함. depth 연결 후 복원 예정.
OBJECT_POOL = [
    "chair", "couch", "bed", "toilet",
    "cup", "bottle", "wine glass", "bowl",
    "banana", "apple", "orange", "sandwich",
    "remote", "cell phone", "book", "laptop", "keyboard", "mouse",
    "handbag", "backpack", "suitcase", "scissors",
    "tv", "refrigerator", "oven", "microwave", "sink",
    "person", "potted plant",
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

        # 카메라 파라미터
        self.hfov_deg = 70.0

        # 정렬/거리 허용 오차 (System 1 판단에도 쓰여서 context로 전달)
        self.center_tolerance_px = 40
        self.distance_tolerance_m = 0.10

        # 장애물 판정 기준 (System 2 영역: 무엇이 막는가)
        self.obstacle_angle_corridor_deg = 15.0
        self.obstacle_min_area_ratio = 0.01

        # 회피 한도
        self.max_avoidance_attempts = 3

        # 시스템 설정
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
-> {{"intent": "drink", "target_object": "cup", "obstacle_classes": ["chair"]}}

User: "My legs hurt, I need to rest"
-> {{"intent": "rest", "target_object": "chair", "obstacle_classes": ["potted plant"]}}

User: "Pass me something to read"
-> {{"intent": "read", "target_object": "book", "obstacle_classes": ["couch"]}}

User: "I want to change the channel"
-> {{"intent": "control_tv", "target_object": "remote", "obstacle_classes": []}}

User: "Help me find my phone"
(no phone visible in scene)
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

    def detect_objects(self, pil_img, target_class, obstacle_classes):
        yolo_results = self.yolo.predict(pil_img, conf=0.05, verbose=False)

        target_class_lower = target_class.lower() if target_class else None
        obstacle_classes_lower = [c.lower() for c in obstacle_classes]

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
        """
        bbox 영역의 depth 값들 중 유효한 것(0이 아닌)의 median을 m 단위로 반환.
        depth_map은 RealSense raw (uint16, mm 단위) 가정.

        Returns:
            float: 거리(m). 유효 픽셀 없으면 None.
        """
        if depth_map is None:
            return None

        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = depth_map.shape[:2]

        # 좌표 클리핑 (이미지 경계 벗어남 방지)
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return None

        roi = depth_map[y1:y2, x1:x2]
        valid = roi[roi > 0]   # 0 = invalid (RealSense 측정 실패)

        if valid.size == 0:
            return None

        distance_mm = float(np.median(valid))
        return distance_mm / 1000.0   # mm → m

    # ── 장애물 판정 ────────────────────────────────────────────

    def is_blocking_obstacle(self, obs_geom, obs_distance, target_geom, target_distance):
        if obs_distance >= target_distance:
            return False, "behind_target"
        angle_diff = abs(obs_geom["yaw_deg"] - target_geom["yaw_deg"])
        if angle_diff > self.obstacle_angle_corridor_deg:
            return False, "out_of_corridor"
        if obs_geom["area_ratio"] < self.obstacle_min_area_ratio:
            return False, "too_small"
        return True, "blocking"

    # ── 행동 결정 (System 2의 핵심) ────────────────────────────

    def select_action(self, target_info, blocking_obstacles, avoidance_attempts):
        """
        상황을 보고 어떤 '종류'의 행동을 할지만 결정.
        구체적 수치 계산은 System 1에서 함.
        """
        # 1. 막는 장애물이 있고 회피 한도 초과 → 사용자 대기
        if blocking_obstacles and avoidance_attempts >= self.max_avoidance_attempts:
            return "wait_user"

        # 2. 막는 장애물이 있음 → 회피
        if blocking_obstacles:
            return "avoid_obstacle"

        # 3. 정렬됐고 적정 거리 도달 → 정지
        if target_info["aligned"]:
            distance_error = target_info["distance"] - self.target_distance_m
            if abs(distance_error) <= self.distance_tolerance_m:
                return "stop_at_target"

        # 4. 그 외 → 추적 (System 1이 정렬/전진 알아서 처리)
        return "track"

    # ── 메인 진입점 ────────────────────────────────────────────

    def plan(self, image, command, depth_map, avoidance_attempts=0):
        """
        이미지+명령 → action_command.

        Args:
            image: PIL.Image 객체 또는 이미지 파일 경로(str).
                   PIL이면 그대로 사용, str이면 로드.
            command: 자연어 사용자 명령
            depth_map: numpy 배열, shape (H, W), dtype uint16, mm 단위.
                       RealSense aligned depth. 0은 invalid.
            avoidance_attempts: 누적 회피 시도 횟수.

        Returns:
            action_command dict:
            {
                "status": "success" | "abort",
                "action": "track" | "avoid_obstacle" | "stop_at_target" | "wait_user" | "abort",
                "reason": <abort 시에만 존재. 실패 사유>,
                "context": {
                    "target": {...},        # System 1이 쓸 모든 raw 정보
                    "obstacles": [...],
                    "image": {...},
                    "config": {...},
                    "avoidance_attempts": int,
                },
                "plan": {...},              # VLM 출력 (디버깅용)
                "timings": {                # 단계별 소요 시간 (ms)
                    "vlm_ms": float,
                    "yolo_ms": float,
                    "post_ms": float,       # 기하 + depth 측정 + 결정 + 패키징
                    "total_ms": float,
                }
            }
        """
        t_total_start = time.time()
        timings = {}

        try:
            # 입력 정규화: PIL 객체 또는 경로 둘 다 받음
            if isinstance(image, str):
                pil_img = Image.open(image).convert("RGB")
                src_label = f"'{image}'"
            else:
                pil_img = image.convert("RGB") if image.mode != "RGB" else image
                src_label = "<PIL Image>"

            img_w, img_h = pil_img.size
            print(
                f"\n📸 [System2] {src_label} "
                f"(명령: {command}, attempts={avoidance_attempts})"
            )

            # 1. VLM
            t_vlm = time.time()
            sys_prompt = self.build_planning_prompt()
            res_vlm = self.ask_vlm(pil_img, sys_prompt, command)
            plan = self.parse_vlm_plan(res_vlm)
            timings["vlm_ms"] = round((time.time() - t_vlm) * 1000, 2)
            print(f"📦 [VLM] {res_vlm}")

            if plan is None:
                timings["total_ms"] = round((time.time() - t_total_start) * 1000, 2)
                return {
                    "status": "abort", "action": "abort",
                    "reason": "vlm_parse_error",
                    "timings": timings,
                }

            target_obj = plan["target_object"]
            obstacle_classes = plan["obstacle_classes"]
            print(f"🧠 [Plan] target={target_obj}, obstacles={obstacle_classes}")

            if not target_obj:
                timings["total_ms"] = round((time.time() - t_total_start) * 1000, 2)
                return {
                    "status": "abort",
                    "action": "abort",
                    "reason": "no_target_object",
                    "plan": plan,
                    "timings": timings,
                }

            # 2. YOLO
            t_yolo = time.time()
            target_det, obstacle_dets = self.detect_objects(
                pil_img, target_obj, obstacle_classes
            )
            timings["yolo_ms"] = round((time.time() - t_yolo) * 1000, 2)

            if target_det is None:
                timings["total_ms"] = round((time.time() - t_total_start) * 1000, 2)
                return {
                    "status": "abort",
                    "action": "abort",
                    "reason": f"target_not_found:{target_obj}",
                    "plan": plan,
                    "timings": timings,
                }

            print(f"🎯 [YOLO target] {target_det['class']} conf={target_det['conf']:.3f}")
            print(f"🚧 [YOLO obstacles] {len(obstacle_dets)}건")

            # 3. 기하 + depth + 결정 (= post-processing)
            t_post = time.time()
            target_geom = self.compute_geometry(target_det["bbox"], img_w, img_h)

            # 3-1. Target 거리 측정 (depth로)
            target_distance = self._read_depth_at_bbox(depth_map, target_det["bbox"])
            if target_distance is None:
                # target depth 측정 실패 → 거리를 모르니 진행 불가
                timings["post_ms"] = round((time.time() - t_post) * 1000, 2)
                timings["total_ms"] = round((time.time() - t_total_start) * 1000, 2)
                return {
                    "status": "abort",
                    "action": "abort",
                    "reason": "target_depth_unavailable",
                    "plan": plan,
                    "timings": timings,
                }
            print(f"📏 [Depth] target={target_distance:.2f}m")

            # 3-2. 장애물 판정 (각 obstacle 거리도 depth로)
            obstacles_info = []
            blocking_obstacles = []

            for od in obstacle_dets:
                og = self.compute_geometry(od["bbox"], img_w, img_h)

                # depth 측정 시도
                obs_distance = self._read_depth_at_bbox(depth_map, od["bbox"])
                depth_failed = obs_distance is None
                if depth_failed:
                    # 측정 실패 → 보수적으로 가까이 있다고 가정 (target보다 가깝게)
                    # is_blocking_obstacle의 거리 조건을 통과하도록 처리
                    obs_distance = max(0.01, target_distance - 0.01)

                blocking, reason = self.is_blocking_obstacle(
                    og, obs_distance, target_geom, target_distance
                )

                entry = {
                    "class": od["class"],
                    "conf": round(od["conf"], 3),
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

            # 4. target context 정리
            target_info = {
                "class": target_det["class"],
                "conf": round(target_det["conf"], 3),
                "bbox": target_det["bbox"],
                "cx": target_geom["cx"],
                "yaw_deg": target_geom["yaw_deg"],
                "aligned": target_geom["aligned"],
                "distance": target_distance,
                "distance_error": target_distance - self.target_distance_m,
            }

            # 5. 행동 종류 결정 (System 2의 핵심 의사결정)
            action = self.select_action(
                target_info, blocking_obstacles, avoidance_attempts
            )

            # 6. context 패키징 (System 1이 쓸 모든 raw 정보)
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
            timings["total_ms"] = round((time.time() - t_total_start) * 1000, 2)

            return {
                "status": "success",
                "action": action,
                "context": context,
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