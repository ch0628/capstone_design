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
  - "emergency_stop"  : 긴급 제동 (충돌 위험)
  - "retry"           : 일시적 실패, 다음 프레임에서 재시도
  - "abort"           : 시스템 에러 (예외 발생)

핵심 인터페이스:
    planner.plan(image_path, command, depth_map, avoidance_attempts)
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
OBJECT_POOL = [
    "chair", "couch", "bed", "toilet",
    "cup", "bottle", "wine glass", "bowl",
    "banana", "apple", "orange", "sandwich",
    "remote", "cell phone", "book", "laptop", "keyboard", "mouse",
    "handbag", "backpack", "suitcase", "scissors",
    "tv", "refrigerator", "oven", "microwave", "sink",
    "person", "potted plant", "cap", "hat"
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

        # 정렬/거리 허용 오차
        self.center_tolerance_px = 40
        self.distance_tolerance_m = 0.10

        # 장애물 판정 기준
        self.obstacle_angle_corridor_deg = 15.0
        self.obstacle_min_area_ratio = 0.01

        # 회피 한도
        self.max_avoidance_attempts = 3

        # 시스템 설정
        self.target_distance_m = 0.3
        self.emergency_brake_distance_m = 0.3

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

    def check_emergency_brake(self, depth_map):
        """
        로봇 바로 앞(이미지 하단 중앙)에 충돌 위험이 있는 장애물이 있는지 체크.
        """
        if depth_map is None:
            return False, 0.0

        h, w = depth_map.shape[:2]
        # 위험 구역 : 하단 70%~95% 영역, 좌우 중앙 30% 영역
        y1, y2 = int(h * 0.7), int(h * 0.95)
        x1, x2 = int(w * 0.35), int(w * 0.65)

        roi = depth_map[y1:y2, x1:x2]
        valid = roi[roi > 0]

        if valid.size == 0:
            return False, 0.0

        min_dist_mm = np.min(valid)
        min_dist_m = float(min_dist_mm) / 1000.0

        if min_dist_m < self.emergency_brake_distance_m:
            return True, min_dist_m

        return False, min_dist_m

    def build_planning_prompt(self):
        pool_str = ", ".join(OBJECT_POOL)
        return f"""You are an intelligent planning module for a home service robot.
Your goal is to solve the user's problem by identifying the most appropriate object in the scene.

STEP-BY-STEP REASONING:
1. Analyze the user's command to identify their current state.
2. Determine the physical property or function needed to improve that state.
3. Evaluate objects in the image from the ALLOWED CLASSES to find the best match.
4. Identify ALL other objects from the ALLOWED CLASSES that appear in the image.

ALLOWED CLASSES (use these only):
[{pool_str}]

STRICT RULES:
- target_object: MUST be clearly visible and the best solution for the user's state. If no solution is visible, set to null.
- reasoning: A very short one-sentence explanation of why this object was chosen.
- obstacle_classes: List EVERY OTHER object from the ALLOWED CLASSES that you can see in the image (besides the target). Do NOT include the target itself. Be thorough - if you see ANY object from the list, include it. The downstream system will filter which ones actually block the path.

Output format (ONE JSON object only):
{{"intent": "<verb>", "state_analysis": "<user's state>", "reasoning": "<short explanation>", "target_object": "<class or null>", "obstacle_classes": ["<class>", ...]}}
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
        
        obstacles = data.get("obstacle_classes", [])
        if not isinstance(obstacles, list):
            obstacles = []
        
        return {
            "intent": data.get("intent"),
            "state": data.get("state_analysis"),
            "reasoning": data.get("reasoning"),
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
        if depth_map is None: return None
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = depth_map.shape[:2]
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1: return None
        roi = depth_map[y1:y2, x1:x2]
        valid = roi[roi > 0]
        if valid.size == 0: return None
        return float(np.median(valid)) / 1000.0

    def is_blocking_obstacle(self, obs_geom, obs_distance, target_geom, target_distance):
        if obs_distance >= target_distance: return False, "behind_target"
        angle_diff = abs(obs_geom["yaw_deg"] - target_geom["yaw_deg"])
        if angle_diff > self.obstacle_angle_corridor_deg: return False, "out_of_corridor"
        if obs_geom["area_ratio"] < self.obstacle_min_area_ratio: return False, "too_small"
        return True, "blocking"

    # ── 행동 결정 ──────────────────────────────────────────────

    def select_action(self, target_info, blocking_obstacles, avoidance_attempts, is_emergency=False):
        if is_emergency: return "emergency_stop"
        if blocking_obstacles and avoidance_attempts >= self.max_avoidance_attempts: return "wait_user"
        if blocking_obstacles: return "avoid_obstacle"
        if target_info["aligned"]:
            distance_error = target_info["distance"] - self.target_distance_m
            if abs(distance_error) <= self.distance_tolerance_m: return "stop_at_target"
        return "track"

    # ── 메인 진입점 ────────────────────────────────────────────

    def plan(self, image, command, depth_map, avoidance_attempts=0):
        t_total_start = time.time()
        timings = {}
        try:
            pil_img = image if not isinstance(image, str) else Image.open(image).convert("RGB")
            img_w, img_h = pil_img.size
            
            # 1. 긴급 제동 체크
            is_emergency, min_dist = self.check_emergency_brake(depth_map)
            
            # 2. VLM
            t_vlm = time.time()
            res_vlm = self.ask_vlm(pil_img, self.build_planning_prompt(), command)
            plan = self.parse_vlm_plan(res_vlm)
            timings["vlm_ms"] = round((time.time() - t_vlm) * 1000, 2)

            if plan is None or not plan["target_object"]:
                return {
                    "status": "retry",
                    "action": "retry",
                    "reason": "no_target",
                    "timings": timings,
                }

            # 3. YOLO
            t_yolo = time.time()
            target_det, obstacle_dets = self.detect_objects(pil_img, plan["target_object"], plan["obstacle_classes"])
            timings["yolo_ms"] = round((time.time() - t_yolo) * 1000, 2)

            # ★ 진단: YOLO 검출 결과
            print(f"   🔍 [YOLO] target_det={'O' if target_det else 'X'}, obstacle_dets={len(obstacle_dets)}건")
            if target_det:
                print(f"   🎯 [YOLO target] bbox={[round(v,1) for v in target_det['bbox']]} "
                      f"conf={target_det['conf']:.3f}")
            for od in obstacle_dets:
                print(f"   🚧 [YOLO obstacle] {od['class']} "
                      f"bbox={[round(v,1) for v in od['bbox']]} "
                      f"conf={od['conf']:.3f}")

            if target_det is None:
                return {
                    "status": "retry",
                    "action": "retry",
                    "reason": "target_not_found",
                    "timings": timings,
                }

            # 4. Post-processing
            t_post = time.time()
            target_distance = self._read_depth_at_bbox(depth_map, target_det["bbox"])
            if target_distance is None:
                return {
                    "status": "retry",
                    "action": "retry",
                    "reason": "target_depth_unavailable",
                    "timings": timings,
                }

            target_geom = self.compute_geometry(target_det["bbox"], img_w, img_h)
            blocking_obstacles = []
            for od in obstacle_dets:
                og = self.compute_geometry(od["bbox"], img_w, img_h)
                odist = self._read_depth_at_bbox(depth_map, od["bbox"]) or (target_distance - 0.01)
                blocking, reason = self.is_blocking_obstacle(og, odist, target_geom, target_distance)
                
                # ★ 진단: 장애물 필터링 상세
                print(f"   🔍 [obstacle filter] {od['class']} "
                      f"conf={od['conf']:.2f} "
                      f"dist={odist:.2f}m yaw={og['yaw_deg']:+.1f}° "
                      f"area={og['area_ratio']:.3f} "
                      f"→ {'BLOCKING' if blocking else f'skip({reason})'}")
                
                if blocking:
                    # system1이 회피 거리/방향 계산 시 사용하도록 yaw, distance 추가
                    od_with_info = {
                        **od,
                        "yaw_deg": og["yaw_deg"],
                        "distance": odist,
                    }
                    blocking_obstacles.append(od_with_info)

            target_info = {"class": target_det["class"], "aligned": target_geom["aligned"], "distance": target_distance}
            action = self.select_action(target_info, blocking_obstacles, avoidance_attempts, is_emergency)

            context = {
                "target": {"class": target_det["class"], "bbox": target_det["bbox"], "yaw_deg": target_geom["yaw_deg"], "aligned": target_geom["aligned"], "distance": target_distance, "distance_error": target_distance - self.target_distance_m},
                "blocking_obstacles": blocking_obstacles,
                "image": {"width": img_w, "height": img_h, "hfov_deg": self.hfov_deg},
                "config": {"target_distance_m": self.target_distance_m, "distance_tolerance_m": self.distance_tolerance_m, "center_tolerance_px": self.center_tolerance_px},
                "avoidance_attempts": avoidance_attempts,
            }

            timings["total_ms"] = round((time.time() - t_total_start) * 1000, 2)
            return {"status": "success", "action": action, "context": context, "plan": plan, "timings": timings}

        except Exception as e:
            return {"status": "abort", "action": "abort", "reason": str(e), "timings": {"total_ms": 0}}