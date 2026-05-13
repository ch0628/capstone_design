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

        # 카메라 파라미터 (HF70)
        self.hfov_deg = 70.0

        # 시스템 설정
        self.target_distance_m = 0.1  # 목표 거리 10cm

        # 제어 파라미터
        self.center_tolerance_px = 40
        self.distance_tolerance_m = 0.05 
        self.blind_threshold_m = 0.35 

        # 장애물 판정 기준
        self.obstacle_angle_corridor_deg = 15.0
        self.obstacle_min_area_ratio = 0.01
        self.max_avoidance_attempts = 3

        # 상태 기억
        self.last_valid_target_distance = None
        self.last_plan = None

        print("✅ System2(VisionPlanner) 준비 완료")

    def reset_state(self):
        """새 명령 시작 시 이전 상태 초기화"""
        self.last_valid_target_distance = None
        self.last_plan = None
        print("🧹 [System2] 상태 초기화 완료")


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
        return f"""You are an intelligent planning module for a home service robot.
Your goal is to solve the user's problem by identifying the most appropriate object in the scene.

STEP-BY-STEP REASONING:
1. Analyze the user's command to identify their current state (e.g., hot, tired, thirsty, bored).
2. Determine the physical property or function needed to improve that state (e.g., cooling, comfort, hydration, entertainment).
3. Evaluate objects in the image from the ALLOWED CLASSES to find the best match for that property.
4. Identify ALL other visible objects from the ALLOWED CLASSES as potential obstacles.

ALLOWED CLASSES (use these only):
[{pool_str}]

STRICT RULES:
- target_object: MUST be clearly visible and the best solution for the user's state. If no solution is visible, set to null.
- reasoning: A very short one-sentence explanation of why this object was chosen.
- obstacle_classes: List EVERY other object from ALLOWED CLASSES visible in the image (excluding target).
  Be thorough - the downstream system filters which ones actually block the path.
  Empty list ONLY if literally no other allowed class is visible.

Output format (ONE JSON object only):
{{"intent": "<verb>", "state_analysis": "<user's state>", "reasoning": "<short explanation>", "target_object": "<class or null>", "obstacle_classes": ["<class>", ...]}}

Examples:

User: "My legs hurt, I need to rest"
Scene shows: chair, potted plant in path
-> {{"intent": "rest", "state_analysis": "tired legs", "reasoning": "User needs to sit down, chair is ideal.", "target_object": "chair", "obstacle_classes": ["potted plant"]}}

User: "Pass me something to read"
Scene shows: book on couch, cup nearby
-> {{"intent": "read", "state_analysis": "wants entertainment", "reasoning": "Book provides reading material.", "target_object": "book", "obstacle_classes": ["couch", "cup"]}}

User: "I want to change the channel"
Scene shows: only remote visible
-> {{"intent": "control_tv", "state_analysis": "wants TV control", "reasoning": "Remote controls the TV.", "target_object": "remote", "obstacle_classes": []}}

User: "Help me find my phone"
Scene shows: no phone visible
-> {{"intent": "locate_phone", "state_analysis": "needs phone", "reasoning": "No phone visible in scene.", "target_object": null, "obstacle_classes": []}}
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

    def select_action(self, target_info, blocking_obstacles, avoidance_attempts):
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
            
            # 2. VLM (블라인드 예상 시 생략)
            can_skip_vlm = (
                self.last_plan is not None and 
                self.last_valid_target_distance is not None and 
                self.last_valid_target_distance < self.blind_threshold_m
            )

            if can_skip_vlm:
                plan = self.last_plan
                timings["vlm_ms"] = 0.0
                print(f"   🎯 [Blind Skip] 근접 상황으로 VLM 생략 (Target: {plan['target_object']})")
            else:
                t_vlm = time.time()
                res_vlm = self.ask_vlm(pil_img, self.build_planning_prompt(), command)
                plan = self.parse_vlm_plan(res_vlm)
                timings["vlm_ms"] = round((time.time() - t_vlm) * 1000, 2)
                if plan and plan["target_object"]:
                    self.last_plan = plan # 성공한 플랜 저장

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

            # 4. Post-processing (Depth & Geometry)
            t_post = time.time()
            target_distance = self._read_depth_at_bbox(depth_map, target_det["bbox"])
            
            # [Blind Approach 로직]
            is_blind = False
            if target_distance is not None:
                self.last_valid_target_distance = target_distance
            else:
                # 거리가 안 읽히는데, 마지막 거리가 35cm 이하였다면 근접 상황으로 간주
                if self.last_valid_target_distance and self.last_valid_target_distance < self.blind_threshold_m:
                    is_blind = True
                    target_distance = self.last_valid_target_distance # 마지막 값으로 대체
                    print(f"   🎯 [Blind Approach] 센서 사각지대 진입. 마지막 유효 거리({target_distance:.2f}m) 기반 전진")
                    
                    # [중요] 다음 사이클에서 또 전진하지 않도록 예측치로 업데이트
                    # 실제 이동은 system1이 하겠지만, 여기서는 목표치에 도달했다고 가정하고 미리 업데이트함
                    self.last_valid_target_distance = self.target_distance_m
                else:
                    return {
                        "status": "retry",
                        "action": "retry",
                        "reason": "target_depth_unavailable",
                        "timings": timings,
                    }

            target_geom = self.compute_geometry(target_det["bbox"], img_w, img_h)
            blocking_obstacles = []
            
            # 장애물 체크 (블라인드 주행 중에는 장애물 체크 생략)
            if not is_blind:
                for od in obstacle_dets:
                    og = self.compute_geometry(od["bbox"], img_w, img_h)
                    odist = self._read_depth_at_bbox(depth_map, od["bbox"]) or (target_distance - 0.01)
                    blocking, reason = self.is_blocking_obstacle(og, odist, target_geom, target_distance)
                    
                    if blocking:
                        od_with_info = {**od, "yaw_deg": og["yaw_deg"], "distance": odist}
                        blocking_obstacles.append(od_with_info)

            target_info = {
                "class": target_det["class"], 
                "aligned": target_geom["aligned"], 
                "distance": target_distance,
                "is_blind": is_blind
            }
            action = self.select_action(target_info, blocking_obstacles, avoidance_attempts)

            # 블라인드 상태에서 목표 도달 판정 강화
            if is_blind:
                # 마지막 거리에서 이미 목표(10cm)에 도달했거나 더 가까우면 정지
                if target_distance <= self.target_distance_m + 0.05:
                    action = "stop_at_target"

            context = {
                "target": {
                    "class": target_det["class"], 
                    "bbox": target_det["bbox"], 
                    "yaw_deg": target_geom["yaw_deg"], 
                    "aligned": target_geom["aligned"], 
                    "distance": target_distance, 
                    "distance_error": target_distance - self.target_distance_m,
                    "is_blind": is_blind
                },
                "blocking_obstacles": blocking_obstacles,
                "image": {"width": img_w, "height": img_h, "hfov_deg": self.hfov_deg},
                "config": {"target_distance_m": self.target_distance_m, "distance_tolerance_m": self.distance_tolerance_m, "center_tolerance_px": self.center_tolerance_px},
                "avoidance_attempts": avoidance_attempts,
            }

            timings["post_ms"] = round((time.time() - t_post) * 1000, 2)
            timings["total_ms"] = round((time.time() - t_total_start) * 1000, 2)
            return {"status": "success", "action": action, "context": context, "plan": plan, "timings": timings}

        except Exception as e:
            return {"status": "abort", "action": "abort", "reason": str(e), "timings": {"total_ms": 0}}