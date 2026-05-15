"""
System 2 — VisionPlanner (merged: 잘된 날 베이스 + retry/진단/blocking 강화)

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
  - "search_rotate"   : 타겟을 찾기 위한 제자리 회전

핵심 인터페이스:
    planner.plan(image, command, depth_map, avoidance_attempts)
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

        # [Optimized] 스마트 캐싱 및 상태 관리
        self.memory = {} # { "class_name": {"dist": 1.2, "yaw": 15.0, "time": 123456} }
        self.memory_expiry_s = 30.0
        self.last_plan = None
        self.last_valid_target_distance = None

        # VLM 캐싱 제어
        self.vlm_refresh_interval = 10 
        self.frame_counter = 0
        self.target_lost_count = 0
        self.max_target_lost_limit = 3 

        # 검색 상태 (360도 탐색)
        self.is_searching = False
        self.search_total_angle = 0.0
        self.search_angle_step = 60.0

        print("✅ System2(VisionPlanner) 준비 완료 (Smart Caching 활성화)")

    def reset_state(self):
        """새 명령 시작 시 이전 상태 초기화"""
        self.last_valid_target_distance = None
        self.last_plan = None
        self.memory = {}
        self.is_searching = False
        self.search_total_angle = 0.0
        self.frame_counter = 0
        self.target_lost_count = 0
        print("🧹 [System2] 상태 초기화 완료")

    # ── 단기 기억 관리 ──────────────────────────────────────────

    def update_memory(self, label, distance, yaw, width_deg=None):
        self.memory[label] = {
            "dist": distance, 
            "yaw": yaw, 
            "width_deg": width_deg,
            "time": time.time()
        }

    def get_from_memory(self, label):
        if label not in self.memory: return None
        item = self.memory[label]
        if time.time() - item["time"] > self.memory_expiry_s:
            del self.memory[label]
            return None
        return item

    def update_memory_on_motion(self, motions):
        """로봇 이동에 따른 메모리 좌표 보정"""
        for m in motions:
            turn_match = re.match(r"TURN (LEFT|RIGHT) ([\d.]+)deg", m)
            if turn_match:
                direction, angle = turn_match.group(1).upper(), float(turn_match.group(2))
                diff = -angle if direction == "LEFT" else angle
                for label in self.memory: self.memory[label]["yaw"] += diff
                
                # 5도 이상 회전 시 다음 프레임에서 VLM 갱신 유도
                if angle > 5.0: 
                    self.frame_counter = self.vlm_refresh_interval
                    self.target_lost_count = 0 
                continue

            move_match = re.match(r"MOVE (FRONT|BACK) ([\d.]+)m", m)
            if move_match:
                direction, dist = move_match.group(1).upper(), float(move_match.group(2))
                sign = 1 if direction == "FRONT" else -1
                new_memory = {}
                for label, item in self.memory.items():
                    yaw_rad = math.radians(item["yaw"])
                    x = item["dist"] * math.cos(yaw_rad) - (dist * sign)
                    y = -item["dist"] * math.sin(yaw_rad)
                    new_dist = math.sqrt(x**2 + y**2)
                    new_yaw = math.degrees(math.atan2(-y, x))
                    
                    new_width_deg = item.get("width_deg")
                    if new_width_deg and new_dist > 0:
                        physical_size = item["dist"] * math.sin(math.radians(new_width_deg / 2))
                        ratio = min(1.0, physical_size / new_dist)
                        new_width_deg = math.degrees(math.asin(ratio)) * 2

                    new_memory[label] = {
                        "dist": new_dist, 
                        "yaw": new_yaw, 
                        "width_deg": new_width_deg,
                        "time": item["time"]
                    }
                self.memory = new_memory

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
"""

    def parse_vlm_plan(self, raw_text):
        json_matches = re.findall(r"\{.*?\}", raw_text, re.DOTALL)
        if not json_matches: return None
        try:
            data = json.loads(json_matches[-1])
        except json.JSONDecodeError: return None

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
        """
        [개선] 타겟과 같은 클래스라도 타겟 본인이 아닌 객체는 장애물로 처리.
        """
        yolo_results = self.yolo.predict(pil_img, conf=0.05, verbose=False)
        target_class_lower = target_class.lower() if target_class else None
        obstacle_classes_lower = [c.lower() for c in obstacle_classes]

        all_dets = []
        for r in yolo_results:
            for box in r.boxes:
                label = r.names[int(box.cls[0])].lower()
                conf = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                all_dets.append({"class": label, "conf": conf, "bbox": bbox})

        # 1. Best Target 선정 (확신도가 가장 높은 객체)
        best_target = None
        best_target_conf = -1.0
        if target_class_lower:
            for d in all_dets:
                if d["class"] == target_class_lower:
                    if d["conf"] > best_target_conf:
                        best_target_conf = d["conf"]
                        best_target = d

        # 2. 장애물 분류 (타겟 본인 제외 모든 해당 클래스 객체)
        obstacle_dets = []
        for d in all_dets:
            if best_target and d["bbox"] == best_target["bbox"]:
                continue
            
            # VLM이 지목한 장애물 클래스이거나, 타겟과 클래스는 같지만 본인이 아닌 경우
            if d["class"] in obstacle_classes_lower or (target_class_lower and d["class"] == target_class_lower):
                obstacle_dets.append(d)

        return best_target, obstacle_dets

    # ── 기하 ───────────────────────────────────────────────────

    def compute_geometry(self, bbox, img_w, img_h):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        yaw_deg, pixel_offset, aligned = self.compute_yaw_to_center(cx, img_w)
        area_ratio = ((x2 - x1) * (y2 - y1)) / (img_w * img_h)
        return {
            "cx": cx, "cy": cy, "yaw_deg": yaw_deg,
            "pixel_offset": pixel_offset, "aligned": aligned, "area_ratio": area_ratio,
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

    def _read_depth_at_bbox(self, depth_map, bbox, label=None):
        if depth_map is None: return None
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        w_px, h_px = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        nx1, nx2 = max(0, cx - w_px // 4), min(cx + w_px // 4, depth_map.shape[1])
        ny1, ny2 = max(0, cy - h_px // 4), min(cy + h_px // 4, depth_map.shape[0])
        
        if nx2 <= nx1 or ny2 <= ny1: return None
        roi = depth_map[ny1:ny2, nx1:nx2]
        valid = roi[roi > 0]
        if valid.size == 0: return None
        
        current_dist = float(np.median(valid)) / 1000.0
        
        if label:
            prev_item = self.get_from_memory(label)
            if prev_item:
                if abs(current_dist - prev_item["dist"]) > 1.5:
                    return prev_item["dist"]
                current_dist = 0.7 * current_dist + 0.3 * prev_item["dist"]
            
        return current_dist

    def is_blocking_obstacle(self, obs_geom, obs_distance, target_geom, target_distance):
        if obs_distance >= target_distance + 0.1: return False, "behind_target"
        
        angle_diff = abs(obs_geom["yaw_deg"] - target_geom["yaw_deg"])
        dynamic_corridor = self.obstacle_angle_corridor_deg
        if obs_distance < 1.5: dynamic_corridor += 5.0
        if angle_diff > dynamic_corridor: return False, "out_of_corridor"
            
        dynamic_min_area = self.obstacle_min_area_ratio
        if obs_distance < 1.0: dynamic_min_area = 0.005 
        if obs_geom.get("area_ratio", 1.0) < dynamic_min_area: return False, "too_small"
            
        return True, "blocking"

    # ── 행동 결정 ──────────────────────────────────────────────

    def select_action(self, target_info, blocking_obstacles, avoidance_attempts):
        if blocking_obstacles and avoidance_attempts >= self.max_avoidance_attempts: 
            return "wait_user"
        if blocking_obstacles: 
            return "avoid_obstacle"
        if target_info["aligned"]:
            distance_error = target_info["distance"] - self.target_distance_m
            if distance_error <= self.distance_tolerance_m: 
                return "stop_at_target"
        return "track"

    # ── 메인 진입점 ────────────────────────────────────────────

    def plan(self, image, command, depth_map, avoidance_attempts=0):
        t_total_start = time.time()
        timings = {"vlm_ms": 0.0, "yolo_ms": 0.0, "post_ms": 0.0}
        self.frame_counter += 1
        
        try:
            pil_img = image if not isinstance(image, str) else Image.open(image).convert("RGB")
            img_w, img_h = pil_img.size
            
            # 1. VLM 스마트 캐싱
            need_vlm = (self.last_plan is None or 
                        self.frame_counter >= self.vlm_refresh_interval or 
                        self.target_lost_count >= self.max_target_lost_limit)

            if need_vlm:
                t_vlm = time.time()
                res_vlm = self.ask_vlm(pil_img, self.build_planning_prompt(), command)
                plan = self.parse_vlm_plan(res_vlm)
                timings["vlm_ms"] = round((time.time() - t_vlm) * 1000, 2)
                if plan and plan["target_object"]:
                    self.last_plan = plan
                    self.frame_counter = 0 
            else:
                plan = self.last_plan

            if plan is None or not plan["target_object"]:
                return {"status": "retry", "action": "retry", "reason": "no_target", "timings": timings}

            # 2. YOLO 검출
            t_yolo = time.time()
            target_det, obstacle_dets = self.detect_objects(pil_img, plan["target_object"], plan["obstacle_classes"])
            timings["yolo_ms"] = round((time.time() - t_yolo) * 1000, 2)

            # 3. 타겟 상실 시 메모리 복구 또는 탐색
            if target_det is None:
                self.target_lost_count += 1
                mem_item = self.get_from_memory(plan["target_object"])
                if mem_item:
                    print(f"   🧠 [Memory Recovery] '{plan['target_object']}' 추적 중")
                    target_info = {
                        "class": plan["target_object"], "aligned": abs(mem_item["yaw"]) <= 5.0,
                        "distance": mem_item["dist"], "yaw_deg": mem_item["yaw"],
                        "distance_error": mem_item["dist"] - self.target_distance_m, "is_memory": True
                    }
                    # 메모리로 추적 중에도 실시간 장애물 체크 (시야 내 장애물)
                    blocking_obstacles = []
                    for od in obstacle_dets:
                        og = self.compute_geometry(od["bbox"], img_w, img_h)
                        odist = self._read_depth_at_bbox(depth_map, od["bbox"], od["class"]) or (mem_item["dist"] - 0.01)
                        if self.is_blocking_obstacle(og, odist, {"yaw_deg": mem_item["yaw"]}, mem_item["dist"])[0]:
                            blocking_obstacles.append({**od, "yaw_deg": og["yaw_deg"], "distance": odist, "source": "yolo"})
                    
                    action = self.select_action(target_info, blocking_obstacles, avoidance_attempts)
                    return {"status": "success", "action": action, "context": {"target": target_info, "blocking_obstacles": blocking_obstacles, "image": {"width": img_w, "height": img_h, "hfov_deg": self.hfov_deg}, "config": {"target_distance_m": self.target_distance_m, "distance_tolerance_m": self.distance_tolerance_m, "center_tolerance_px": self.center_tolerance_px}, "avoidance_attempts": avoidance_attempts}, "plan": plan, "timings": timings}

                if not self.is_searching: self.is_searching = True; self.search_total_angle = 0.0
                if self.search_total_angle < 360.0:
                    self.search_total_angle += self.search_angle_step
                    return {"status": "success", "action": "search_rotate", "context": {"action": "search_rotate", "turn_deg": self.search_angle_step, "direction": "right"}, "plan": plan, "timings": timings}
                else:
                    self.is_searching = False
                    return {"status": "retry", "action": "retry", "reason": "target_lost", "timings": timings}

            # 4. 타겟 발견 시 정상 처리
            self.target_lost_count = 0
            if self.is_searching: self.is_searching = False; self.search_total_angle = 0.0

            t_post = time.time()
            target_distance = self._read_depth_at_bbox(depth_map, target_det["bbox"], target_det["class"])
            
            # [개선] 블라인드 로직 완전 제거: Depth 정보가 없으면 즉시 retry
            if target_distance is None:
                return {"status": "retry", "action": "retry", "reason": "depth_lost", "timings": timings}
            
            self.last_valid_target_distance = target_distance
            target_geom = self.compute_geometry(target_det["bbox"], img_w, img_h)
            self.update_memory(target_det["class"], target_distance, target_geom["yaw_deg"])

            # 실시간 장애물 체크 (YOLO + Memory)
            blocking_obstacles = []
            # 1) YOLO 장애물
            for od in obstacle_dets:
                og = self.compute_geometry(od["bbox"], img_w, img_h)
                x1, _, x2, _ = od["bbox"]
                fx = (img_w / 2) / math.tan(math.radians(self.hfov_deg) / 2)
                width_deg = math.degrees(math.atan(((x2 - x1) / 2) / fx)) * 2
                odist = self._read_depth_at_bbox(depth_map, od["bbox"], od["class"]) or (target_distance - 0.01)
                
                self.update_memory(od["class"], odist, og["yaw_deg"], width_deg)
                if self.is_blocking_obstacle(og, odist, target_geom, target_distance)[0]:
                    blocking_obstacles.append({**od, "yaw_deg": og["yaw_deg"], "distance": odist, "width_deg": width_deg, "source": "yolo"})
            
            # 2) 메모리 장애물 (시야 밖)
            for label, item in self.memory.items():
                if label == target_det["class"] or any(o["class"] == label for o in blocking_obstacles): continue
                if self.is_blocking_obstacle({"yaw_deg": item["yaw"], "area_ratio": 0.1}, item["dist"], target_geom, target_distance)[0]:
                    blocking_obstacles.append({
                        "class": label, "distance": item["dist"], "yaw_deg": item["yaw"], 
                        "width_deg": item.get("width_deg"), "bbox": [0,0,0,0], "source": "memory"
                    })

            action = self.select_action({"class": target_det["class"], "aligned": target_geom["aligned"], "distance": target_distance}, blocking_obstacles, avoidance_attempts)

            timings["post_ms"] = round((time.time() - t_post) * 1000, 2)
            timings["total_ms"] = round((time.time() - t_total_start) * 1000, 2)
            
            context = {
                "target": {**target_det, "yaw_deg": target_geom["yaw_deg"], "aligned": target_geom["aligned"], "distance": target_distance, "distance_error": target_distance - self.target_distance_m},
                "blocking_obstacles": blocking_obstacles,
                "image": {"width": img_w, "height": img_h, "hfov_deg": self.hfov_deg},
                "config": {"target_distance_m": self.target_distance_m, "distance_tolerance_m": self.distance_tolerance_m, "center_tolerance_px": self.center_tolerance_px},
                "avoidance_attempts": avoidance_attempts,
            }
            return {"status": "success", "action": action, "context": context, "plan": plan, "timings": timings}

        except Exception as e:
            import traceback; traceback.print_exc()
            return {"status": "abort", "action": "abort", "reason": str(e), "timings": {"total_ms": 0}}
