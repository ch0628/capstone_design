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

        # [Optimized] 스마트 캐싱 및 상태 관리
        self.memory = {} # { "class_name": {"dist": 1.2, "yaw": 15.0, "time": 123456} }
        self.memory_expiry_s = 30.0
        self.last_plan = None
        self.last_valid_target_distance = None

        # VLM 캐싱 제어
        self.vlm_refresh_interval = 10 # 10프레임마다 강제 갱신
        self.frame_counter = 0
        self.target_lost_count = 0
        self.max_target_lost_limit = 3 # 3번 이상 타겟 놓치면 VLM 다시 부름

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
            "width_deg": width_deg, # [추가] 장애물 폭(각도) 정보
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
                # 큰 회전 발생 시 다음 프레임에서 VLM 갱신 유도 위해 카운터 조절
                if angle > 45: self.frame_counter = self.vlm_refresh_interval
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
                    
                    # [보정] 거리가 가까워지면 시야에서 차지하는 각도(width_deg)는 커짐
                    # 물리적 크기(W) = dist * sin(width_deg)
                    # 새로운 width_deg = arcsin(W / new_dist)
                    new_width_deg = item.get("width_deg")
                    if new_width_deg and new_dist > 0:
                        physical_size = item["dist"] * math.sin(math.radians(new_width_deg / 2))
                        # 안전을 위해 arcsin 값이 1을 넘지 않도록 클램핑
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

    def _read_depth_at_bbox(self, depth_map, bbox, label=None):
        """
        BBox 내의 Depth를 읽되, ROI 축소와 시간적 필터링(EMA)을 적용.
        """
        if depth_map is None: return None
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # 1. ROI 축소 (중앙부 50% 영역만 사용하여 배경 노이즈 제거)
        w_px, h_px = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        nx1, nx2 = max(0, cx - w_px // 4), min(cx + w_px // 4, depth_map.shape[1])
        ny1, ny2 = max(0, cy - h_px // 4), min(cy + h_px // 4, depth_map.shape[0])
        
        if nx2 <= nx1 or ny2 <= ny1: return None
        roi = depth_map[ny1:ny2, nx1:nx2]
        valid = roi[roi > 0]
        if valid.size == 0: return None
        
        current_dist = float(np.median(valid)) / 1000.0
        
        # 2. 시간적 필터링 (STM 활용)
        if label:
            prev_item = self.get_from_memory(label)
            if prev_item:
                # 급격한 변화(예: 1.5m 이상 갑자기 튐) 시 이전 값 중시하여 노이즈 제거
                if abs(current_dist - prev_item["dist"]) > 1.5:
                    return prev_item["dist"]
                # 지수 가중 이동 평균 (EMA): 현재 70%, 과거 30%
                current_dist = 0.7 * current_dist + 0.3 * prev_item["dist"]
            
        return current_dist

    def is_blocking_obstacle(self, obs_geom, obs_distance, target_geom, target_distance):
        if obs_distance >= target_distance + 0.1: return False, "behind_target"
        angle_diff = abs(obs_geom["yaw_deg"] - target_geom["yaw_deg"])
        # 장애물이 경로(타겟 방향) 주변 15도 내에 있는지 확인
        if angle_diff > self.obstacle_angle_corridor_deg: return False, "out_of_corridor"
        if obs_geom.get("area_ratio", 1.0) < self.obstacle_min_area_ratio: return False, "too_small"
        return True, "blocking"

    # ── 행동 결정 ──────────────────────────────────────────────

    def select_action(self, target_info, blocking_obstacles, avoidance_attempts):
        if blocking_obstacles and avoidance_attempts >= self.max_avoidance_attempts: return "wait_user"
        if blocking_obstacles: return "avoid_obstacle"
        if target_info["aligned"]:
            distance_error = target_info["distance"] - self.target_distance_m
            # 목표 거리에 도달했거나 근접했으면 정지
            if distance_error <= self.distance_tolerance_m: return "stop_at_target"
        return "track"

    # ── 메인 진입점 ────────────────────────────────────────────

    def plan(self, image, command, depth_map, avoidance_attempts=0):
        t_total_start = time.time()
        timings = {"vlm_ms": 0.0, "yolo_ms": 0.0, "post_ms": 0.0}
        self.frame_counter += 1
        
        try:
            pil_img = image if not isinstance(image, str) else Image.open(image).convert("RGB")
            img_w, img_h = pil_img.size
            
            # ── [1. VLM 스마트 캐싱 결정] ──
            need_vlm = False
            # 사유 1: 첫 실행
            if self.last_plan is None: need_vlm = True
            # 사유 2: 주기적 갱신
            elif self.frame_counter >= self.vlm_refresh_interval: need_vlm = True
            # 사유 3: 타겟을 일정 기간 놓침 (확증 편향 방지)
            elif self.target_lost_count >= self.max_target_lost_limit: need_vlm = True
            
            # 근접 상황(Blind)이면 VLM 강제 생략 (성능 최우선)
            is_blind_area = (
                self.last_valid_target_distance is not None and 
                self.last_valid_target_distance < self.blind_threshold_m
            )
            if is_blind_area: need_vlm = False

            if need_vlm:
                t_vlm = time.time()
                res_vlm = self.ask_vlm(pil_img, self.build_planning_prompt(), command)
                plan = self.parse_vlm_plan(res_vlm)
                timings["vlm_ms"] = round((time.time() - t_vlm) * 1000, 2)
                if plan and plan["target_object"]:
                    self.last_plan = plan
                    self.frame_counter = 0 # 카운터 리셋
            else:
                plan = self.last_plan
                # print(f"   ⚡ [VLM Cached] Frame: {self.frame_counter}/{self.vlm_refresh_interval}")

            if plan is None or not plan["target_object"]:
                return {"status": "retry", "action": "retry", "reason": "no_target", "timings": timings}

            # ── [2. YOLO 검출] ──
            t_yolo = time.time()
            target_det, obstacle_dets = self.detect_objects(pil_img, plan["target_object"], plan["obstacle_classes"])
            timings["yolo_ms"] = round((time.time() - t_yolo) * 1000, 2)

            # ── [3. 타겟 검출 실패 대응] ──
            if target_det is None:
                self.target_lost_count += 1
                
                # 메모리 복구 시도
                mem_item = self.get_from_memory(plan["target_object"])
                if mem_item:
                    print(f"   🧠 [Memory Recovery] '{plan['target_object']}' 추적 중 (Memory)")
                    target_info = {
                        "class": plan["target_object"], "aligned": abs(mem_item["yaw"]) <= 5.0,
                        "distance": mem_item["dist"], "yaw_deg": mem_item["yaw"],
                        "distance_error": mem_item["dist"] - self.target_distance_m, "is_memory": True
                    }
                    action = "stop_at_target" if mem_item["dist"] <= self.target_distance_m + 0.05 and target_info["aligned"] else "track"
                    return {"status": "success", "action": action, "context": {"target": target_info, "blocking_obstacles": [], "image": {"width": img_w, "height": img_h, "hfov_deg": self.hfov_deg}, "config": {"target_distance_m": self.target_distance_m, "distance_tolerance_m": self.distance_tolerance_m, "center_tolerance_px": self.center_tolerance_px}}, "plan": plan, "timings": timings}

                # 탐색 모드
                if not self.is_searching: self.is_searching = True; self.search_total_angle = 0.0
                if self.search_total_angle < 360.0:
                    self.search_total_angle += self.search_angle_step
                    return {"status": "success", "action": "search_rotate", "context": {"action": "search_rotate", "turn_deg": self.search_angle_step, "direction": "right"}, "plan": plan, "timings": timings}
                else:
                    self.is_searching = False
                    return {"status": "retry", "action": "retry", "reason": "target_lost", "timings": timings}

            # 타겟 발견 시 상태 업데이트
            self.target_lost_count = 0
            if self.is_searching: self.is_searching = False; self.search_total_angle = 0.0

            # ── [4. 후처리: Depth & Geometry & Memory Obstacles] ──
            t_post = time.time()
            target_distance = self._read_depth_at_bbox(depth_map, target_det["bbox"], target_det["class"])
            
            is_blind = False
            if target_distance is not None:
                self.last_valid_target_distance = target_distance
                target_geom = self.compute_geometry(target_det["bbox"], img_w, img_h)
                self.update_memory(target_det["class"], target_distance, target_geom["yaw_deg"])
            else:
                if is_blind_area:
                    is_blind = True
                    target_distance = self.last_valid_target_distance
                    target_geom = {"aligned": True, "yaw_deg": 0.0}
                else:
                    return {"status": "retry", "action": "retry", "reason": "depth_lost", "timings": timings}

            # 장애물 체크 (YOLO + Memory)
            blocking_obstacles = []
            if not is_blind:
                # YOLO 장애물
                for od in obstacle_dets:
                    og = self.compute_geometry(od["bbox"], img_w, img_h)
                    
                    # [추가] BBox 폭을 각도로 변환
                    x1, _, x2, _ = od["bbox"]
                    fx = (img_w / 2) / math.tan(math.radians(self.hfov_deg) / 2)
                    width_deg = math.degrees(math.atan(((x2 - x1) / 2) / fx)) * 2
                    
                    odist = self._read_depth_at_bbox(depth_map, od["bbox"], od["class"]) or (target_distance - 0.01)
                    
                    # 메모리 저장 시 폭 정보 포함
                    self.update_memory(od["class"], odist, og["yaw_deg"], width_deg)
                    
                    if self.is_blocking_obstacle(og, odist, target_geom, target_distance)[0]:
                        blocking_obstacles.append({**od, "yaw_deg": og["yaw_deg"], "distance": odist, "width_deg": width_deg, "source": "yolo"})
                
                # 메모리 장애물 (시야 밖)
                for label, item in self.memory.items():
                    if label == target_det["class"] or any(o["class"] == label for o in blocking_obstacles): continue
                    if self.is_blocking_obstacle({"yaw_deg": item["yaw"], "area_ratio": 0.1}, item["dist"], target_geom, target_distance)[0]:
                        blocking_obstacles.append({
                            "class": label, "distance": item["dist"], "yaw_deg": item["yaw"], 
                            "width_deg": item.get("width_deg"), # 메모리에 저장된 폭 정보 전달
                            "bbox": [0,0,0,0], "source": "memory"
                        })

            action = self.select_action({"class": target_det["class"], "aligned": target_geom["aligned"], "distance": target_distance}, blocking_obstacles, avoidance_attempts)
            if is_blind and target_distance <= self.target_distance_m + 0.05: action = "stop_at_target"

            timings["post_ms"] = round((time.time() - t_post) * 1000, 2)
            timings["total_ms"] = round((time.time() - t_total_start) * 1000, 2)
            
            context = {
                "target": {**target_det, "yaw_deg": target_geom["yaw_deg"], "aligned": target_geom["aligned"], "distance": target_distance, "distance_error": target_distance - self.target_distance_m, "is_blind": is_blind},
                "blocking_obstacles": blocking_obstacles,
                "image": {"width": img_w, "height": img_h, "hfov_deg": self.hfov_deg},
                "config": {"target_distance_m": self.target_distance_m, "distance_tolerance_m": self.distance_tolerance_m, "center_tolerance_px": self.center_tolerance_px},
                "avoidance_attempts": avoidance_attempts,
            }
            return {"status": "success", "action": action, "context": context, "plan": plan, "timings": timings}

        except Exception as e:
            import traceback; traceback.print_exc()
            return {"status": "abort", "action": "abort", "reason": str(e), "timings": {"total_ms": 0}}

        except Exception as e:
            return {"status": "abort", "action": "abort", "reason": str(e), "timings": {"total_ms": 0}}