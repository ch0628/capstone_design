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

핵심 인터페이스:
    planner.plan(image, command, depth_map, avoidance_attempts)
        -> action_command dict {"status", "action", "context", "debug_frame", ...}

debug_frame: YOLO BBox가 그려진 BGR numpy (main.py에서 send_to_pc로 Local PC 전송, JSON 응답에는 포함 안 됨)
"""

import json
import re
import math
import cv2
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
        self.obstacle_angle_corridor_deg = 20.0
        self.obstacle_min_area_ratio = 0.01
        self.max_avoidance_attempts = 3
        self.target_distance_m = 0.15      # 목표: 타겟 15cm 앞에서 정지
        self.obstacle_avoid_early_m = 0.50  # 보수적 회피: 이 거리 이내 장애물부터 회피 시작

        # 삼각형 유사 원리로 depth dead zone을 극복하기 위한 캐시
        # 마지막으로 depth가 유효했을 때의 (거리_m, bbox_픽셀_높이) 저장
        self._last_target_depth_cache: dict[str, tuple[float, float]] = {}
        self._last_obs_depth_cache: dict[str, tuple[float, float]] = {}

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

    def detect_objects(self, pil_img, target_class, obstacle_classes, conf=0.05):
        yolo_results = self.yolo.predict(pil_img, conf=conf, verbose=False)

        target_class_lower = target_class.lower() if target_class else None
        obstacle_classes_lower = [c.lower() for c in obstacle_classes]

        best_target = None
        best_target_conf = -1.0
        obstacle_dets = []

        for r in yolo_results:
            img_h_orig, img_w_orig = r.orig_shape
            for box in r.boxes:
                label = r.names[int(box.cls[0])].lower()
                conf_val = float(box.conf[0])
                bbox = box.xyxy[0].tolist()

                if target_class_lower and label == target_class_lower:
                    if conf_val > best_target_conf:
                        best_target_conf = conf_val
                        best_target = {"class": label, "conf": conf_val, "bbox": bbox}
                elif label in obstacle_classes_lower:
                    x1, y1, x2, y2 = bbox
                    area_ratio = ((x2 - x1) * (y2 - y1)) / (img_w_orig * img_h_orig)
                    if area_ratio >= self.obstacle_min_area_ratio:
                        obstacle_dets.append({"class": label, "conf": conf_val, "bbox": bbox})

        return best_target, obstacle_dets

    def build_annotated_frame(self, pil_img, target_det, obstacle_dets, plan=None, action=None):
        """YOLO BBox가 그려진 BGR 프레임 (Local PC debug_frame_sender용)."""
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        for od in obstacle_dets:
            x1, y1, x2, y2 = [int(v) for v in od["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"{od['class']} {od['conf']:.2f}",
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
                f"TARGET {target_det['class']} {target_det['conf']:.2f}",
                (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        if plan:
            label = plan.get("target_object") or "no_target"
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

    def _estimate_distance_by_bbox_ratio(self, bbox, obj_class, cache):
        """
        삼각형 유사 원리로 depth dead zone 내 거리 추정.

        원리: 카메라 bbox 픽셀 높이는 거리에 반비례하므로
              d_now = d_last × (h_last / h_now)
        물체의 실제 크기를 가정하지 않고, 마지막으로 depth가 유효했던
        시점의 (d_last, h_last) 캐시만 사용.
        """
        entry = cache.get(obj_class)
        if entry is None:
            return None  # 캐시 없음 → 추정 불가
        d_last, h_last = entry
        _, y1, _, y2 = bbox
        h_now = y2 - y1
        if h_now <= 0 or h_last <= 0:
            return None
        return round(d_last * (h_last / h_now), 3)

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

    def select_action(self, target_info, blocking_obstacles, avoidance_attempts):
        if blocking_obstacles:
            if avoidance_attempts >= self.max_avoidance_attempts:
                return "wait_user"

            closest = min(blocking_obstacles, key=lambda o: o["distance"])

            # depth·추정 모두 실패 → 거리 불명, 보수적으로 즉시 회피
            if closest.get("depth_measurement") == "estimated_failed":
                return "avoid_obstacle"

            # obstacle_avoid_early_m(0.5m) 이내면 회피, 그보다 멀면 track으로 접근
            # → 1m·2m 밖 장애물에 즉시 반응해 경로가 흐트러지는 문제 방지
            if closest["distance"] <= self.obstacle_avoid_early_m:
                return "avoid_obstacle"

            # 아직 멀다 → 일단 target 방향으로 접근, 다음 프레임에서 재평가
            return "track"

        # 장애물 없음 — 정렬 후 거리 접근
        if not target_info["aligned"]:
            return "track"

        dist = target_info.get("distance")
        if dist is None:
            # depth dead zone 진입 + 정렬 완료 → 목표 거리 도달로 간주
            return "stop_at_target"

        distance_error = dist - self.target_distance_m
        if abs(distance_error) <= self.distance_tolerance_m:
            return "stop_at_target"

        return "track"

    # ── 메인 진입점 ────────────────────────────────────────────

    def plan(self, image, command, depth_map, avoidance_attempts=0):
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
            print(
                f"\n📸 [System2] {src_label} "
                f"(명령: {command}, attempts={avoidance_attempts})"
            )

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
            obstacle_classes = plan["obstacle_classes"]
            print(f"🧠 [Plan] target={target_obj}, obstacles={obstacle_classes}")

            if not target_obj:
                timings["total_ms"] = round((time.time() - t_total_start) * 1000, 2)
                return {
                    "status": "retry", "action": "retry",
                    "reason": "no_target",
                    "plan": plan, "timings": timings,
                }

            t_yolo = time.time()
            target_det, obstacle_dets = self.detect_objects(
                pil_img, target_obj, obstacle_classes
            )
            timings["yolo_ms"] = round((time.time() - t_yolo) * 1000, 2)

            annotated_frame = self.build_annotated_frame(
                pil_img, target_det, obstacle_dets, plan=plan
            )

            print(f"   🔍 [YOLO] target_det={'O' if target_det else 'X'}, obstacle_dets={len(obstacle_dets)}건")
            if target_det:
                print(f"   🎯 [YOLO target] bbox={[round(v,1) for v in target_det['bbox']]} "
                      f"conf={target_det['conf']:.3f}")
            for od in obstacle_dets:
                print(f"   🚧 [YOLO obstacle] {od['class']} "
                      f"bbox={[round(v,1) for v in od['bbox']]} "
                      f"conf={od['conf']:.3f}")

            # 회전 직후 카메라 흔들림으로 conf가 임계 직하로 떨어지는 경우를 보완.
            # 첫 시도에서 타겟을 못 찾으면 낮은 conf로 한 번 더 시도.
            if target_det is None:
                target_det_retry, obstacle_dets_retry = self.detect_objects(
                    pil_img, target_obj, obstacle_classes, conf=0.02
                )
                if target_det_retry is not None:
                    print(f"   🔄 [YOLO 재시도 conf=0.02] 타겟 발견: "
                          f"conf={target_det_retry['conf']:.3f}")
                    target_det = target_det_retry
                    # obstacle_dets는 conf=0.05 원래 결과 유지 (노이즈 방지)

            if target_det is None:
                timings["total_ms"] = round((time.time() - t_total_start) * 1000, 2)
                return {
                    "status": "retry", "action": "retry",
                    "reason": "target_not_found",
                    "plan": plan, "timings": timings,
                    "debug_frame": annotated_frame,
                }

            t_post = time.time()
            target_geom = self.compute_geometry(target_det["bbox"], img_w, img_h)

            target_distance = self._read_depth_at_bbox(depth_map, target_det["bbox"])

            if target_distance is not None:
                # 유효한 depth → 캐시 갱신 (dead zone 진입 후 추정에 사용)
                _, ty1, _, ty2 = target_det["bbox"]
                self._last_target_depth_cache[target_det["class"]] = (target_distance, ty2 - ty1)
                print(f"📏 [Depth] target={target_distance:.2f}m")
            else:
                # depth dead zone 진입 — 삼각형 유사 원리로 거리 추정
                est_dist = self._estimate_distance_by_bbox_ratio(
                    target_det["bbox"], target_det["class"], self._last_target_depth_cache
                )
                aligned = target_geom["aligned"]
                target_info = {
                    "class": target_det["class"],
                    "conf": round(target_det["conf"], 3),
                    "bbox": target_det["bbox"],
                    "cx": target_geom["cx"],
                    "yaw_deg": target_geom["yaw_deg"],
                    "aligned": aligned,
                    "distance": est_dist,  # 추정 거리 (캐시 없으면 None)
                    "distance_error": (est_dist - self.target_distance_m) if est_dist is not None else None,
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
                # 추정 거리로 도달 여부 판단. 정렬 + dead zone 진입도 stop으로 처리.
                reached = (
                    est_dist is not None
                    and est_dist <= self.target_distance_m + self.distance_tolerance_m
                )
                if reached or aligned:
                    action = "stop_at_target"
                    dist_str = f"{est_dist:.2f}m" if est_dist is not None else "N/A"
                    print(f"📏 [Depth] dead_zone est={dist_str} → stop_at_target")
                else:
                    action = "track"
                    dist_str = f"{est_dist:.2f}m" if est_dist is not None else "N/A"
                    print(f"📏 [Depth] dead_zone est={dist_str} > {self.target_distance_m}m → track")
                timings["post_ms"] = round((time.time() - t_post) * 1000, 2)
                timings["total_ms"] = round((time.time() - t_total_start) * 1000, 2)
                return {
                    "status": "success",
                    "action": action,
                    "context": context,
                    "plan": plan,
                    "timings": timings,
                    "debug_frame": annotated_frame,
                }

            obstacles_info = []
            blocking_obstacles = []

            for od in obstacle_dets:
                og = self.compute_geometry(od["bbox"], img_w, img_h)

                obs_distance = self._read_depth_at_bbox(depth_map, od["bbox"])
                if obs_distance is not None:
                    # 유효한 depth → 캐시 갱신
                    _, oy1, _, oy2 = od["bbox"]
                    self._last_obs_depth_cache[od["class"]] = (obs_distance, oy2 - oy1)
                    depth_status = "measured"
                else:
                    # depth 실패 → 삼각형 원리로 추정
                    est = self._estimate_distance_by_bbox_ratio(
                        od["bbox"], od["class"], self._last_obs_depth_cache
                    )
                    if est is not None:
                        obs_distance = est
                        depth_status = "estimated"      # 추정 성공 → 거리 기반 판단 가능
                    else:
                        obs_distance = max(0.01, target_distance - 0.05)
                        depth_status = "estimated_failed"  # 추정도 불가 → 보수적 즉시 회피

                blocking, reason = self.is_blocking_obstacle(
                    og, od["bbox"], obs_distance,
                    target_geom, target_det["bbox"], target_distance,
                )

                ox1, _, ox2, _ = od["bbox"]
                tx1, _, tx2, _ = target_det["bbox"]
                tw = tx2 - tx1
                x_overlap = (ox2 >= (tx1 - tw)) and (ox1 <= (tx2 + tw))

                print(f"   🔍 [obstacle filter] {od['class']} "
                      f"conf={od['conf']:.2f} "
                      f"dist={obs_distance:.2f}m yaw={og['yaw_deg']:+.1f}° "
                      f"x_overlap={'O' if x_overlap else 'X'} "
                      f"→ {'BLOCKING' if blocking else f'skip({reason})'}")

                entry = {
                    "class": od["class"],
                    "conf": round(od["conf"], 3),
                    "bbox": od["bbox"],
                    "yaw_deg": og["yaw_deg"],
                    "area_ratio": round(og["area_ratio"], 4),
                    "distance": round(obs_distance, 3),
                    "depth_measurement": depth_status,
                    "blocking": blocking,
                    "skip_reason": None if blocking else reason,
                }
                obstacles_info.append(entry)
                if blocking:
                    blocking_obstacles.append(entry)

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

            action = self.select_action(
                target_info, blocking_obstacles, avoidance_attempts
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
            timings["total_ms"] = round((time.time() - t_total_start) * 1000, 2)

            return {
                "status": "success",
                "action": action,
                "context": context,
                "plan": plan,
                "timings": timings,
                "debug_frame": annotated_frame,
            }

        except Exception as e:
            print(f"❌ [System2] 에러: {e}")
            timings["total_ms"] = round((time.time() - t_total_start) * 1000, 2)
            return {
                "status": "abort", "action": "abort", "reason": str(e),
                "timings": timings,
            }