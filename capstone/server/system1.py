"""
System 1 — MotionExecutor

하위 제어 모듈.
System 2가 보낸 action_command를 받아서:
  1) "어떻게 실행할지" 자체 판단 (회전 먼저? 전진? 회피 각도 얼마?)
  2) 모터 제어 명령 생성
  3) 모터 제어 실행 (mock — 실제 모터는 RPi의 motor.py가 처리)

행동(action) 종류별 처리:
  - "track"           : aligned 보고 회전 또는 전진 결정
  - "avoid_obstacle"  : bbox 기반 회피 각도 계산 후 회전+전진
  - "stop_at_target"  : 모든 모터 정지
  - "wait_user"       : 사용자 메시지 출력하고 대기
  - "retry"           : 모터 동작 없이 sender에게 reason 전달 (재시도 유도)
  - "emergency_stop"  : 긴급 제동
  - "abort"           : 시스템 에러 → STOP

핵심 인터페이스:
    executor.execute(action_command)
        -> ExecutionResult dict {
            "executed_motions": [...],
            "next_action_hint": str,
            "next_action_reason": str (retry/wait_user 시),
            "next_avoidance_attempts": int,
        }
"""

import math


class MotionExecutor:
    def __init__(self):
        # 회피 동작 파라미터
        self.avoid_safety_margin_deg = 10.0
        self.avoid_turn_min_deg = 10.0
        self.avoid_turn_max_deg = 30.0
        self.avoid_pass_buffer_m = 0.1   # 장애물 너머 여유 마진

        # 한 사이클당 최대 이동/회전
        self.max_forward_per_cycle_m = 0.4
        self.max_turn_per_cycle_deg = 20.0

        self.motor_log = []

        print("✅ System1(MotionExecutor) 준비 완료 (mock 모드)")
        print(f"   max_forward_per_cycle = {self.max_forward_per_cycle_m}m")
        print(f"   max_turn_per_cycle = {self.max_turn_per_cycle_deg}deg")

    # ── 모터 명령 (mock) ───────────────────────────────────────

    def _motor_turn(self, direction, amount_deg):
        cmd = f"TURN {direction.upper()} {amount_deg:.2f}deg"
        print(f"   🔧 [모터] {cmd}")
        self.motor_log.append(cmd)
        return cmd

    def _motor_move(self, direction, distance_m):
        cmd = f"MOVE {direction.upper()} {distance_m:.3f}m"
        print(f"   🔧 [모터] {cmd}")
        self.motor_log.append(cmd)
        return cmd

    def _motor_stop(self):
        cmd = "STOP"
        print(f"   🔧 [모터] {cmd}")
        self.motor_log.append(cmd)
        return cmd

    # ── 행동별 실행 로직 ───────────────────────────────────────

    def _execute_track(self, context):
        """
        정렬 안 되면 회전, 정렬됐으면 전진/후진.
        한 사이클에 한 동작만.
        """
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
        """
        가장 위험한 장애물 골라서 회피.
        - bbox 폭으로 회피 각도
        - 장애물 depth 기반 동적 전진 거리 (max 한도 내)
        """
        target = context["target"]
        blocking = context["blocking_obstacles"]
        img_w = context["image"]["width"]
        hfov_deg = context["image"]["hfov_deg"]
        executed = []

        # 가장 위험한 장애물 = target과 yaw 차이가 가장 작은 것
        most_blocking = min(
            blocking,
            key=lambda o: abs(o["yaw_deg"] - target["yaw_deg"]),
        )

        # 회피 방향
        if most_blocking["yaw_deg"] > target["yaw_deg"]:
            avoid_dir = "left"
        else:
            avoid_dir = "right"

        # bbox 기반 회피 각도
        x1, _, x2, _ = most_blocking["bbox"]
        half_width_px = (x2 - x1) / 2
        hfov_rad = math.radians(hfov_deg)
        fx = (img_w / 2) / math.tan(hfov_rad / 2)
        half_angle_deg = math.degrees(math.atan(half_width_px / fx))

        raw_turn = half_angle_deg + self.avoid_safety_margin_deg
        clamped_turn = max(
            self.avoid_turn_min_deg,
            min(self.avoid_turn_max_deg, raw_turn),
        )

        # 장애물 거리 기반 동적 전진
        obstacle_distance = most_blocking.get("distance")
        if obstacle_distance is not None and obstacle_distance > 0:
            raw_forward = obstacle_distance + self.avoid_pass_buffer_m
            avoid_forward = min(raw_forward, self.max_forward_per_cycle_m)
            forward_info = f"obstacle={obstacle_distance:.2f}m → forward={avoid_forward:.2f}m"
        else:
            avoid_forward = self.max_forward_per_cycle_m
            forward_info = f"obstacle depth 없음 → fallback {avoid_forward:.2f}m"

        print(
            f"   ↪ 회피 대상: {most_blocking['class']} "
            f"(turn raw={raw_turn:.2f}°, clamped={clamped_turn:.2f}°, {forward_info})"
        )

        executed.append(self._motor_turn(avoid_dir, clamped_turn))
        executed.append(self._motor_move("front", avoid_forward))

        return executed, "reevaluate"

    def _execute_stop_at_target(self, context):
        executed = [self._motor_stop()]
        target = context["target"]
        print(f"   🎯 도착! target={target['class']}, distance={target['distance']:.2f}m")
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

    # ── 메인 진입점 ────────────────────────────────────────────

    def execute(self, action_command):
        """
        action_command 받아서 실제 모터 동작 수행.
        """
        action = action_command.get("action")
        status = action_command.get("status", "success")

        print(f"\n🤖 [System1] action='{action}' 실행")

        # abort: 시스템 에러 → 정지만
        if status == "abort" or action == "abort":
            self._motor_stop()
            return {
                "executed_motions": ["STOP"],
                "next_action_hint": "abort",
                "next_avoidance_attempts": 0,
            }

        # retry: 모터 동작 없이 sender에게 reason 전달
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

        # 행동별 분기
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