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
  - "emergency_stop"  : 긴급 제동 (제거됨 - system2에서 호출 안함)
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
        self.avoid_safety_margin_deg = 12.0 # 마진 증가
        self.avoid_turn_min_deg = 15.0      # 최소 회전각 증가
        self.avoid_turn_max_deg = 45.0      # 최대 회전각 증가 (30 -> 45)
        self.avoid_pass_buffer_m = 0.2    # 장애물 너머 여유 마진

        # 한 사이클당 최대 이동/회전
        self.max_forward_per_cycle_m = 0.4
        self.max_turn_per_cycle_deg = 20.0

        self.motor_log = []

        print("✅ System1(MotionExecutor) 준비 완료 (mock 모드)")

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
        [개선] 다중 장애물 대응 로직
        1. 모든 blocking_obstacles의 yaw 범위를 계산
        2. 이들을 모두 비껴갈 수 있는 통합 회피 각도 산출
        """
        target = context["target"]
        blocking = context["blocking_obstacles"]
        img_w = context["image"]["width"]
        hfov_deg = context["image"]["hfov_deg"]
        executed = []

        if not blocking:
            return [self._motor_stop()], "reevaluate"

        # 1. 모든 장애물의 각도 범위 파악
        # yaw_min, yaw_max: 로봇 정면 기준 장애물 덩어리의 왼쪽 끝과 오른쪽 끝
        all_yaw_min = 999
        all_yaw_max = -999
        min_dist = 999

        for o in blocking:
            yaw = o["yaw_deg"]
            width = o.get("width_deg", 10.0) # 폭 정보 없으면 기본 10도
            
            o_min = yaw - (width / 2)
            o_max = yaw + (width / 2)
            
            if o_min < all_yaw_min: all_yaw_min = o_min
            if o_max > all_yaw_max: all_yaw_max = o_max
            if o["distance"] < min_dist: min_dist = o["distance"]

        # 2. 회피 방향 결정 (장애물 덩어리의 중심이 target보다 어디에 있나)
        cluster_center_yaw = (all_yaw_min + all_yaw_max) / 2
        if cluster_center_yaw > target["yaw_deg"]:
            avoid_dir = "left"
            # 왼쪽으로 피하려면 장애물의 왼쪽 끝(all_yaw_min)보다 더 왼쪽으로 가야함
            needed_turn = abs(all_yaw_min - target["yaw_deg"]) + self.avoid_safety_margin_deg
        else:
            avoid_dir = "right"
            # 오른쪽으로 피하려면 장애물의 오른쪽 끝(all_yaw_max)보다 더 오른쪽으로 가야함
            needed_turn = abs(all_yaw_max - target["yaw_deg"]) + self.avoid_safety_margin_deg

        clamped_turn = max(self.avoid_turn_min_deg, min(self.avoid_turn_max_deg, needed_turn))

        # 3. 전진 거리 (가장 가까운 장애물 기준)
        raw_forward = min_dist + self.avoid_pass_buffer_m
        avoid_forward = min(raw_forward, self.max_forward_per_cycle_m)

        print(f"   🚧 [Multi-Avoid] 장애물 {len(blocking)}개 감지 (범위: {all_yaw_min:.1f}°~{all_yaw_max:.1f}°)")
        print(f"   ↪ 회피: {avoid_dir} {clamped_turn:.2f}°, 전진 {avoid_forward:.2f}m")

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
        action = action_command.get("action")
        status = action_command.get("status", "success")

        print(f"\n🤖 [System1] action='{action}' 실행")

        if status == "abort" or action == "abort":
            self._motor_stop()
            return {"executed_motions": ["STOP"], "next_action_hint": "abort", "next_avoidance_attempts": 0}

        if action == "retry":
            reason = action_command.get("reason", "unknown")
            return {"executed_motions": [], "next_action_hint": "retry", "next_action_reason": reason, "next_avoidance_attempts": 0}

        context = action_command.get("context", {})
        current_attempts = context.get("avoidance_attempts", 0)

        if action == "track":
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
        elif action == "search_rotate":
            # system2가 요청한 탐색 회전 실행
            direction = context.get("direction", "right")
            turn_deg = context.get("turn_deg", 30.0)
            executed = [self._motor_turn(direction, turn_deg)]
            hint = "reevaluate"
            next_attempts = 0
        else:
            self._motor_stop()
            executed = ["STOP"]
            hint = "abort"
            next_attempts = 0

        return {
            "executed_motions": executed,
            "next_action_hint": hint,
            "next_avoidance_attempts": next_attempts,
        }
