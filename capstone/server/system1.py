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
        self.avoid_safety_margin_deg = 15.0
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
            return executed, "continue", 0.0

        distance_error = target["distance_error"]
        tolerance = config["distance_tolerance_m"]

        if distance_error > tolerance:
            move_dist = min(distance_error, self.max_forward_per_cycle_m)
            executed.append(self._motor_move("front", move_dist))
            return executed, "continue", move_dist
        elif distance_error < -tolerance:
            move_dist = min(abs(distance_error), self.max_forward_per_cycle_m)
            executed.append(self._motor_move("back", move_dist))
            return executed, "continue", 0.0
        else:
            executed.append(self._motor_stop())
            return executed, "done", 0.0

    def _execute_avoid_obstacle(self, context):
        """
        가장 가까운 blocking 장애물의 측정 상태(depth_measurement) 기준으로 분기:
          measured + 0.3m 초과       → (b) 직진해서 거리 좁힘
          measured + 0.3m 이하       → (c) 회피
          estimated_odometry         → (c) 즉시 회피 (측정 실패 = 임계 도달)
          estimated_failed           → (c) 즉시 회피 (캐시 없음, 0.3m 가정)
        """
        target = context["target"]
        blocking = context["blocking_obstacles"]
        img_w = context["image"]["width"]
        hfov_deg = context["image"]["hfov_deg"]
        executed = []

        # 가장 위험한 장애물 = 거리가 가장 가까운 것
        most_blocking = min(blocking, key=lambda o: o["distance"])
        obstacle_distance = most_blocking["distance"]
        depth_status = most_blocking.get("depth_measurement", "measured")

        # ── 분기 (b): 측정 성공 + 여유 있음 → 직진 ──────────────
        if depth_status == "measured" and obstacle_distance > 0.3:
            move_dist = min(obstacle_distance - 0.3, self.max_forward_per_cycle_m)
            print(
                f"   ↪ 회피 대상: {most_blocking['class']} "
                f"(obstacle={obstacle_distance:.2f}m measured > 0.3m → 직진 {move_dist:.2f}m로 거리 좁힘)"
            )
            executed.append(self._motor_move("front", move_dist))
            return executed, "reevaluate", move_dist

        # ── 분기 (c): 회피 진입 ─────────────────────────────────
        if depth_status == "estimated_failed":
            # 캐시 없음 → 0.3m 가정 (v5 fallback 유지)
            obstacle_distance_for_avoid = 0.3
            print(
                f"   ↪ 회피 대상: {most_blocking['class']} "
                f"(depth 측정 실패, 캐시 없음 → 0.3m 가정하고 회피)"
            )
        elif depth_status == "estimated_odometry":
            # 측정 실패하지만 odometry 추정값 있음 → 그 값 사용
            obstacle_distance_for_avoid = obstacle_distance
            print(
                f"   ↪ 회피 대상: {most_blocking['class']} "
                f"(obstacle={obstacle_distance:.2f}m odometry → 즉시 회피)"
            )
        else:
            # measured + 0.3m 이하 → 그 측정값으로 회피
            obstacle_distance_for_avoid = obstacle_distance
            print(
                f"   ↪ 회피 대상: {most_blocking['class']} "
                f"(obstacle={obstacle_distance:.2f}m measured ≤ 0.3m → 회피)"
            )

        # 회피 방향
        if most_blocking["yaw_deg"] > target["yaw_deg"]:
            avoid_dir = "left"
        else:
            avoid_dir = "right"

        # bbox 기반 회피 각도 (상한 제거, 하한만 유지)
        x1, _, x2, _ = most_blocking["bbox"]
        half_width_px = (x2 - x1) / 2
        hfov_rad = math.radians(hfov_deg)
        fx = (img_w / 2) / math.tan(hfov_rad / 2)
        half_angle_deg = math.degrees(math.atan(half_width_px / fx))

        raw_turn = half_angle_deg + self.avoid_safety_margin_deg
        clamped_turn = max(self.avoid_turn_min_deg, raw_turn)

        # 대각선 보정 전진: 회전각만큼 정면 진행이 줄어드므로 더 길게 이동
        turn_rad = math.radians(clamped_turn)
        if turn_rad >= math.pi / 2:
            diagonal_distance = obstacle_distance_for_avoid + self.avoid_pass_buffer_m
        else:
            diagonal_distance = (obstacle_distance_for_avoid + self.avoid_pass_buffer_m) / math.cos(turn_rad)
        avoid_forward = diagonal_distance

        # 복귀 회전: 회피 회전과 같은 각도, 반대 방향
        return_dir = "right" if avoid_dir == "left" else "left"

        print(
            f"   ↪ turn={clamped_turn:.2f}°({avoid_dir}), "
            f"diagonal_forward={avoid_forward:.2f}m, "
            f"return turn {clamped_turn:.2f}°"
        )

        executed.append(self._motor_turn(avoid_dir, clamped_turn))
        executed.append(self._motor_move("front", avoid_forward))
        executed.append(self._motor_turn(return_dir, clamped_turn))

        return executed, "reevaluate", avoid_forward

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
        forward_distance = 0.0
        if action == "emergency_stop":
            executed = [self._motor_stop()]
            print("🚨 [긴급 제동] 로봇 바로 앞에 장애물이 감지되어 정지합니다.")
            hint = "reevaluate"
            next_attempts = current_attempts
        elif action == "track":
            executed, hint, forward_distance = self._execute_track(context)
            next_attempts = 0
        elif action == "avoid_obstacle":
            executed, hint, forward_distance = self._execute_avoid_obstacle(context)
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
            "forward_distance_this_cycle": forward_distance,
        }