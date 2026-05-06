"""
System 1 — MotionExecutor

하위 제어 모듈.
System 2가 보낸 action_command를 받아서:
  1) "어떻게 실행할지" 자체 판단 (회전 먼저? 전진? 회피 각도 얼마?)
  2) 모터 제어 명령 생성
  3) 모터 제어 실행 (현재는 mock — print만)

행동(action) 종류별 처리:
  - "track"           : aligned 보고 회전 또는 전진 결정
  - "avoid_obstacle"  : bbox 기반 회피 각도 계산 후 회전+전진
  - "stop_at_target"  : 모든 모터 정지
  - "wait_user"       : 사용자 메시지 출력하고 대기

핵심 인터페이스:
    executor.execute(action_command)
        -> ExecutionResult dict {
            "executed_motions": [...],   # 실제 한 모터 동작 목록
            "next_action_hint": str,     # main.py에 다음 사이클 힌트
            "next_avoidance_attempts": int,
        }

진짜 통합 시:
    - self._motor_turn(...) 등에서 Arduino 시리얼 명령 전송
    - 현재는 print로만 보여줌 (mock)
"""

import math


class MotionExecutor:
    def __init__(self):
        # 회피 동작 파라미터 (System 1 영역: 어떻게 실행할지)
        self.avoid_safety_margin_deg = 10.0   # 장애물 가장자리 너머 마진
        self.avoid_turn_min_deg = 10.0        # 회피 각도 하한
        self.avoid_turn_max_deg = 30.0        # 회피 각도 상한
        self.avoid_forward_m = 0.5            # 회피 시 전진 거리

        # 한 사이클당 최대 이동/회전 (안전: 짧게 끊어서 카메라 재확인)
        self.max_forward_per_cycle_m = 0.4    # 한 번에 최대 40cm
        self.max_turn_per_cycle_deg = 20.0    # 한 번에 최대 20도

        # 추후: serial.Serial("/dev/ttyACM0", 9600) 등으로 Arduino 연결
        self.motor_log = []  # 디버깅용 모터 명령 로그

        print("✅ System1(MotionExecutor) 준비 완료 (mock 모드)")
        print(f"   max_forward_per_cycle = {self.max_forward_per_cycle_m}m")
        print(f"   max_turn_per_cycle = {self.max_turn_per_cycle_deg}deg")

    # ── 모터 명령 (mock) ───────────────────────────────────────

    def _motor_turn(self, direction, amount_deg):
        """좌/우 회전 명령. 실제 구현 시 Arduino로 시리얼 송신."""
        cmd = f"TURN {direction.upper()} {amount_deg:.2f}deg"
        print(f"   🔧 [모터] {cmd}")
        self.motor_log.append(cmd)
        return cmd

    def _motor_move(self, direction, distance_m):
        """전/후진 명령."""
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
        추적 실행. 정렬 안 되면 회전, 정렬됐으면 전진/후진.
        한 사이클에 한 동작만 (다음 카메라 프레임에서 재평가).
        max_forward_per_cycle_m, max_turn_per_cycle_deg로 한 사이클당 제한.
        """
        target = context["target"]
        config = context["config"]
        executed = []

        # 정렬 우선
        if not target["aligned"]:
            yaw = target["yaw_deg"]
            direction = "left" if yaw < 0 else "right"
            # 한 번에 최대 max_turn_per_cycle_deg까지만 회전
            turn_amount = min(abs(yaw), self.max_turn_per_cycle_deg)
            executed.append(self._motor_turn(direction, turn_amount))
            return executed, "continue"

        # 정렬됐으면 거리 조정
        distance_error = target["distance_error"]
        tolerance = config["distance_tolerance_m"]

        if distance_error > tolerance:
            # 한 번에 최대 max_forward_per_cycle_m까지만 전진
            move_dist = min(distance_error, self.max_forward_per_cycle_m)
            executed.append(self._motor_move("front", move_dist))
            return executed, "continue"
        elif distance_error < -tolerance:
            # 후진도 동일 제한 적용
            move_dist = min(abs(distance_error), self.max_forward_per_cycle_m)
            executed.append(self._motor_move("back", move_dist))
            return executed, "continue"
        else:
            # 정렬됐고 거리도 OK면 사실상 stop_at_target 케이스인데,
            # System 2가 미리 잡아내서 여기 거의 안 옴
            executed.append(self._motor_stop())
            return executed, "done"

    def _execute_avoid_obstacle(self, context):
        """
        회피 실행.
        - 가장 위험한 장애물 선택 (target과 yaw 차이 최소)
        - bbox 폭으로 회피 각도 계산 (마진 + 클램핑)
        - 회전 → 전진 순서로 한 사이클에 둘 다 실행
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

        # 회피 방향 결정
        if most_blocking["yaw_deg"] > target["yaw_deg"]:
            avoid_dir = "left"   # 장애물이 target 오른쪽 → 왼쪽으로 우회
        else:
            avoid_dir = "right"

        # bbox 기반 회피 각도 계산
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

        print(
            f"   ↪ 회피 대상: {most_blocking['class']} "
            f"(raw={raw_turn:.2f}°, clamped={clamped_turn:.2f}°)"
        )

        # 회전 → 전진 순차 실행 (한 사이클에)
        executed.append(self._motor_turn(avoid_dir, clamped_turn))
        executed.append(self._motor_move("front", self.avoid_forward_m))

        return executed, "reevaluate"

    def _execute_stop_at_target(self, context):
        executed = [self._motor_stop()]
        target = context["target"]
        print(f"   🎯 도착! target={target['class']}, distance={target['distance']:.2f}m")
        return executed, "done"

    def _execute_wait_user(self, context):
        executed = [self._motor_stop()]
        blocking = context["blocking_obstacles"]
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

        Returns:
            dict: {
                "executed_motions": list[str],
                "next_action_hint": "continue" | "reevaluate" | "done" | "wait_user" | "abort",
                "next_avoidance_attempts": int,  # 다음 사이클에 넘길 카운트
            }
        """
        action = action_command.get("action")
        status = action_command.get("status", "success")

        print(f"\n🤖 [System1] action='{action}' 실행")

        # abort: 그냥 정지만
        if status == "abort" or action == "abort":
            self._motor_stop()
            return {
                "executed_motions": ["STOP"],
                "next_action_hint": "abort",
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
            next_attempts = 0  # 정상 추적 복귀 시 카운트 리셋
        elif action == "avoid_obstacle":
            executed, hint = self._execute_avoid_obstacle(context)
            next_attempts = current_attempts + 1
        elif action == "stop_at_target":
            executed, hint = self._execute_stop_at_target(context)
            next_attempts = 0
        elif action == "wait_user":
            executed, hint = self._execute_wait_user(context)
            next_attempts = current_attempts  # 변경 없음
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