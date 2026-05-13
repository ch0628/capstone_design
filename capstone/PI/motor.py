"""
motor.py — RPi에서 실행되는 모터 제어 모듈

서버(system1.MotionExecutor)는 "TURN LEFT 15.30deg", "MOVE FRONT 0.500m", "STOP" 같은
문자열로 명령을 보낸다. 이 모듈은 그 문자열을 받아 PCA9685로 실제 모터를 돌린다.

인코더가 없으므로 시간 기반 제어:
  - MOVE 거리(m) → 거리 / LINEAR_SPEED_MPS 초만큼 전진/후진
  - TURN 각도(deg) → 각도 / ANGULAR_SPEED_DPS 초만큼 회전
  - STOP → 즉시 정지

캘리브레이션 결과 (실측):
  - 채널 매핑: 0=LF, 1=LB, 2=RB, 3=RF
  - BASE_SPEED = 0x6000 에서:
    - 직진 (LEFT_TRIM=0.88 적용): 약 0.285 m/s
    - 회전 (양쪽 풀): 약 45 deg/s
  - 우회전이 좌회전보다 살짝 부족할 수 있음 (오른쪽 모터가 약함).
    카메라 피드백으로 보정되니 무시 가능.
"""

import time
import re
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_character_lcd.character_lcd_i2c import Character_LCD_I2C


# ── 속도 설정 (실측 캘리브레이션) ──────────────────────────
BASE_SPEED = 0xC000        # 또는 사용 중인 새 값

# 직진 trim
LEFT_TRIM = 0.869          # ★ 0.88 → 0.869
RIGHT_TRIM = 1.00

LEFT_SPEED = int(BASE_SPEED * LEFT_TRIM)
RIGHT_SPEED = int(BASE_SPEED * RIGHT_TRIM)

# 시간 환산 상수 (실측)
LINEAR_SPEED_MPS = 0.400   # ★ 0.285 → 0.400
ANGULAR_SPEED_DPS = 135.0  # ★ 45.0 → 135.0

# 모터 동작 후 settle delay (관성 잦아들기 위해 대기)
SETTLE_DELAY_S = 0.15

# 너무 짧은 동작은 모터가 못 따라가니 최소 시간 보장
MIN_ACTION_TIME_S = 0.05


# ── PCA9685 채널 매핑 (실측 진단) ─────────────────────────
LEFT_FORWARD = 0    # 채널 0 = 왼쪽 정방향
LEFT_BACKWARD = 1   # 채널 1 = 왼쪽 역방향
RIGHT_BACKWARD = 2  # 채널 2 = 오른쪽 역방향 (주의: 정방향 아님)
RIGHT_FORWARD = 3   # 채널 3 = 오른쪽 정방향


# LCD 설정
LCD_COLUMNS = 16
LCD_ROWS = 2


class MotorDriver:
    def __init__(self, use_lcd=True):
        # I2C 초기화
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(self.i2c)
        self.pca.frequency = 60

        # LCD (없으면 스킵)
        self.lcd = None
        if use_lcd:
            try:
                self.lcd = Character_LCD_I2C(self.i2c, LCD_COLUMNS, LCD_ROWS)
                self.lcd.backlight = True
                self._display_msg("Motor ready")
            except Exception as e:
                print(f"⚠️  LCD 초기화 실패 (무시하고 계속): {e}")
                self.lcd = None

        print(f"✅ MotorDriver 준비 완료")
        print(f"   직진용: LEFT={hex(LEFT_SPEED)}, RIGHT={hex(RIGHT_SPEED)}")
        print(f"   회전용: 양쪽 {hex(BASE_SPEED)}")
        print(f"   LINEAR={LINEAR_SPEED_MPS}m/s, ANGULAR={ANGULAR_SPEED_DPS}deg/s")

    # ── 저수준 제어 ──────────────────────────────────────────

    def _display_msg(self, line1, line2=""):
        if self.lcd is None:
            return
        try:
            self.lcd.clear()
            self.lcd.message = f"{line1}\n{line2}"
        except Exception:
            pass

    def _set_motors(self, left_fwd, left_back, right_fwd, right_back):
        """
        의미 단위로 4채널 PWM 설정.
        채널 번호가 아닌 '어느 모터 어느 방향'으로 받음 → 헷갈릴 일 없음.
        """
        self.pca.channels[LEFT_FORWARD].duty_cycle = left_fwd
        self.pca.channels[LEFT_BACKWARD].duty_cycle = left_back
        self.pca.channels[RIGHT_FORWARD].duty_cycle = right_fwd
        self.pca.channels[RIGHT_BACKWARD].duty_cycle = right_back

    def _stop_all(self):
        self._set_motors(0, 0, 0, 0)

    # ── 동작 (블로킹) ────────────────────────────────────────

    def move(self, direction, distance_m):
        """
        전진 또는 후진. distance_m만큼 이동할 때까지 블로킹.
        TRIM 적용 (휨 보정).

        Args:
            direction: "front" | "back"
            distance_m: 거리(미터). 양수.
        """
        if distance_m <= 0:
            print(f"   ⚠️  거리 0 이하 무시: {distance_m}m")
            return

        duration_s = max(MIN_ACTION_TIME_S, distance_m / LINEAR_SPEED_MPS)

        if direction == "front":
            self._display_msg("Moving...", f"FRONT {distance_m:.2f}m")
            print(f"   🔧 [모터] MOVE FRONT {distance_m:.3f}m ({duration_s:.2f}s)")
            self._set_motors(LEFT_SPEED, 0, RIGHT_SPEED, 0)
        elif direction == "back":
            self._display_msg("Moving...", f"BACK {distance_m:.2f}m")
            print(f"   🔧 [모터] MOVE BACK {distance_m:.3f}m ({duration_s:.2f}s)")
            self._set_motors(0, LEFT_SPEED, 0, RIGHT_SPEED)
        else:
            print(f"   ⚠️  알 수 없는 방향: {direction}")
            return

        time.sleep(duration_s)
        self._stop_all()
        time.sleep(SETTLE_DELAY_S)

    def turn(self, direction, angle_deg):
        """
        제자리 회전. angle_deg만큼 돌 때까지 블로킹.
        TRIM 미적용 (회전력 확보 위해 양쪽 풀파워).

        Args:
            direction: "left" | "right"
            angle_deg: 각도(도). 양수.
        """
        if angle_deg <= 0:
            print(f"   ⚠️  각도 0 이하 무시: {angle_deg}deg")
            return

        duration_s = max(MIN_ACTION_TIME_S, angle_deg / ANGULAR_SPEED_DPS)

        if direction == "left":
            self._display_msg("Turning...", f"LEFT {angle_deg:.1f}deg")
            print(f"   🔧 [모터] TURN LEFT {angle_deg:.2f}deg ({duration_s:.2f}s)")
            # 좌회전: 왼쪽 후진 + 오른쪽 전진
            self._set_motors(0, BASE_SPEED, BASE_SPEED, 0)
        elif direction == "right":
            self._display_msg("Turning...", f"RIGHT {angle_deg:.1f}deg")
            print(f"   🔧 [모터] TURN RIGHT {angle_deg:.2f}deg ({duration_s:.2f}s)")
            # 우회전: 왼쪽 전진 + 오른쪽 후진
            self._set_motors(BASE_SPEED, 0, 0, BASE_SPEED)
        else:
            print(f"   ⚠️  알 수 없는 방향: {direction}")
            return

        time.sleep(duration_s)
        self._stop_all()
        time.sleep(SETTLE_DELAY_S)

    def stop(self):
        self._display_msg("STOP")
        print(f"   🔧 [모터] STOP")
        self._stop_all()

    # ── 명령 문자열 파싱 및 실행 ─────────────────────────────

    def execute_command(self, cmd_str):
        """
        서버가 보낸 모터 명령 문자열 1개를 실행.

        지원 형식:
          "MOVE FRONT 0.500m"
          "MOVE BACK 0.300m"
          "TURN LEFT 15.30deg"
          "TURN RIGHT 22.00deg"
          "STOP"
        """
        cmd_str = cmd_str.strip()

        if cmd_str == "STOP":
            self.stop()
            return True

        m = re.match(r"MOVE\s+(FRONT|BACK)\s+([\d.]+)m", cmd_str, re.IGNORECASE)
        if m:
            self.move(m.group(1).lower(), float(m.group(2)))
            return True

        m = re.match(r"TURN\s+(LEFT|RIGHT)\s+([\d.]+)deg", cmd_str, re.IGNORECASE)
        if m:
            self.turn(m.group(1).lower(), float(m.group(2)))
            return True

        print(f"   ⚠️  파싱 실패: '{cmd_str}'")
        return False

    def execute_motions(self, motion_list):
        """서버 응답의 executed_motions 리스트를 순서대로 실행."""
        if not motion_list:
            print("   (실행할 모터 명령 없음)")
            return

        print(f"🤖 [모터 실행] {len(motion_list)}개 명령 시작")
        for cmd in motion_list:
            self.execute_command(cmd)
        print(f"✅ [모터 실행] 완료")

    def cleanup(self):
        try:
            self._stop_all()
            self._display_msg("Shutdown")
            time.sleep(0.1)
            self.pca.deinit()
        except Exception as e:
            print(f"⚠️  cleanup 에러 (무시): {e}")


# ── 단독 실행: 동작 테스트 ────────────────────────────────────
if __name__ == "__main__":
    """
    sender.py 없이 모터만 단독 테스트.
    명령 파싱이 잘 되는지, 시간 환산이 맞는지 확인용.
    """
    driver = MotorDriver()

    try:
        print("\n=== 동작 테스트 ===")
        print("바닥에 놓고 진행하세요.\n")

        input("Enter: MOVE FRONT 0.5m >>> ")
        driver.execute_command("MOVE FRONT 0.500m")

        input("Enter: TURN LEFT 90deg >>> ")
        driver.execute_command("TURN LEFT 90.00deg")

        input("Enter: TURN RIGHT 90deg >>> ")
        driver.execute_command("TURN RIGHT 90.00deg")

        input("Enter: MOVE BACK 0.3m >>> ")
        driver.execute_command("MOVE BACK 0.300m")

        input("Enter: STOP >>> ")
        driver.execute_command("STOP")

        print("\n✅ 테스트 완료!")

    except KeyboardInterrupt:
        print("\n중단됨")
    finally:
        driver.cleanup()