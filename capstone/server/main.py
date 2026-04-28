"""
main.py — 오케스트라 지휘자

System 2와 System 1을 연결한다.
- 카메라 이미지 / depth 입력 받기 (현재는 임시값)
- System 2 호출 → action_command
- System 1 호출 → 모터 실행
- next_action_hint 보고 다음 사이클 어떻게 돌릴지 판단
- 회피 카운트 누적 관리

진짜 운영 시:
  - while 루프 돌면서 매 사이클마다 카메라 새로 찍고 plan→execute 반복
  - depth 카메라에서 current_distance 읽어옴
현재(테스트용):
  - test_cases.json에서 case 하나 읽어서 단발 실행
  - current_distance는 임시 1.7m
"""

import json
from system2 import VisionPlanner
from system1 import MotionExecutor


# 임시 거리값 (depth 카메라 미연결 상태)
MOCK_CURRENT_DISTANCE_M = 1.7


def run_single_case(planner, executor, case, attempts=0):
    """
    단발 테스트: 한 이미지에 대해 plan → execute 한 번 실행.
    """
    print("=" * 60)
    print(f"🎬 케이스 실행: command='{case['command']}'")
    print("=" * 60)

    # 1. System 2: 행동 결정
    action_command = planner.plan(
        image_path=case["image_path"],
        command=case["command"],
        current_distance=MOCK_CURRENT_DISTANCE_M,
        avoidance_attempts=attempts,
    )

    # 2. System 1: 행동 실행
    execution = executor.execute(action_command)

    # 3. 결과 정리
    result = {
        "action_command": _strip_internal(action_command),
        "execution": execution,
    }
    return result


def _strip_internal(action_command):
    """출력용으로 너무 큰 raw bbox 등 정리 (디버깅 가독성)."""
    if "context" in action_command:
        ctx = action_command["context"]
        # bbox는 round해서 출력
        if "target" in ctx and "bbox" in ctx["target"]:
            ctx["target"]["bbox"] = [round(v, 1) for v in ctx["target"]["bbox"]]
        for o in ctx.get("obstacles", []):
            if "bbox" in o:
                o["bbox"] = [round(v, 1) for v in o["bbox"]]
        for o in ctx.get("blocking_obstacles", []):
            if "bbox" in o:
                o["bbox"] = [round(v, 1) for v in o["bbox"]]
        # 로그 가독성을 위해 출력에서만 숨김 (System 1은 이미 사용 완료)
        ctx.pop("image", None)
        ctx.pop("config", None)
    return action_command


if __name__ == "__main__":
    file_path = "test_cases.json"
    with open(file_path, "r", encoding="utf-8") as f:
        all_cases = json.load(f)

    case_id = "ob3"
    case = all_cases[case_id]

    # 두 시스템 초기화
    planner = VisionPlanner()
    executor = MotionExecutor()

    # 단발 실행
    result = run_single_case(planner, executor, case, attempts=0)

    # 저장
    all_cases[case_id]["actual"] = result
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(all_cases, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 60)
    print("📌 [최종 결과]")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f"\n🎯 [예상 결과값 (참고)]: {case.get('expected', 'N/A')}")