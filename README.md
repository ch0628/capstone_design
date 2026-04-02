# Capstone Design

캡스톤 디자인 프로젝트용 저장소입니다.  
VLM과 YOLO를 결합하여 이미지 기반으로 목표 객체를 추론하고, 객체 탐지 결과를 바탕으로 다음 액션을 결정하는 실험 코드를 포함합니다.

## Project Structure

- `environment.yml` : Conda 환경 설정 파일
- `test.jpg` : 테스트용 이미지
- `*.py` : VLM + YOLO 추론 실행 코드

## Environment Setup

먼저 conda 환경을 생성하고 활성화합니다.

```bash
conda env create -f environment.yml
conda activate vla_v2
```

이후 필요한 패키지를 별도로 설치합니다.

```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main"
```