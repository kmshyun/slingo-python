# SLingo — 수어 인식 (Python)

2026 전국과학전람회 출품작 **SLingo**의 Python 파트입니다. 웹캠/영상에서 손·포즈 랜드마크를 추출하고 LSTM 기반 모델로 수어 단어를 인식하며, 인식 결과를 VRM 아바타 모션으로 보여주는 코드를 포함합니다.

## 구성

| 경로 | 설명 |
|------|------|
| `code/` | 핵심 수어 인식·학습 코드 (데이터 수집, 증강, 학습, 추론) |
| `code/final.py` | 메인 실행 스크립트 |
| `code/collect.py` | 수어 데이터 수집 |
| `code/2learn.py`, `3learn_aug.py`, `augment.py` | 학습 / 데이터 증강 |
| `code/checkpoints/` | 학습된 LSTM 체크포인트 (`.pt`, Git LFS) |
| `slingo_avatar/` | VRM 아바타 모션 렌더링 / 미리보기 |
| `knn_model/` | 초기 KNN·CNN 실험 모델 (`.keras`, `.h5`, Git LFS) |
| `susi.py` | 보조 스크립트 |

## 대용량 파일 (Git LFS)

학습된 모델과 아바타 에셋(`.keras`, `.h5`, `.pt`, `.vrm`, `.blend`)은 **Git LFS**로 관리됩니다. 클론 시:

```bash
git lfs install
git clone <repo-url>
```

## 실행

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt   # (의존성 목록은 코드 상단 import 참고)
python code/final.py
```

> 참고: 원본 수집 이미지 데이터셋(`knn_model/A/`)과 로컬 가상환경은 저장소에서 제외되어 있습니다.
