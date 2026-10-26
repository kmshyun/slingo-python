# SLingo 아바타 수어 시범 (slingo_avatar)

SLingo 앱에 **본인 얼굴 없이 귀여운 VRM 3D 아바타가 수어를 시범**하는 기능.
기존 MediaPipe 랜드마크 데이터(`../slingo_dataset_maker/dataset`)를 아바타 모션으로 변환한다.

## 전체 파이프라인

```
[.npy 랜드마크]  →  export_to_unity.py  →  [motion_json/*.json]  →  Unity(VRM+C#)  →  실시간 아바타 / mp4
   (수집 완료)          (완료)                 (완료, 12단어)         (진행)          (앱 임베드 / 폴백)
```

## 폴더 구조
- `motion_json/` — 단어별 모션 JSON 12개. 좌표계: **어깨중점 원점, 어깨너비=1, y-up, +z 뷰어쪽**.
- `unity_scripts/` — Unity 에 넣을 C#.
  - `SignClipData.cs` — JSON 데이터 클래스.
  - `SignAvatarPlayer.cs` — VRM Humanoid 에 모션 재생(팔 IK + 손가락 회전).
- `tools/` — `export_to_unity.py`(변환기), `viz_clip.py`(스켈레톤 미리보기).
- `preview/` — 복원 모션 확인용 mp4/PNG.
- `avatar_view.py` — 가벼운 cv2 2D 아바타 뷰어 + **웹캠 수어 인식**(아래).

## 웹캠 수어 인식 (avatar_view.py --live)

웹캠 앞에서 수어를 하면 2D 아바타가 따라 움직이고, 손을 내리면 그 구간을 하나의
제스처로 보고 12단어 중 하나로 판정한다.

```bash
cd "/Users/shkim/Desktop/2026 전국전람회/code"
venv/bin/python slingo_avatar/avatar_view.py --live            # 아바타 + 인식
venv/bin/python slingo_avatar/avatar_view.py --live --cam      # 웹캠 원본도 같이
venv/bin/python slingo_avatar/avatar_view.py --live --no-recog # 아바타만(모델 안 씀)
venv/bin/python slingo_avatar/avatar_view.py --live --conf 0.7 # 확신 임계값 조정
```

조작: **SPACE** 지금 버퍼 즉시 판정 · **C** 버퍼 비우기 · **M** 카메라 패널 · **Q** 종료

동작 방식
- 검출: MediaPipe Tasks(`models/*.task`), `collect.py` 와 동일한 설정·전처리(139차원 특징).
- 구간 분할: 손 5프레임 연속 검출 → 녹화 시작, 손 8프레임 연속 없음 → 구간 끝 → 판정
  (끝 3프레임은 잘라냄). 10프레임 미만이면 판정 안 함.
- 분류: `../slingo_dataset_maker/checkpoints/best_lstm.pt` (BiLSTM 192x2, 12라벨).
  결과는 상위 3개를 확률과 함께 표시하고, `--conf` 미만이면 `?` 로 보여준다.
- 15fps 로 상한을 둬서 학습 데이터와 시퀀스 속도를 맞춘다.

> 모델을 다시 학습하면(`slingo_dataset_maker/train.py`) 체크포인트만 갱신되고
> 이 뷰어는 그대로 새 라벨을 읽는다.

## 학습용 아바타 영상 만들기 (--record)

웹캠 앞에서 수어를 하면 아바타가 따라 하고, 그 **아바타 화면만** mp4 로 저장한다.

```bash
cd "/Users/shkim/Desktop/2026 전국전람회/code"
venv/bin/python slingo_avatar/avatar_view.py --live --record
```

**R** → 3초 카운트다운(화면에 3·2·1) → **5초 녹화** → `videos/001.mp4` 저장.
다음에 R 을 누르면 `002.mp4`, `003.mp4` ... 자동 연번. 이름은 나중에 직접 바꾸면 된다.

- 저장되는 것: 720x720 아바타 캔버스만. HUD(인식 결과·카운트다운·REC 표시)와
  `--cam` 웹캠 패널은 **영상에 들어가지 않는다**.
- 15fps · H.264(avc1, 실패 시 mp4v) · 항상 정확히 75프레임 = 5.00초.
  렌더가 느려도 벽시계 기준으로 프레임을 채우므로 재생 속도가 느려지지 않는다.
- 녹화 중 **R** 또는 **Q** = 취소(만들던 파일 삭제).

옵션: `--outdir <폴더>` `--seconds 5` `--countdown 3` `--fps 15`

```bash
# 예: 3초짜리를 다른 폴더에
venv/bin/python slingo_avatar/avatar_view.py --live --record --seconds 3 --outdir ~/Desktop/수어영상
```

## 데이터 재생성
```bash
cd "/Users/shkim/Desktop/2026 전국전람회/code"
venv/bin/python slingo_avatar/tools/export_to_unity.py         # 단어별 대표클립 1개
venv/bin/python slingo_avatar/tools/export_to_unity.py --all   # 전 클립
```

---

## Unity 세팅 (사용자 작업)

### 1. 귀여운 아바타 만들기 — VRoid Studio (무료)
1. https://vroid.com/en/studio 에서 VRoid Studio 다운로드/설치.
2. 캐릭터 생성 → 얼굴/머리/의상 커스터마이즈 (귀엽게!).
3. **Export → VRM** (VRM0 또는 VRM1). 파일명 예: `slingo_avatar.vrm`.

### 2. Unity 프로젝트
1. Unity Hub → Unity **6000.x LTS** (또는 2022.3 LTS) 설치, 3D(URP) 프로젝트 생성.
2. **UniVRM** 임포트: https://github.com/vrm-c/UniVRM/releases 에서 최신 `.unitypackage`
   (VRM0/VRM1 맞는 것) → Assets 에 드래그.
3. Package Manager → **Newtonsoft Json** (`com.unity.nuget.newtonsoft-json`) 설치.
4. `.vrm` 을 Assets 에 드래그 → 프리팹 생성 → 씬에 배치.
   - Inspector 에서 Animator 의 Avatar 가 **Humanoid** 인지 확인.

### 3. 스크립트 연결
1. `unity_scripts/*.cs` 를 `Assets/Scripts/` 에 복사.
2. `motion_json/*.json` 을 `Assets/Resources/motion_json/` 에 복사 (Resources 폴더 필수).
3. 아바타 GameObject 에 **SignAvatarPlayer** 컴포넌트 추가.
4. Animator 에 **IK Pass 켠 레이어**가 있는 Animator Controller 연결
   (빈 컨트롤러 + Base Layer 톱니 → IK Pass 체크).
5. Inspector 에서 `wordToPlay = 안녕`, Play ▶.

### 4. 라이브 튜닝 (아바타 나오면 같이)
Play 중 SignAvatarPlayer 인스펙터에서 조정:
- `mirrorX` — 좌우 반전(데이터가 거울이라 켜는 게 기본).
- `reachScale` / `depthOffset` — 손 위치·깊이 보정.
- `fingerWeight` — 손가락 회전 세기.

> ⚠️ **손가락 정확도가 품질 핵심.** 수어는 손모양 조금만 틀려도 뜻이 바뀐다.
> 단어별로 눈으로 대조하며 튜닝할 것.

---

## 다음 단계
1. ~~모션 데이터 검증~~ ✅
2. ~~.npy → JSON 변환~~ ✅
3. ~~Unity C# 리타게팅 코드~~ ✅ (초안, 아바타로 튜닝 필요)
4. VRoid 아바타 + Unity 세팅 ← **지금 여기**
5. 데스크톱에서 손가락 정확도 튜닝
6. **경로 A**: Unity as a Library → 앱 임베드 → `SignLearnDetailView` 연결
   **경로 B(폴백)**: Unity Recorder → 단어별 mp4 → 앱 Media3 재생
7. 앱 티어 해금 UI/로직 (`AppState.kt` 티어 시스템에 연결)
