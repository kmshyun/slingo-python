# -*- coding: utf-8 -*-
"""
SLingo 수어 데이터셋 수집기 (standalone)
=========================================

기존 1main.py 는 MediaPipe '레거시(mp.solutions)' API 로 랜드마크를 뽑았지만,
SLingo 안드로이드 앱은 MediaPipe 'Tasks' API(hand_landmarker.task / pose_landmarker.task)
로 추론한다. 두 파이프라인은 내부 모델이 달라서 손 z(깊이)·handedness 분포가 어긋나고,
그 결과 학습은 잘 돼도 앱에서는 인식이 안 되는 train/inference mismatch 가 생긴다.

이 프로그램은 **앱과 완전히 동일한 조건**으로 데이터를 수집한다.
  - 동일한 .task 모델 (app/src/main/assets 에서 그대로 복사)
  - 동일한 전처리: 좌우반전(거울) 프레임, RunningMode.IMAGE, num_hands=2, conf=0.5
  - 동일한 139차원 특징 순서 (SignFeatureExtractor.kt / 2learn.py 와 일치)

출력: dataset/<라벨>/clip_XXXX.npy  (2learn.py 가 그대로 학습에 사용)

의존성:  mediapipe>=0.10.20, opencv-python, numpy, pillow
실행:    python collect.py                 # 기본 20개 라벨, 수동 모드
         python collect.py --auto          # 자동 캡처(손 나타나면 녹화, 사라지면 저장)
         python collect.py --out ../dataset # 출력 폴더 지정
"""

import argparse
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# =========================================================================
# 설정
# =========================================================================
HERE = Path(__file__).resolve().parent
HAND_TASK = HERE / "models" / "hand_landmarker.task"
POSE_TASK = HERE / "models" / "pose_landmarker.task"

# dataset 폴더가 비어 있을 때만 쓰는 초기 라벨 목록.
# 평소에는 dataset 하위 폴더를 읽어서 쓴다(아래 default_labels 참고) —
# 하드코딩해 두면 나중에 추가한 클래스를 수집할 수 없게 된다.
SEED_LABELS = [
    "사람",
    "시간",
    "친구",
    "가족",
    "마음",
    "집",
    "물",
    "도움",
    "약속",
    "선물",
    "오늘",
    "안녕",
]


def default_labels(out_root: Path):
    """dataset 에 이미 있는 클래스를 라벨 목록으로 쓴다."""
    if out_root.exists():
        found = sorted(
            unicodedata.normalize("NFC", p.name)
            for p in out_root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
        if found:
            return found
    return list(SEED_LABELS)

# 한글을 못 그릴 때(폰트 없음) HUD 용 로마자 폴백
ROMAJA = {
    "아빠": "appa", "먹다": "meokda", "친구": "chingu", "주다": "juda",
    "가다": "gada", "좋다": "jota", "싫어하다": "sireohada", "안녕": "annyeong",
    "배고프다": "baegopeuda", "보다": "boda", "사랑": "sarang", "피곤하다": "pigonhada",
    "엄마": "eomma", "밥": "bap", "학교": "hakgyo", "고마워": "gomawo",
    "나": "na", "언제": "eonje", "왜": "wae", "너": "neo",
}

# --- 앱 추론 설정과 일치시키는 값들 (건드리지 말 것) ---
DET_CONF = 0.5          # SignCamera.kt 와 동일
TRK_CONF = 0.5
PRESENCE_CONF = 0.5
NUM_HANDS = 2
HAND_MIN_BOX = 0.03     # 너무 작은 손 검출 제거 (SignFeatureExtractor 와 동일)

# --- 수집 정책 (데이터 품질) ---
FPS_TARGET = 15.0       # 앱은 ~15fps 로 상한 → 시퀀스 속도를 맞춤
MIN_SAVE_FRAMES = 10    # 이보다 짧은 클립은 저장 안 함
PREPARE_SEC = 0.7       # r 누른 뒤 준비 시간
TRIM_TAIL = 3           # 클립 끝에서 손 빠지는 프레임 잘라냄
START_CONSEC = 5        # 연속 '손 있음' 프레임 → 녹화 시작 확정
STOP_CONSEC = 8         # 연속 '손 없음' 프레임 → 녹화 종료(자동저장) 판정
MAX_TEMP_SEC = 3.0      # 시작 확정 전 임시버퍼 최대 길이

CAP_W, CAP_H = 640, 480  # 앱 분석 해상도(ANALYSIS_SIZE)와 동일


# =========================================================================
# 한글 텍스트 렌더링 (OpenCV putText 는 한글 불가 → PIL 사용)
# =========================================================================
_FONT_CACHE = {}
_FONT_CANDIDATES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/Library/Fonts/AppleGothic.ttf",
    "/System/Library/Fonts/Supplemental/NanumGothic.ttf",
]
try:
    from PIL import ImageFont, ImageDraw, Image
    _PIL_OK = True
except Exception:
    _PIL_OK = False


def _load_font(size):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                continue
    _FONT_CACHE[size] = font
    return font


def put_text(img, text, org, size=24, color=(255, 255, 255)):
    """한글 포함 텍스트를 BGR 이미지에 그린다. PIL/폰트 없으면 로마자 폴백."""
    x, y = org
    if _PIL_OK:
        font = _load_font(size)
        if font is not None:
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            # PIL 은 좌상단 기준, cv2 는 baseline 기준 → y 를 위로 올려 맞춤
            draw.text((x, y - size), text, font=font, fill=(color[2], color[1], color[0]))
            img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            return
    # 폴백: 라벨을 로마자로
    ascii_text = "".join(ROMAJA.get(ch, ch) for ch in [text]) if text in ROMAJA else text
    cv2.putText(img, ascii_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                size / 30.0, color, 2, cv2.LINE_AA)


# =========================================================================
# MediaPipe Tasks 초기화 (앱과 동일)
# =========================================================================
def make_hand():
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(HAND_TASK)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=DET_CONF,
        min_hand_presence_confidence=PRESENCE_CONF,
        min_tracking_confidence=TRK_CONF,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def make_pose():
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(POSE_TASK)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=DET_CONF,
        min_pose_presence_confidence=PRESENCE_CONF,
        min_tracking_confidence=TRK_CONF,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


# =========================================================================
# 특징 추출 — SignFeatureExtractor.kt / build_feature_from_frame 와 1:1 동일
# =========================================================================
def normalize_hand(pts):
    """손목(0) 기준 평행이동 후 ||lm[9]-wrist|| 로 정규화 → (21,3)"""
    lm = pts.copy()
    wrist = lm[0].copy()
    lm -= wrist
    scale = float(np.linalg.norm(lm[9])) + 1e-6
    lm /= scale
    return lm


def build_record(hand_res, pose_res):
    """한 프레임 → 저장용 record dict. 양손 모두 없으면 None (클립에 넣지 않음)."""
    # ----- Pose (어깨) -----
    has_body = False
    ms = np.zeros(3, np.float32)
    sw = 0.0
    lsh = np.zeros(3, np.float32)
    rsh = np.zeros(3, np.float32)
    if pose_res.pose_landmarks:
        pl = pose_res.pose_landmarks[0]
        if len(pl) > 12:
            l, r = pl[11], pl[12]
            lsh = np.array([l.x, l.y, l.z], np.float32)
            rsh = np.array([r.x, r.y, r.z], np.float32)
            ms = (lsh + rsh) / 2.0
            sw = float(np.linalg.norm(lsh - rsh) + 1e-6)
            has_body = True

    # ----- Hands -----
    right_norm = left_norm = None
    right_root = left_root = None
    hands_lm = hand_res.hand_landmarks or []
    handed = hand_res.handedness or []

    # 손을 (norm, root, handedness) 로 모아둔 뒤 배정한다.
    # handedness 로 바로 슬롯에 넣으면 두 손이 같은 라벨로 나올 때 하나가
    # 덮어써져 반대쪽 63차원이 통째로 0이 된다(두 손이 가까워지는 동작에서 빈발).
    cands = []
    for i, lm in enumerate(hands_lm):
        if len(lm) < 21:
            continue
        pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        xmin, ymin = pts[:, 0].min(), pts[:, 1].min()
        xmax, ymax = pts[:, 0].max(), pts[:, 1].max()
        if (xmax - xmin) < HAND_MIN_BOX and (ymax - ymin) < HAND_MIN_BOX:
            continue
        root = pts[0].copy()
        norm = normalize_hand(pts)
        label = "Right"
        if i < len(handed) and handed[i]:
            label = handed[i][0].category_name
        cands.append((norm, root, label))

    if len(cands) >= 2:
        # 양손이 잡힌 프레임: handedness 를 믿지 않고 화면 x 좌표로 배정한다.
        # 정렬 방향은 기존 dataset/체크포인트의 관례에 맞춘다.
        # (기존 데이터는 양손 프레임의 97~100% 에서 right 슬롯의 x 가 더 작다.)
        cands.sort(key=lambda c: c[1][0])
        right_norm, right_root = cands[0][0], cands[0][1]
        left_norm, left_root = cands[1][0], cands[1][1]
    elif len(cands) == 1:
        # 한 손만 잡힌 프레임은 x 로 판단할 수 없으므로 handedness 를 쓴다.
        norm, root, label = cands[0]
        if label == "Right":
            right_norm, right_root = norm, root
        else:
            left_norm, left_root = norm, root

    if right_norm is None and left_norm is None:
        return None, (lsh, rsh, ms, has_body, hands_lm)

    def rel_ms(root):
        if root is None or not has_body or sw <= 0.0:
            return [0.0, 0.0, 0.0]
        return ((root - ms) / sw).tolist()

    rec = {
        "timestamp": time.time(),
        "hand_mask": [1 if right_norm is not None else 0,
                      1 if left_norm is not None else 0],
        "right": right_norm.tolist() if right_norm is not None else [],
        "left": left_norm.tolist() if left_norm is not None else [],
        "body_mask": int(has_body),
        "ms": ms.tolist(),
        "sw": sw,
        "right_root": right_root.tolist() if right_root is not None else [],
        "left_root": left_root.tolist() if left_root is not None else [],
        "right_rel_ms": rel_ms(right_root),
        "left_rel_ms": rel_ms(left_root),
    }
    return rec, (lsh, rsh, ms, has_body, hands_lm)


# =========================================================================
# 저장 — 2learn.py 가 읽는 pack .npy 형식 (1main.py 와 동일 키)
# =========================================================================
def next_clip_index(label_dir: Path) -> int:
    pat = re.compile(r"^clip_(\d{4})$")
    mx = 0
    if label_dir.exists():
        for child in label_dir.iterdir():
            m = pat.match(child.stem)
            if m:
                mx = max(mx, int(m.group(1)))
    return mx + 1


def save_clip(label_dir: Path, idx: int, records, label: str) -> bool:
    if len(records) < MIN_SAVE_FRAMES:
        print(f"[WARN] 프레임 부족({len(records)}<{MIN_SAVE_FRAMES}) → 저장 안 함")
        return False

    N = len(records)
    ts = np.zeros((N,), np.float64)
    mask = np.zeros((N, 2), np.int8)
    body = np.zeros((N,), np.int8)
    right = np.zeros((N, 21, 3), np.float32)
    left = np.zeros((N, 21, 3), np.float32)
    ms = np.zeros((N, 3), np.float32)
    sw = np.zeros((N,), np.float32)
    rr = np.zeros((N, 3), np.float32)
    lr = np.zeros((N, 3), np.float32)
    rrel = np.zeros((N, 3), np.float32)
    lrel = np.zeros((N, 3), np.float32)

    for i, rec in enumerate(records):
        ts[i] = rec["timestamp"]
        m = rec["hand_mask"]
        mask[i] = m
        if m[0] and rec["right"]:
            right[i] = np.asarray(rec["right"], np.float32)
        if m[1] and rec["left"]:
            left[i] = np.asarray(rec["left"], np.float32)
        body[i] = int(rec["body_mask"])
        ms[i] = np.asarray(rec["ms"], np.float32)
        sw[i] = float(rec["sw"])
        rr[i] = np.asarray(rec["right_root"] or [0, 0, 0], np.float32)
        lr[i] = np.asarray(rec["left_root"] or [0, 0, 0], np.float32)
        rrel[i] = np.asarray(rec["right_rel_ms"], np.float32)
        lrel[i] = np.asarray(rec["left_rel_ms"], np.float32)

    meta = {
        "label": label,
        "created_at": datetime.now().isoformat(),
        "frames": int(N),
        "version": "tasks-hands+pose.v1",
        "pipeline": "mediapipe.tasks (앱과 동일)",
        "mediapipe": mp.__version__,
        "note": "hand local(21x3) wrist-normalized + mid-shoulder anchor + rel_ms",
    }

    label_dir.mkdir(parents=True, exist_ok=True)
    out = label_dir / f"clip_{idx:04d}.npy"
    np.save(out, {
        "meta": meta, "timestamps": ts,
        "hand_mask": mask, "body_mask": body,
        "right": right, "left": left,
        "ms": ms, "sw": sw,
        "right_root": rr, "left_root": lr,
        "right_rel_ms": rrel, "left_rel_ms": lrel,
    }, allow_pickle=True)
    print(f"[SAVE] {label} -> {out.name} (frames={N})")
    return True


def count_clips(label_dir: Path) -> int:
    if not label_dir.exists():
        return 0
    return len(list(label_dir.glob("clip_*.npy")))


# =========================================================================
# 오버레이
# =========================================================================
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def draw_overlay(vis, extra):
    lsh, rsh, ms, has_body, hands_lm = extra
    h, w = vis.shape[:2]
    for lm in hands_lm:
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        for a, b in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(vis, pts[a], pts[b], (240, 108, 198), 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(vis, p, 3, (226, 58, 168), -1, cv2.LINE_AA)
    if has_body:
        lx, ly = int(lsh[0] * w), int(lsh[1] * h)
        rx, ry = int(rsh[0] * w), int(rsh[1] * h)
        mx, my = int(ms[0] * w), int(ms[1] * h)
        cv2.line(vis, (lx, ly), (rx, ry), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(vis, (mx, my), 6, (0, 200, 0), -1, cv2.LINE_AA)


# =========================================================================
# 메인
# =========================================================================
def main():
    ap = argparse.ArgumentParser(description="SLingo 수어 데이터셋 수집기")
    ap.add_argument("--out", default=str(HERE / "dataset"),
                    help="데이터셋 출력 폴더 (기본: ./dataset)")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="수집할 라벨 목록 (기본: dataset 폴더에 있는 클래스 전체)")
    ap.add_argument("--auto", action="store_true",
                    help="자동 캡처: 손이 나타나면 녹화 시작, 사라지면 자동 저장")
    ap.add_argument("--target", type=int, default=20,
                    help="라벨당 목표 클립 수 (HUD 표시용)")
    ap.add_argument("--camera", type=int, default=0, help="웹캠 인덱스")
    args = ap.parse_args()

    if not HAND_TASK.exists() or not POSE_TASK.exists():
        print(f"[ERR] .task 모델이 없습니다: {HAND_TASK} / {POSE_TASK}")
        print("      SLingo app/src/main/assets 의 .task 를 models/ 로 복사하세요.")
        return

    out_root = Path(args.out).resolve()
    labels = list(args.labels) if args.labels else default_labels(out_root)
    if not labels:
        print("[ERR] 수집할 라벨이 없습니다. --labels 로 지정하세요.")
        return
    print(f"[LABELS] {len(labels)}개: {', '.join(labels)}")
    li = 0  # 현재 라벨 인덱스

    hand = make_hand()
    pose = make_pose()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[ERR] 웹캠을 열 수 없습니다.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)

    print("=" * 60)
    print(f"출력 폴더 : {out_root}")
    print(f"라벨 {len(labels)}개, 라벨당 목표 {args.target} 클립")
    print(f"모드      : {'자동 캡처' if args.auto else '수동'}")
    print("-" * 60)
    print("키:  r=녹화토글  s=클립저장  d=버퍼폐기  u=마지막클립삭제")
    print("     n/p=다음/이전 라벨  a=자동모드 토글  l=현황출력  q=종료")
    print("=" * 60)

    win = "SLingo Dataset Maker"
    buffer, temp = [], []
    recording = False
    started = False
    start_run = stop_run = 0
    temp_start = 0.0
    ignore_until = 0.0
    next_t = time.time()
    dt = 1.0 / FPS_TARGET
    auto = args.auto

    def reset_gate():
        nonlocal started, start_run, stop_run, temp_start
        temp.clear()
        started = False
        start_run = stop_run = 0
        temp_start = 0.0

    def cur_label():
        return labels[li]

    def cur_dir():
        return out_root / cur_label()

    clip_idx = next_clip_index(cur_dir())

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERR] 프레임 읽기 실패")
            break
        frame = cv2.flip(frame, 1)      # 앱과 동일한 거울(좌우반전) 좌표계
        now = time.time()
        vis = frame.copy()
        extra = (np.zeros(3, np.float32), np.zeros(3, np.float32),
                 np.zeros(3, np.float32), False, [])

        if recording and now >= next_t:
            next_t = now + dt
            if now >= ignore_until:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                hres = hand.detect(mp_img)
                pres = pose.detect(mp_img)
                rec, extra = build_record(hres, pres)

                has_hand = rec is not None
                if not started:
                    if has_hand:
                        if temp_start == 0.0:
                            temp_start = now
                        temp.append(rec)
                        start_run += 1
                        if start_run >= START_CONSEC:
                            started = True
                            buffer.extend(temp)
                            temp.clear()
                    else:
                        start_run = 0
                        if temp and (now - temp_start) > MAX_TEMP_SEC:
                            temp.clear()
                            temp_start = 0.0
                else:
                    if has_hand:
                        buffer.append(rec)
                        stop_run = 0
                    else:
                        stop_run += 1
                        if stop_run >= STOP_CONSEC:
                            # 손 사라짐 → 클립 경계
                            cut = min(TRIM_TAIL, len(buffer))
                            clip = buffer[:-cut] if cut > 0 else buffer[:]
                            if auto and len(clip) >= MIN_SAVE_FRAMES:
                                if save_clip(cur_dir(), clip_idx, clip, cur_label()):
                                    clip_idx += 1
                                buffer = []
                                reset_gate()
                            else:
                                # 수동: 버퍼 유지, 게이트만 리셋해 다음 손짓 대기
                                stop_run = 0

        # ---------- HUD ----------
        draw_overlay(vis, extra)
        cnt = count_clips(cur_dir())
        gate = "STARTED" if started else "WAIT"
        put_text(vis, f"[{li + 1}/{len(labels)}] {cur_label()}", (12, 34), 30,
                 (0, 255, 255))
        put_text(vis, f"clips {cnt}/{args.target}   buffer {len(buffer)}",
                 (12, 70), 24, (255, 255, 255))
        put_text(vis,
                 f"REC {'ON' if recording else 'OFF'}  "
                 f"{'AUTO' if auto else 'MANUAL'}  gate {gate}",
                 (12, 102), 24,
                 (80, 220, 80) if recording else (180, 180, 180))
        if now < ignore_until:
            put_text(vis, f"준비... {ignore_until - now:.1f}s", (12, 140), 26,
                     (0, 220, 255))
        put_text(vis, "r rec  s save  d drop  u undo  n/p label  a auto  q quit",
                 (12, vis.shape[0] - 16), 20, (170, 170, 170))

        cv2.imshow(win, vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            recording = not recording
            next_t = time.time()
            ignore_until = time.time() + PREPARE_SEC
            reset_gate()
            print(f"[REC] {'시작' if recording else '정지'}  ({cur_label()})")
        elif key == ord("a"):
            auto = not auto
            print(f"[MODE] {'자동' if auto else '수동'}")
        elif key == ord("s"):
            cut = min(TRIM_TAIL, len(buffer))
            clip = buffer[:-cut] if cut > 0 else buffer[:]
            if save_clip(cur_dir(), clip_idx, clip, cur_label()):
                clip_idx += 1
            buffer = []
            reset_gate()
            ignore_until = time.time() + PREPARE_SEC
        elif key == ord("d"):
            buffer = []
            reset_gate()
            ignore_until = time.time() + PREPARE_SEC
            print("[DROP] 버퍼 폐기")
        elif key == ord("u"):
            last = clip_idx - 1
            f = cur_dir() / f"clip_{last:04d}.npy"
            if last >= 1 and f.exists():
                f.unlink()
                clip_idx = last
                print(f"[UNDO] {f.name} 삭제")
            else:
                print("[UNDO] 삭제할 클립 없음")
        elif key in (ord("n"), ord("p")):
            li = (li + (1 if key == ord("n") else -1)) % len(labels)
            clip_idx = next_clip_index(cur_dir())
            buffer = []
            reset_gate()
            recording = False
            print(f"[LABEL] → {cur_label()} (다음 idx {clip_idx:04d})")
        elif key == ord("l"):
            print("-" * 40)
            for lb in labels:
                print(f"  {lb:<10} {count_clips(out_root / lb):>3} clips")
            print("-" * 40)

    cap.release()
    cv2.destroyAllWindows()
    hand.close()
    pose.close()

    print("\n[요약] 라벨별 수집 현황")
    total = 0
    for lb in labels:
        c = count_clips(out_root / lb)
        total += c
        print(f"  {lb:<10} {c:>3}")
    print(f"  합계 {total} clips")
    print(f"\n다음 단계: 2learn.py 의 DATA_ROOT 를 다음 폴더로 지정 후 학습")
    print(f"          DATA_ROOT = {out_root}")


if __name__ == "__main__":
    main()
