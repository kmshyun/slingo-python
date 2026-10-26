import json
import time
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from collections import deque

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import ollama
import os

from PIL import Image, ImageDraw, ImageFont
OLLAMA_MODEL = "exaone3.5:7.8b"


# ============================================================
# 한글 지원을 위한 텍스트 그리기 함수
# ============================================================
def find_apple_gothic():
    # macOS에서 Apple SD Gothic Neo가 위치하는 표준 경로 목록
    candidates = [
        "/System/Library/Fonts/Supplemental/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/AppFonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Core/AppleSDGothicNeo.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
            
    # 전체 탐색(os.walk)을 제거하고, 없으면 안전하게 None을 반환하여 기본 폰트 사용
    return None

# 프로그램 실행 시 폰트 경로를 딱 한 번만 미리 로드하여 속도를 높입니다.
FONT_PATH = find_apple_gothic()
if FONT_PATH:
    print(f"[Font Success] 한글 폰트를 찾았습니다: {FONT_PATH}")
else:
    print("[Font Warning] Apple SD Gothic Neo를 찾지 못했습니다. 기본 폰트를 사용합니다.")


def draw_korean_text(img, text, org, font_size, color=(255, 255, 255)):
    try:
        if FONT_PATH:
            font = ImageFont.truetype(FONT_PATH, font_size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    draw = ImageDraw.Draw(img_pil)
    draw.text(org, text, font=font, fill=color)
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

BASE_DIR = Path(__file__).resolve().parent

# code/dataset 으로 학습한(2learn.py) 체크포인트를 사용한다.
CKPT_DIR = BASE_DIR / "checkpoints"
DUMP_DIR = BASE_DIR / "debug_runtime"   # x 키로 저장하는 런타임 특징 덤프

# 앱/수집기와 동일한 MediaPipe Tasks 모델 (code/models/ 에 둔다)
HAND_TASK = BASE_DIR / "models" / "hand_landmarker.task"
POSE_TASK = BASE_DIR / "models" / "pose_landmarker.task"


# ============================================================
# 수집(collect.py) 조건과 반드시 일치시켜야 하는 값들
# 하나라도 어긋나면 train/inference mismatch 로 인식률이 급락한다.
# ============================================================
DET_CONF = 0.5
TRK_CONF = 0.5
PRESENCE_CONF = 0.5
NUM_HANDS = 2
HAND_MIN_BOX = 0.03      # 너무 작은 손 검출 제거
CAP_W, CAP_H = 640, 480  # 수집 해상도(4:3)와 동일
FPS_TARGET = 15.0        # 수집 프레임레이트와 동일

# 화면 출력 크기 (분석은 CAP_W x CAP_H 로 하고, 표시만 확대)
DISP_W, DISP_H = 1280, 720
WIN_NAME = "Sign"


# ============================================================
# UI CONFIG
# ============================================================
WINDOW_MARGIN = 18
SHOW_FACE_BOX = True
SHOW_FACE_MESH = False
SHOW_EMOTION_LABEL = False      # 실제 표정 인식이 아니므로 기본 OFF       # SHOW_EMOTION_LABEL=True일 때만 표시
SHOW_SHORTCUT_PANEL = True


# ============================================================
# Ollama 문장 생성 유틸
# ============================================================
def _simple_tokens(s: str):
    s = re.sub(r"[^a-zA-Z0-9']+", " ", s.lower())
    return [t for t in s.split() if t]


# ============================================================
# 2. 문장 완성 함수 (Ollama 로컬 AI 버전)
# ============================================================
def make_sentence(words: str) -> str:
    """
    수어 단어들을 입력받아, M4 맥북의 로컬 GPU(Ollama)를 이용하여
    조사와 어미가 매끄럽게 결합된 하나의 자연스러운 한국어 문장으로 정제합니다.
    """
    base = " ".join(words.split()).strip()
    if not base:
        return ""

    try:
        # 외부 인터넷 서버를 거치지 않고 맥북 내부에서 즉시 추론합니다.

        response = ollama.chat(
            model=OLLAMA_MODEL,
messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 수어(수화) 단어들을 입력받아, 조사와 어미를 매끄럽게 붙여 하나의 자연스러운 한국어 구어체 문장으로 완성하는 전문가입니다.\n\n"
                        "[반드시 지켜야 할 출력 규칙]\n"
                        "1. 딱딱한 문어체 대신, 실제 일상 대화에서 쓰는 자연스러운 한국어 구어체로 만드세요.\n"
                        "2. 영어, 따옴표, 부연 설명, 이모지 등을 절대 포함하지 말고 오직 완성된 '한국어 문장 딱 한 줄'만 출력하세요.\n"
                        "3. 별다른 조건이 없으면 제일 처음 오는 단어를 문장의 주어로 사용하세요. (예: '엄마 나 사랑' -> '엄마가 나를 사랑해.' 이지 '내가 엄마를 사랑해.'가 아님!)\n"
                        "4. 문맥상 목적어가 생략되어 있다면 '나(를)'를 목적어로 간주하여 문장을 완성하세요. (예: '엄마 주다 밥' -> '엄마가 나한테 밥을 줘.')\n"
                        "5. '나'라는 단어가 가장 먼저 온다면, 문장의 주어는 '나(내가)'로 시작하세요.\n\n"
                        "[올바른 변환 예시 (Few-Shot)]\n"
                        "- 입력: '학교 가다 좋다'\n"
                        "  출력: 나는 학교 가는 게 좋아.\n"
                        "- 입력: '나 배고프다 피자 먹다'\n"
                        "  출력: 나 배고파서 피자 먹고 있어.\n"
                        "- 입력: '엄마 피곤하다 왜'\n"
                        "  출력: 엄마는 왜 이렇게 피곤해?"
                    )
                },
                {
                    "role": "user",
                    "content": f"제시된 단어들을 자연스러운 한국어 구어체 문장 한 줄로 만들어주세요.\n단어: {base}"
                }
            ],
            options={
                "temperature": 0.5,  # 문장 표현의 자연스러움과 다양성을 위해 온도를 살짝 올림
                "top_p": 0.9
            }
        )
        
        # Ollama의 응답 구조에서 텍스트 추출
        natural_sentence = response["message"]["content"].strip()
        
        # 터미널 디버깅용 로그 출력
        print(f"[Ollama Success] Raw: {base} -> Result: {natural_sentence}")
        return natural_sentence

    except Exception as e:
        print("[Ollama API Error]", e)
        # 만약 로컬 AI에 오류가 나더라도 프로그램이 튕기지 않고 단어 버퍼라도 보여주도록 예외 처리
        return base


# ============================================================
# Model
# ============================================================
class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden: int = 192,
        layers: int = 2,
        dropout: float = 0.2,
        bidir: bool = True,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=bidir,
            batch_first=True,
        )
        out_dim = hidden * (2 if bidir else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hn, _) = self.lstm(packed)

        layers = self.lstm.num_layers
        directions = 2 if self.lstm.bidirectional else 1
        hn = hn.view(layers, directions, x.size(0), self.lstm.hidden_size)
        last = hn[-1].transpose(0, 1).contiguous().view(x.size(0), -1)
        return self.head(last)


def load_model():
    ckpt_path = CKPT_DIR / "best_lstm.pt"
    jpath = CKPT_DIR / "label2id.json"

    if not ckpt_path.exists() or not jpath.exists():
        raise FileNotFoundError("checkpoints/best_lstm.pt 또는 label2id.json 없음")

    with open(jpath, "r", encoding="utf-8") as f:
        label2id = json.load(f)

    # 라벨은 dataset 폴더명에서 왔는데, macOS 파일명은 한글을 분해형(NFD)으로
    # 저장한다. 그대로 두면 PIL 이 자모를 합치지 못해 '듣다' 가 '듣다' 로
    # 그려지고, Ollama 에도 분해된 문자열이 넘어간다. 완성형(NFC)으로 통일한다.
    id2label = {i: unicodedata.normalize("NFC", lb) for lb, i in label2id.items()}

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get(
        "config",
        {
            "input_dim": 139,
            "hidden": 192,
            "layers": 2,
            "dropout": 0.2,
            "bidir": True,
        },
    )

    model = BiLSTMClassifier(**cfg, num_classes=len(label2id))
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, id2label


# ============================================================
# MediaPipe / Feature
#   손/포즈 모두 Tasks API (collect.py 와 동일한 .task 모델).
#   mediapipe 0.10.35 에는 레거시 mp.solutions 가 아예 없으므로
#   얼굴 박스는 포즈 랜드마크(0~10: 코/눈/귀/입)로 대신 구한다.
# ============================================================
# 포즈 랜드마크 중 얼굴에 해당하는 인덱스
FACE_POSE_IDX = list(range(0, 11))

# collect.py 와 동일한 손 골격 연결선
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def make_hand_landmarker():
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(HAND_TASK)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=DET_CONF,
        min_hand_presence_confidence=PRESENCE_CONF,
        min_tracking_confidence=TRK_CONF,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def make_pose_landmarker():
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(POSE_TASK)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=DET_CONF,
        min_pose_presence_confidence=PRESENCE_CONF,
        min_tracking_confidence=TRK_CONF,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


def normalize_hand(lm_xyz):
    lm = lm_xyz.copy()
    wrist = lm[0].copy()
    lm -= wrist
    scale = float(np.linalg.norm(lm[9])) + 1e-6
    lm /= scale
    return lm


def face_box_from_pose(pl):
    """포즈 랜드마크의 얼굴 부위(코/눈/귀/입)로 얼굴 박스를 근사한다."""
    pts = [(pl[i].x, pl[i].y) for i in FACE_POSE_IDX if i < len(pl)]
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)

    # 귀~귀 폭 기준으로 이마/턱까지 덮도록 여유를 준다
    mx = (x1 - x0) * 0.18
    my = (y1 - y0) * 0.9
    x0, x1 = max(0.0, x0 - mx), min(1.0, x1 + mx)
    y0, y1 = max(0.0, y0 - my), min(1.0, y1 + my * 0.8)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def analyze_frame(frame_bgr, hand, pose):
    """한 프레임 → (139차원 특징, 얼굴박스, 얼굴점, 손점)
    좌표는 모두 0~1 정규화 값으로 돌려주고, 화면 크기 변환은 그릴 때 한다."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    hand_res = hand.detect(mp_img)
    pose_res = pose.detect(mp_img)

    hands_lm = hand_res.hand_landmarks or []
    handed = hand_res.handedness or []

    hand_points_for_draw = [
        [(p.x, p.y) for p in lm] for lm in hands_lm
    ]

    # ----- Pose feature (어깨) -----
    has_body = False
    ms = np.zeros(3, np.float32)
    sw = 0.0
    face_box = None
    face_points = []   # Tasks 전용 빌드에는 face mesh 가 없다 (SHOW_FACE_MESH 미지원)

    if pose_res.pose_landmarks:
        pl = pose_res.pose_landmarks[0]
        face_box = face_box_from_pose(pl)
        if len(pl) > 12:
            l, r = pl[11], pl[12]
            lsh = np.array([l.x, l.y, l.z], np.float32)
            rsh = np.array([r.x, r.y, r.z], np.float32)
            ms = (lsh + rsh) / 2.0
            sw = float(np.linalg.norm(lsh - rsh) + 1e-6)
            has_body = True

    # ----- Hand feature -----
    right_norm, left_norm = None, None
    right_root, left_root = None, None

    # collect.py 의 손 배정과 반드시 동일해야 한다.
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

        wrist_root = pts[0].copy()
        pts_norm = normalize_hand(pts)

        label = "Right"
        if i < len(handed) and handed[i]:
            label = handed[i][0].category_name

        cands.append((pts_norm, wrist_root, label))

    if len(cands) >= 2:
        # 양손이 잡힌 프레임: handedness 를 믿지 않고 화면 x 좌표로 배정한다.
        # 정렬 방향은 기존 dataset/체크포인트의 관례에 맞춘다.
        # (기존 데이터는 양손 프레임의 97~100% 에서 right 슬롯의 x 가 더 작다.)
        cands.sort(key=lambda c: c[1][0])
        right_norm, right_root = cands[0][0], cands[0][1]
        left_norm, left_root = cands[1][0], cands[1][1]
    elif len(cands) == 1:
        # 한 손만 잡힌 프레임은 x 로 판단할 수 없으므로 handedness 를 쓴다.
        pts_norm, wrist_root, label = cands[0]
        if label == "Right":
            right_norm, right_root = pts_norm, wrist_root
        else:
            left_norm, left_root = pts_norm, wrist_root

    feat = None
    if right_norm is not None or left_norm is not None:
        def rel_ms(root):
            if (root is None) or (not has_body) or (sw <= 0.0):
                return np.array([0.0, 0.0, 0.0], np.float32)
            return ((root - ms) / sw).astype(np.float32)

        hand_mask = np.array(
            [
                1 if right_norm is not None else 0,
                1 if left_norm is not None else 0,
            ],
            np.float32,
        )

        right = right_norm if right_norm is not None else np.zeros((21, 3), np.float32)
        left = left_norm if left_norm is not None else np.zeros((21, 3), np.float32)

        feat = np.concatenate(
            [
                right.reshape(-1),
                left.reshape(-1),
                hand_mask,
                np.array([1.0 if has_body else 0.0], np.float32),
                ms.astype(np.float32),
                np.array([sw], np.float32),
                rel_ms(right_root),
                rel_ms(left_root),
            ],
            axis=0,
        ).astype(np.float32)[None, :]

        np.nan_to_num(feat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return feat, face_box, face_points, hand_points_for_draw


def predict_from_ring(ring, model, id2label, tail_n=30):
    if len(ring) == 0:
        return None, 0.0, []

    x = np.concatenate(list(ring), axis=0)
    if x.shape[0] > tail_n:
        x = x[-tail_n:]

    x_t = torch.from_numpy(x).unsqueeze(0)
    lengths = torch.tensor([x.shape[0]], dtype=torch.long)

    with torch.no_grad():
        logits = model(x_t, lengths)
        prob = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    order = np.argsort(prob)[::-1]
    candidates = [(id2label[int(i)], float(prob[int(i)])) for i in order[:6]]

    best_i = int(prob.argmax())
    return id2label[best_i], float(prob[best_i]), candidates


# ============================================================
# UI Drawing
# ============================================================
PALETTE = {
    "green": (100, 255, 120),
    "blue": (255, 180, 70),
    "white": (235, 235, 235),
    "red": (100, 95, 255),
    "yellow": (70, 210, 255),
    "orange": (80, 160, 255),
    "pink": (235, 170, 255),
}

PRED_COLORS = [
    PALETTE["green"],
    PALETTE["blue"],
    PALETTE["white"],
    PALETTE["red"],
    PALETTE["yellow"],
    PALETTE["orange"],
]


def rounded_rect(img, x, y, w, h, color, r=16, thickness=-1):
    if w <= 0 or h <= 0:
        return
    r = max(1, min(int(r), w // 2, h // 2))
    x, y, w, h = int(x), int(y), int(w), int(h)

    if thickness < 0:
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
        cv2.circle(img, (x + r, y + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x + w - r, y + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x + r, y + h - r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x + w - r, y + h - r), r, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x + r, y), (x + w - r, y), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x + r, y + h), (x + w - r, y + h), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y + r), (x, y + h - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x + w, y + r), (x + w, y + h - r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x + r, y + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x + w - r, y + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x + r, y + h - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x + w - r, y + h - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)


def glass_panel(img, x, y, w, h, alpha=0.14, radius=18, border_alpha=0.24):
    overlay = img.copy()
    rounded_rect(overlay, x, y, w, h, (255, 255, 255), r=radius, thickness=-1)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

    border = img.copy()
    rounded_rect(border, x, y, w, h, (255, 255, 255), r=radius, thickness=1)
    cv2.addWeighted(border, border_alpha, img, 1.0 - border_alpha, 0, img)


# ============================================================
# 기존 shadow_text 함수 수정 (안전성 강화)
# ============================================================
def shadow_text(img, text, org, scale=1.0, color=(255, 255, 255), thickness=2):
    # cv2.putText의 scale(배율)을 Pillow의 픽셀 단위를 위해 정수로 변환 (기본 26px 기준)
    font_size = int(scale * 26)
    
    # y축 좌표 보정: cv2.putText는 텍스트의 좌측 '하단' 기준이지만, Pillow는 좌측 '상단' 기준입니다.
    # 이 차이 때문에 글씨가 화면 위로 밀려서 안 보였을 수 있으므로 아래로 조절합니다.
    adjusted_org = (org[0], org[1] - font_size)
    
    # 1. 그림자 효과 적용
    shadow_org = (adjusted_org[0] + 2, adjusted_org[1] + 2)
    img_with_shadow = draw_korean_text(img, text, shadow_org, font_size, color=(20, 20, 25))
    
    # 2. 본문 텍스트 적용 후 원본 이미지 데이터 갱신
    final_img = draw_korean_text(img_with_shadow, text, adjusted_org, font_size, color=color)
    
    # 원본 이미지 객체의 메모리 주소를 유지하며 데이터를 덮어씁니다.
    np.copyto(img, final_img)


# 그라디언트/비네트는 프레임 크기에만 의존하므로 매 프레임 다시 만들지 않고 캐시한다.
_TINT_CACHE = {}


def _tint_layers(h, w):
    key = (h, w)
    if key not in _TINT_CACHE:
        # warm pink top glow
        a = np.clip(1.0 - np.arange(h) / max(h * 0.62, 1), 0.0, None)
        grad = np.zeros((h, w, 3), np.uint8)
        grad[:, :, 0] = (30 * a).astype(np.uint8)[:, None]
        grad[:, :, 1] = (32 * a).astype(np.uint8)[:, None]
        grad[:, :, 2] = (85 * a).astype(np.uint8)[:, None]

        # vignette
        yy, xx = np.indices((h, w))
        cx, cy = w / 2, h / 2
        dist = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
        vignette = np.clip((dist - 0.35) / 0.8, 0, 1)[..., None]

        _TINT_CACHE[key] = (grad, vignette)
    return _TINT_CACHE[key]


def apply_cinematic_tint(img):
    h, w = img.shape[:2]
    grad, vignette = _tint_layers(h, w)

    # dark base overlay
    base = np.zeros_like(img)
    base[:] = (28, 20, 35)
    cv2.addWeighted(base, 0.18, img, 0.82, 0, img)

    cv2.addWeighted(grad, 0.28, img, 0.72, 0, img)

    dark = np.zeros_like(img)
    dark[:] = (0, 0, 0)
    overlay = img.copy()
    cv2.addWeighted(dark, 0.35, overlay, 0.65, 0, overlay)
    img[:] = (img * (1 - vignette * 0.45) + overlay * (vignette * 0.45)).astype(np.uint8)


def draw_window_shell(img):
    """Draw only a subtle AR-style boundary. No fake macOS title bar inside the video."""
    h, w = img.shape[:2]
    x, y = WINDOW_MARGIN, WINDOW_MARGIN
    ww, hh = w - WINDOW_MARGIN * 2, h - WINDOW_MARGIN * 2

    # The real macOS window already has a title bar, so keep only a thin AR border.
    rounded_rect(img, x, y, ww, hh, (235, 235, 235), r=18, thickness=1)

def draw_header(img, rec, buffer_len):
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    top_y = WINDOW_MARGIN + 18

    # Top-right status cluster: time / REC-OFF / buffer / clear
    time_str = time.strftime("%a  %H:%M")
    shadow_text(img, time_str, (w - WINDOW_MARGIN - 390, top_y + 25), scale=0.70, thickness=2)

    rec_x = w - WINDOW_MARGIN - 235
    rec_y = top_y + 6
    glass_panel(img, rec_x, rec_y, 72, 30, alpha=0.16, radius=15)
    cv2.circle(img, (rec_x + 15, rec_y + 15), 5, (80, 80, 255) if rec else (150, 150, 150), -1, cv2.LINE_AA)
    cv2.putText(img, "REC" if rec else "OFF", (rec_x + 27, rec_y + 20), font, 0.52, (255, 120, 120) if rec else (225, 225, 225), 2, cv2.LINE_AA)

    buf_x = rec_x + 82
    glass_panel(img, buf_x, rec_y, 96, 30, alpha=0.16, radius=15)
    cv2.putText(img, f"Buffer {buffer_len}", (buf_x + 12, rec_y + 20), font, 0.48, (230, 230, 230), 1, cv2.LINE_AA)


# ============================================================
# 2. 깨지던 UI 패널 함수들 전부 업데이트
# ============================================================

def draw_gloss_chips(img, words):
    h, w = img.shape[:2]
    y = WINDOW_MARGIN + 58
    x = WINDOW_MARGIN + 30

    # 'GLASS' 타이틀 출력 보정 (한글 변환 준비)
    img[:] = draw_korean_text(img, "WORDS", (x, y + 6), 14, (210, 205, 205))
    x += 58

    if not words:
        return

    for idx, word in enumerate(words[-8:]):
        text = str(word).upper()
        # 대략적인 글자 가로폭 계산 (Pillow 대응)
        font_size = 15
        tw = len(text) * int(font_size * 0.7)
        chip_w = tw + 36
        if x + chip_w > w - 430:
            break
        glass_panel(img, x, y, chip_w, 34, alpha=0.15, radius=17)
        color = PRED_COLORS[idx % len(PRED_COLORS)]
        cv2.circle(img, (x + 15, y + 17), 5, color, -1, cv2.LINE_AA)
        
        # 원래 cv2.putText 하던 곳을 교체
        img[:] = draw_korean_text(img, text, (x + 27, y + 8), font_size, (255, 255, 255))
        x += chip_w + 10


def draw_record_badge(img, rec):
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    x = WINDOW_MARGIN + 26
    y = WINDOW_MARGIN + 104
    glass_panel(img, x, y, 118, 58, alpha=0.15, radius=14)
    cv2.circle(img, (x + 16, y + 20), 5, PALETTE["green"] if rec else (170, 170, 170), -1, cv2.LINE_AA)
    cv2.putText(img, "REC ON" if rec else "REC OFF", (x + 28, y + 25), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def wrap_text_to_lines(text, font, scale, thickness, max_w):
    words = text.split()
    lines = []
    cur = []
    for word in words:
        trial = " ".join(cur + [word])
        (tw, _), _ = cv2.getTextSize(trial, font, scale, thickness)
        if tw <= max_w or not cur:
            cur.append(word)
        else:
            lines.append(" ".join(cur))
            cur = [word]
    if cur:
        lines.append(" ".join(cur))
    return lines


def draw_top_sentence(img, sentence):
    if not sentence:
        return
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.15 if w < 1400 else 1.28
    thickness = 3
    max_w = int(w * 0.58)
    lines = wrap_text_to_lines(sentence, font, scale, thickness, max_w)

    line_h = 46
    box_w = max_w + 42
    box_h = max(74, 24 + len(lines) * line_h)
    x = (w - box_w) // 2
    y = WINDOW_MARGIN + 104

    glass_panel(img, x, y, box_w, box_h, alpha=0.13, radius=22, border_alpha=0.18)
    for i, line in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(line, font, scale, thickness)
        tx = x + (box_w - tw) // 2
        ty = y + 48 + i * line_h
        shadow_text(img, line, (tx, ty), scale=scale, thickness=thickness)


def draw_face_box(img, face_box):
    if not SHOW_FACE_BOX or face_box is None:
        return
    x, y, bw, bh = face_box
    rounded_rect(img, x, y, bw, bh, (255, 255, 255), r=8, thickness=2)


def draw_face_mesh_dots(img, face_points):
    if not SHOW_FACE_MESH or not face_points:
        return
    overlay = img.copy()
    for p in face_points[::2]:
        cv2.circle(overlay, p, 1, (245, 240, 240), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.33, img, 0.67, 0, img)


def draw_hand_glow(img, hand_pts):
    if not hand_pts or len(hand_pts) < 21:
        return
    overlay = img.copy()
    for a, b in HAND_CONNECTIONS:
        cv2.line(overlay, hand_pts[a], hand_pts[b], PALETTE["pink"], 7, cv2.LINE_AA)
        cv2.line(img, hand_pts[a], hand_pts[b], (255, 240, 255), 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

    for p in hand_pts:
        cv2.circle(img, p, 7, PALETTE["pink"], -1, cv2.LINE_AA)
        cv2.circle(img, p, 3, (255, 255, 255), -1, cv2.LINE_AA)


def draw_candidates_panel(img, candidates):
    h, w = img.shape[:2]
    if not candidates:
        return

    panel_w = 210
    panel_h = 300
    x = w - WINDOW_MARGIN - panel_w - 28
    y = WINDOW_MARGIN + 170

    glass_panel(img, x, y, panel_w, panel_h, alpha=0.12, radius=18, border_alpha=0.18)
    img[:] = draw_korean_text(img, "TOP CANDIDATES", (x + 18, y + 16), 14, (255, 255, 255))

    item_x = x + 14
    item_y = y + 48
    item_w = panel_w - 28
    item_h = 35

    for idx, (label, prob) in enumerate(candidates[:6]):
        yy = item_y + idx * 41
        glass_panel(img, item_x, yy, item_w, item_h, alpha=0.12, radius=16, border_alpha=0.18)
        color = PRED_COLORS[idx % len(PRED_COLORS)]
        cv2.circle(img, (item_x + 16, yy + item_h // 2), 5, color, -1, cv2.LINE_AA)
        
        # 라벨 한글 깨짐 해결
        img[:] = draw_korean_text(img, str(label), (item_x + 30, yy + 8), 14, (255, 255, 255))
        
        pct = f"{int(prob * 100)}%"
        img[:] = draw_korean_text(img, pct, (item_x + item_w - 45, yy + 9), 12, (230, 230, 230))

def draw_shortcuts_panel(img):
    if not SHOW_SHORTCUT_PANEL:
        return
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    panel_w = 210
    panel_h = 200
    x = w - WINDOW_MARGIN - panel_w - 28
    y = h - WINDOW_MARGIN - panel_h - 60

    # 너무 작으면 숨김
    if y < WINDOW_MARGIN + 430:
        return

    glass_panel(img, x, y, panel_w, panel_h, alpha=0.11, radius=18, border_alpha=0.18)
    cv2.putText(img, "SHORTCUTS", (x + 18, y + 30), font, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(img, (x + 18, y + 44), (x + panel_w - 18, y + 44), (210, 210, 210), 1, cv2.LINE_AA)

    items = [
        ("r", "녹화 / 녹화멈춤"),
        ("s", "지금 예측하기"),
        ("d", "가장 마지막 항목 삭제"),
        ("k", "자연어로 변환"),
        ("c", "전부 삭제"),
        ("q", "종료"),
    ]
    for i, (key, desc) in enumerate(items):
        yy = y + 66 + i * 28
        cv2.putText(img, key, (x + 20, yy), font, 0.55, (120, 255, 100), 2, cv2.LINE_AA)
        cv2.putText(img, desc, (x + 50, yy), font, 0.48, (235, 235, 235), 1, cv2.LINE_AA)


def draw_predict_card(img, word, prob):
    h, w = img.shape[:2]
    box_w = 235
    box_h = 112
    x = (w - box_w) // 2
    y = h - WINDOW_MARGIN - 140

    glass_panel(img, x, y, box_w, box_h, alpha=0.14, radius=22, border_alpha=0.21)

    title = "PREDICT"
    img[:] = draw_korean_text(img, title, (x + 85, y + 10), 16, PALETTE["green"])

    main = (word or "READY").upper()
    # 글자 수에 따른 대략적인 중앙 정렬 좌표 계산
    main_font_size = 28
    tw = len(main) * int(main_font_size * 0.65)
    img[:] = draw_korean_text(img, main, (x + (box_w - tw) // 2, y + 38), main_font_size, (255, 255, 255))

    pct = int(max(0.0, min(1.0, prob)) * 100)
    small = f"{pct}% 확신" if word else "수화 대기중"
    img[:] = draw_korean_text(img, small, (x + 25, y + 74), 12, (130, 255, 100))

    bar_x = x + 25
    bar_y = y + 100
    bar_w = box_w - 50
    rounded_rect(img, bar_x, bar_y, bar_w, 7, (105, 105, 105), r=4, thickness=-1)
    fill = int(bar_w * max(0.0, min(1.0, prob)))
    if fill > 0:
        rounded_rect(img, bar_x, bar_y, max(fill, 7), 7, PALETTE["green"], r=4, thickness=-1)

def draw_menu_and_footer(img):
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    mx = WINDOW_MARGIN + 26
    my = h - WINDOW_MARGIN - 78
    glass_panel(img, mx, my, 42, 42, alpha=0.14, radius=21, border_alpha=0.22)
    for i in range(3):
        yy = my + 13 + i * 9
        cv2.line(img, (mx + 13, yy), (mx + 29, yy), (255, 255, 255), 2, cv2.LINE_AA)

    hint_x = WINDOW_MARGIN + 26
    hint_y = h - WINDOW_MARGIN - 18
    hint = "r:record   s:predict   d:delete   k:finish   c:clear   x:dump   q:quit"
    glass_panel(img, hint_x, hint_y - 23, 455, 34, alpha=0.10, radius=14, border_alpha=0.18)
    cv2.putText(img, hint, (hint_x + 12, hint_y), font, 0.46, (235, 235, 235), 1, cv2.LINE_AA)


def letterbox(frame, out_w=DISP_W, out_h=DISP_H):
    """4:3 분석 프레임을 비율 유지한 채 16:9 캔버스를 빈틈없이 채우고,
    넘치는 부분은 중앙 기준으로 잘라낸다(레터박스 없음).
    반환: (캔버스, 배치영역 x, y, w, h) — x/y는 잘려나간 만큼 음수일 수 있다."""
    h, w = frame.shape[:2]
    scale = max(out_w / w, out_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    ox, oy = (nw - out_w) // 2, (nh - out_h) // 2
    canvas = resized[oy:oy + out_h, ox:ox + out_w]
    return canvas, (-ox, -oy, nw, nh)


def to_px(pt_norm, box):
    """정규화 좌표 → 캔버스 픽셀 좌표"""
    ox, oy, nw, nh = box
    return (int(ox + pt_norm[0] * nw), int(oy + pt_norm[1] * nh))


def draw_full_ui(frame, rec, buffer_len, sentence_text, chip_words, preview_word, preview_prob, candidates, face_box, face_points, hands_for_draw):
    vis, box = letterbox(frame)
    apply_cinematic_tint(vis)

    draw_window_shell(vis)
    draw_header(vis, rec, buffer_len)
    draw_gloss_chips(vis, chip_words)
    draw_record_badge(vis, rec)

    if sentence_text:
        draw_top_sentence(vis, sentence_text)

    # 정규화 좌표를 캔버스 픽셀로 변환해서 그린다.
    ox, oy, nw, nh = box
    draw_face_mesh_dots(vis, [to_px(p, box) for p in face_points])
    if face_box is not None:
        fx, fy, fw, fh = face_box
        draw_face_box(vis, (int(ox + fx * nw), int(oy + fy * nh),
                            int(fw * nw), int(fh * nh)))
    else:
        draw_face_box(vis, None)

    for pts in hands_for_draw:
        draw_hand_glow(vis, [to_px(p, box) for p in pts])

    draw_candidates_panel(vis, candidates)
    draw_shortcuts_panel(vis)
    draw_predict_card(vis, preview_word, preview_prob)
    draw_menu_and_footer(vis)

    return vis


# ============================================================
# Main Loop
# ============================================================
def main():
    win_len = 30
    pred_step = 5
    min_prob = 0.6
    hold_sec = 2.0

    stable_hold = 0.6
    stable_reset = 0.4
    tail_n = 30
    flush_n = 10
    repeat_cd = 0.5

    if not HAND_TASK.exists() or not POSE_TASK.exists():
        print(f"[ERR] .task 모델이 없습니다: {HAND_TASK} / {POSE_TASK}")
        return

    model, id2label = load_model()
    print(f"[MODEL] {CKPT_DIR}  라벨 {len(id2label)}개")
    print(f"[MODEL] {sorted(id2label.values())}")

    hands = make_hand_landmarker()
    pose = make_pose_landmarker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    # 수집(collect.py) 과 동일한 분석 해상도. 화면 출력만 확대한다.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)

    rec = False
    ring = deque(maxlen=win_len)
    frame_cnt = 0

    sentence = []
    final_sentence = ""
    last_raw_tokens = []

    last_text = None
    last_prob = 0.0
    show_until = 0.0
    top_candidates = []

    prev_word = None
    last_add_t = 0.0
    last_no_det = 0.0
    stable_word = None
    stable_t = 0.0
    stable_done = False

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, DISP_W, DISP_H)

    # 분석 주기 (수집 때와 동일한 15fps). 사이 프레임은 직전 결과로 그린다.
    analysis_dt = 1.0 / FPS_TARGET
    next_t = 0.0
    feat = None
    face_box = None
    face_points = []
    hands_for_draw = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        now = time.time()

        # 프레임레이트가 학습 데이터와 다르면 같은 동작이라도 시퀀스 길이/속도가
        # 달라져 인식률이 떨어진다. 그래서 분석은 15fps 로 고정한다.
        analyzed = now >= next_t
        if analyzed:
            next_t = now + analysis_dt
            feat, face_box, face_points, hands_for_draw = analyze_frame(frame, hands, pose)

        if not rec:
            last_no_det = now
            stable_word = None
            stable_done = False
        elif analyzed:
            if feat is None:
                last_no_det = now
                if len(ring) > flush_n:
                    ring = deque(list(ring)[-flush_n:], maxlen=ring.maxlen)
                if (now - stable_t) > stable_reset:
                    stable_word = None
                    stable_done = False
            else:
                ring.append(feat)
                frame_cnt += 1

                if len(ring) >= 6 and (frame_cnt % pred_step == 0):
                    word, prob, candidates = predict_from_ring(ring, model, id2label, tail_n=tail_n)
                    if word is not None:
                        last_text = word
                        last_prob = prob
                        show_until = now + 0.65
                        top_candidates = candidates

                        if prob >= min_prob:
                            if stable_word == word:
                                if (now - stable_t) >= stable_hold and not stable_done:
                                    cond_new = word != prev_word
                                    cond_rep = (last_no_det > last_add_t and (now - last_add_t) > repeat_cd)
                                    if cond_new or cond_rep:
                                        sentence.append(word)
                                        final_sentence = ""
                                        last_raw_tokens = []
                                        prev_word = word
                                        last_add_t = now
                                        stable_done = True
                            else:
                                stable_word = word
                                stable_t = now
                                stable_done = False
                        else:
                            last_no_det = now
                            if len(ring) > flush_n:
                                ring = deque(list(ring)[-flush_n:], maxlen=ring.maxlen)
                            if (now - stable_t) > stable_reset:
                                stable_word = None
                                stable_done = False

        # Center-top sentence panel appears ONLY after pressing k and converting with Ollama.
        sentence_text = final_sentence if final_sentence else ""

        if last_text and now <= show_until and not final_sentence:
            preview_word = last_text
            preview_prob = last_prob
        elif sentence:
            preview_word = sentence[-1]
            preview_prob = last_prob
        else:
            preview_word = None
            preview_prob = 0.0

        # Top GLOSS chips show the raw tokens that will be converted.
        # During accumulation: current tokens. After k: keep the last converted token list visible.
        chip_words = sentence[-8:] if sentence else last_raw_tokens[-8:]

        vis = draw_full_ui(
            frame=frame,
            rec=rec,
            buffer_len=len(ring),
            sentence_text=sentence_text,
            chip_words=chip_words,
            preview_word=preview_word,
            preview_prob=preview_prob,
            candidates=top_candidates,
            face_box=face_box,
            face_points=face_points,
            hands_for_draw=hands_for_draw,
        )

        cv2.imshow(WIN_NAME, vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("x"):
            # 디버그: 현재 ring 에 들어있는 실제 런타임 특징을 그대로 덤프한다.
            # dataset 의 녹화 클립과 비교해 어디가 다른지 확인하기 위한 것.
            if len(ring) == 0:
                print("[DUMP] buffer empty")
                continue
            arr = np.concatenate(list(ring), axis=0)
            word, prob, candidates = predict_from_ring(ring, model, id2label, tail_n=tail_n)
            DUMP_DIR.mkdir(parents=True, exist_ok=True)
            out = DUMP_DIR / f"runtime_{datetime.now():%H%M%S}.npy"
            np.save(out, {
                "feat": arr,
                "pred": word,
                "prob": prob,
                "candidates": candidates,
                "frames": int(arr.shape[0]),
            }, allow_pickle=True)
            print(f"[DUMP] {out.name}  {arr.shape[0]}프레임  예측={word} {prob:.2f}")
            print("       " + ", ".join(f"{w} {p:.2f}" for w, p in candidates[:4]))

        elif key == ord("r"):
            rec = not rec
            if rec:
                ring.clear()
                frame_cnt = 0
                print("[REC] start")
            else:
                print("[REC] pause")
            stable_word = None
            stable_done = False
            stable_t = 0.0

        elif key == ord("s"):
            if len(ring) == 0:
                print("[PRED] buffer empty")
                continue
            word, prob, candidates = predict_from_ring(ring, model, id2label, tail_n=tail_n)
            if word is None:
                print("[PRED] failed")
                continue

            last_text = word
            last_prob = prob
            show_until = time.time() + hold_sec
            top_candidates = candidates

            if prob < min_prob:
                last_no_det = time.time()
                if len(ring) > flush_n:
                    ring = deque(list(ring)[-flush_n:], maxlen=ring.maxlen)
                print(f"[PRED] (low conf) {word}: {prob:.3f}")
            else:
                sentence.append(word)
                final_sentence = ""
                last_raw_tokens = []
                prev_word = word
                last_add_t = time.time()
                print(f"[PRED] {word}: {prob:.3f}")
                if stable_word == word:
                    stable_done = True

        elif key == ord("d"):
            if sentence:
                removed = sentence.pop() 
                print(f"[DEL] '{removed}' 삭제됨")
            else:
                final_sentence = ""
                last_raw_tokens = []
                print("[DEL] final sentence cleared")
            prev_word = None
            last_add_t = 0.0

        elif key == ord("c"):
            sentence.clear()
            final_sentence = ""
            last_raw_tokens = []
            last_text = None
            last_prob = 0.0
            top_candidates = []
            prev_word = None
            last_add_t = 0.0
            print("[CLEAR] sentence cleared")

        elif key == ord("k"):
            if not sentence:
                print("[SENTENCE] 문장이 비어 있습니다.")
            else:
                raw = " ".join(sentence)
                natural = make_sentence(raw)
                print(f"[RAW] {raw}")
                print(f"[SENTENCE] {natural}")
                final_sentence = natural
                last_raw_tokens = sentence.copy()
                # Do not push the natural sentence into the bottom prediction card.
                last_text = None
                last_prob = 0.0
                show_until = 0.0
                sentence.clear()
                prev_word = None
                last_add_t = 0.0

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    pose.close()


if __name__ == "__main__":
    main()