# -*- coding: utf-8 -*-
"""
SLingo 2D 아바타 뷰어 (가벼운 cv2 창).
collect.py 처럼 MediaPipe 손 랜드마크를 쓰되, 점 대신 '귀여운 2D 캐릭터'로 그린다.

모드:
  --clip 안녕      저장된 클립(.npy) 재생 (웹캠 불필요, 즉시)
  --live           웹캠 실시간 + 수어 제스처 인식 (collect.py 와 동일한 MediaPipe/전처리)
  --live --record  위 + R 키로 학습용 아바타 영상(mp4) 녹화
  --shot out.png   중간 프레임 1장 PNG로 저장(확인용, 창 안 띄움)

조작(클립 창): SPACE 일시정지 · N/P 다음/이전 단어 · ← → 한프레임 · Q 종료
조작(라이브 창): R 영상녹화 · SPACE 즉시 판정 · C 버퍼 비우기 · M 카메라 패널 · Q 종료

녹화: R → 3초 카운트다운 → 5초 녹화 → videos/001.mp4, 002.mp4 ... 로 자동 연번 저장.
      HUD/카메라 패널 없이 720x720 아바타 캔버스만 15fps 로 들어간다.
      녹화 중 R 또는 Q 를 누르면 취소하고 파일을 지운다.
"""
import argparse, sys, time, math
from pathlib import Path
import numpy as np, cv2

BASE = Path("/Users/shkim/Desktop/2026 전국전람회/code/slingo_dataset_maker/dataset")
MODELS = Path("/Users/shkim/Desktop/2026 전국전람회/code/slingo_dataset_maker/models")
CKPT = Path("/Users/shkim/Desktop/2026 전국전람회/code/slingo_dataset_maker/checkpoints/best_lstm.pt")

# --- collect.py 와 반드시 같아야 하는 검출/전처리 상수 ---
DET_CONF = TRK_CONF = PRESENCE_CONF = 0.5
NUM_HANDS = 2
HAND_MIN_BOX = 0.03      # 너무 작은 손 검출 제거
CAP_W, CAP_H = 640, 480  # 앱 분석 해상도
FPS_TARGET = 15.0        # 학습 데이터와 동일한 시퀀스 속도

# --- 제스처 구간 자동 분할 (collect.py 자동수집 게이트와 동일) ---
START_CONSEC = 5         # 연속 '손 있음' → 구간 시작
STOP_CONSEC = 8          # 연속 '손 없음' → 구간 종료 후 판정
TRIM_TAIL = 3            # 끝에서 손 빠지는 프레임 잘라냄
MIN_PRED_FRAMES = 10     # 이보다 짧으면 판정 안 함
MAX_SEQ_FRAMES = 150     # 폭주 방지 상한
HOLD_SEC = 2.5           # 인식 결과 표시 유지 시간

# --- 학습용 영상 녹화 (R 키) ---
VIDEO_DIR = Path(__file__).resolve().parent / "videos"
COUNTDOWN_SEC = 3.0      # R 누른 뒤 준비 시간
RECORD_SEC = 5.0         # 녹화 길이
REC_FPS = 15.0           # 저장 fps (= FPS_TARGET, 아바타가 갱신되는 속도)

INPUT_DIM = 139          # 손63+손63+마스크2+몸1+어깨중점3+어깨너비1+상대좌표3+3

W = H = 720
# 고정 앵커: 어깨 중점과 어깨너비를 캔버스에 고정 → 아바타가 안 흔들리고 중앙에 섬
CX, CY = int(W*0.5), int(H*0.44)
SW = int(W*0.26)                 # 어깨너비(px)
HAND = 0.30                      # 손 크기 = 어깨너비의 0.30배(정규화손 곱)

# 색 (BGR) — 파스텔 귀여운 톤
COL_BG     = (245, 240, 250)
COL_SKIN   = (200, 218, 246)
COL_SKIN_L = (170, 195, 235)
COL_HAIR   = (120, 90, 150)
COL_CLOTH  = (210, 180, 175)
COL_CLOTH2 = (190, 155, 150)
COL_CHEEK  = (150, 170, 240)
COL_LINE   = (110, 100, 130)

FINGERS = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]

# ----------------- 한글 텍스트 (cv2.putText 는 한글 불가 → PIL) -----------------
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
    if size not in _FONT_CACHE:
        font = None
        for path in _FONT_CANDIDATES:
            if Path(path).exists():
                try:
                    font = ImageFont.truetype(path, size); break
                except Exception:
                    continue
        _FONT_CACHE[size] = font
    return _FONT_CACHE[size]

def put_text(img, text, org, size=24, color=(255,255,255)):
    """한글 포함 텍스트를 BGR 이미지에 그린다(좌상단 기준). 폰트 없으면 cv2 폴백."""
    x, y = org
    if _PIL_OK:
        font = _load_font(size)
        if font is not None:
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ImageDraw.Draw(pil).text((x, y), text, font=font,
                                     fill=(color[2], color[1], color[0]))
            img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            return
    cv2.putText(img, text, (x, y+size), cv2.FONT_HERSHEY_SIMPLEX,
                size/30.0, color, 2, cv2.LINE_AA)

def capsule(img, p1, p2, r, color):
    cv2.line(img, p1, p2, color, r*2, cv2.LINE_AA)
    cv2.circle(img, p1, r, color, -1, cv2.LINE_AA)
    cv2.circle(img, p2, r, color, -1, cv2.LINE_AA)

def draw_hand(img, pts):
    """pts: 21x2 정수 픽셀."""
    # 손바닥 채우기
    palm = np.array([pts[0], pts[1], pts[5], pts[9], pts[13], pts[17]], np.int32)
    cv2.fillConvexPoly(img, cv2.convexHull(palm), COL_SKIN, cv2.LINE_AA)
    # 손가락 캡슐
    for f in FINGERS:
        chain = [0]+f if False else f
        prev = pts[0] if f[0]==1 else pts[f[0]-1] if False else pts[f[0]]
        # 손목/MCP -> 끝까지 두께 점점 감소
        pathpts = [pts[0], pts[f[0]], pts[f[1]], pts[f[2]], pts[f[3]]] if f[0]!=1 \
                  else [pts[0], pts[1], pts[2], pts[3], pts[4]]
        for k in range(len(pathpts)-1):
            r = max(5, 9-k*1)
            capsule(img, tuple(pathpts[k]), tuple(pathpts[k+1]), r, COL_SKIN)
    # 손끝 마디 강조
    for f in FINGERS:
        cv2.circle(img, tuple(pts[f[-1]]), 5, COL_SKIN_L, -1, cv2.LINE_AA)
    # 손목 원
    cv2.circle(img, tuple(pts[0]), 11, COL_SKIN, -1, cv2.LINE_AA)

def draw_face(img, cx, cy, r):
    # 머리
    cv2.circle(img, (cx, cy), r, COL_SKIN, -1, cv2.LINE_AA)
    # 머리카락(위쪽 반원 + 옆머리)
    cv2.ellipse(img, (cx, cy-2), (r+3, r+3), 0, 180, 360, COL_HAIR, -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy-2), (r+3, r+3), 0, 150, 210, COL_HAIR, 8, cv2.LINE_AA)
    cv2.circle(img, (cx-int(r*0.7), cy-int(r*0.9)), 8, COL_HAIR, -1, cv2.LINE_AA)  # 트윈테일 느낌
    cv2.circle(img, (cx+int(r*0.7), cy-int(r*0.9)), 8, COL_HAIR, -1, cv2.LINE_AA)
    # 볼터치
    cv2.circle(img, (cx-int(r*0.45), cy+int(r*0.25)), 6, COL_CHEEK, -1, cv2.LINE_AA)
    cv2.circle(img, (cx+int(r*0.45), cy+int(r*0.25)), 6, COL_CHEEK, -1, cv2.LINE_AA)
    # 눈
    for sx in (-1, 1):
        ex = cx + sx*int(r*0.4)
        ey = cy + int(r*0.05)
        cv2.circle(img, (ex, ey), 7, (60,50,70), -1, cv2.LINE_AA)
        cv2.circle(img, (ex-2, ey-2), 2, (255,255,255), -1, cv2.LINE_AA)
    # 입(미소)
    cv2.ellipse(img, (cx, cy+int(r*0.4)), (int(r*0.22), int(r*0.16)), 0, 20, 160, (90,90,150), 3, cv2.LINE_AA)

def draw_avatar(relR, handR, pR, relL, handL, pL, label=""):
    img = np.full((H, W, 3), COL_BG, np.uint8)
    # 어깨 고정
    Lsh = (CX + SW//2, CY); Rsh = (CX - SW//2, CY)
    # 몸통(사다리꼴)
    hipY = CY + int(H*0.32)
    body = np.array([[Rsh[0]-10, Rsh[1]], [Lsh[0]+10, Lsh[1]],
                     [Lsh[0]-6, hipY], [Rsh[0]+6, hipY]], np.int32)
    cv2.fillConvexPoly(img, body, COL_CLOTH, cv2.LINE_AA)
    cv2.polylines(img, [body], True, COL_CLOTH2, 2, cv2.LINE_AA)
    # 목 + 머리
    headR = int(SW*0.42)
    cv2.rectangle(img, (CX-10, CY-headR//2), (CX+10, CY+6), COL_SKIN, -1, cv2.LINE_AA)
    draw_face(img, CX, CY - headR - 6, headR)

    def hand_pts(rel, hand):
        wx = CX + rel[0]*SW; wy = CY + rel[1]*SW
        pts = []
        for i in range(21):
            x = wx + hand[i,0]*HAND*SW
            y = wy + hand[i,1]*HAND*SW
            pts.append([int(x), int(y)])
        return np.array(pts, np.int32), (int(wx), int(wy))

    # 팔(어깨->손목) + 손
    for present, rel, hand, sh in ((pR, relR, handR, Rsh), (pL, relL, handL, Lsh)):
        if present:
            pts, wrist = hand_pts(rel, hand)
            capsule(img, sh, wrist, 12, COL_CLOTH)      # 소매
            capsule(img, sh, wrist, 8, COL_CLOTH2)
            draw_hand(img, pts)
        else:
            # 손 없으면 팔 살짝 내림
            rest = (sh[0]+ (18 if sh is Lsh else -18), sh[1]+int(H*0.18))
            capsule(img, sh, rest, 12, COL_CLOTH)
    if label:
        cv2.putText(img, label, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, COL_LINE, 2, cv2.LINE_AA)
    return img

# ----------------- 클립 재생 -----------------
def load_clip(word):
    d = BASE / word
    clips = sorted(d.glob("clip_*.npy"))
    if not clips: return None
    # 손 커버리지 좋은 클립
    best, bestcov = None, -1
    for c in clips:
        o = np.load(c, allow_pickle=True).item()
        cov = np.asarray(o["hand_mask"]).mean()
        if cov > bestcov: best, bestcov = o, cov
    return best

def frames_from_clip(o):
    N = int(o["meta"]["frames"])
    R, L = np.asarray(o["right"]), np.asarray(o["left"])
    rr, lr = np.asarray(o["right_rel_ms"]), np.asarray(o["left_rel_ms"])
    hm = np.asarray(o["hand_mask"])
    out = []
    for t in range(N):
        out.append((rr[t], R[t], bool(hm[t,0]), lr[t], L[t], bool(hm[t,1])))
    return out

def words_list():
    return sorted([p.name for p in BASE.iterdir() if p.is_dir()])

def play_clips(start_word):
    words = words_list()
    wi = words.index(start_word) if start_word in words else 0
    while True:
        word = words[wi]
        o = load_clip(word)
        if o is None: wi = (wi+1) % len(words); continue
        frames = frames_from_clip(o)
        i, paused = 0, False
        while True:
            fr = frames[i]
            img = draw_avatar(*fr, label=f"{word}  {i+1}/{len(frames)}")
            cv2.imshow("SLingo Avatar", img)
            k = cv2.waitKey(0 if paused else 70) & 0xFF
            if k == ord('q'): cv2.destroyAllWindows(); return
            elif k == ord(' '): paused = not paused
            elif k in (ord('n'),): wi=(wi+1)%len(words); break
            elif k in (ord('p'),): wi=(wi-1)%len(words); break
            elif k == 83: i=(i+1)%len(frames); paused=True   # →
            elif k == 81: i=(i-1)%len(frames); paused=True   # ←
            else:
                if not paused: i = (i+1) % len(frames)

# ----------------- 특징 추출 (collect.py / train.py 와 1:1 동일) -----------------
def normalize_hand(pts):
    """손목(0) 기준 평행이동 후 ||lm[9]-wrist|| 로 정규화 → (21,3)"""
    lm = pts.copy()
    lm -= lm[0].copy()
    return lm / (float(np.linalg.norm(lm[9])) + 1e-6)

def build_record(hand_res, pose_res):
    """한 프레임 → record dict. 양손 모두 없으면 None."""
    has_body, ms, sw = False, np.zeros(3, np.float32), 0.0
    if pose_res.pose_landmarks:
        pl = pose_res.pose_landmarks[0]
        if len(pl) > 12:
            l, r = pl[11], pl[12]
            lsh = np.array([l.x, l.y, l.z], np.float32)
            rsh = np.array([r.x, r.y, r.z], np.float32)
            ms = (lsh + rsh) / 2.0
            sw = float(np.linalg.norm(lsh - rsh) + 1e-6)
            has_body = True

    right_norm = left_norm = right_root = left_root = None
    handed = hand_res.handedness or []
    for i, lm in enumerate(hand_res.hand_landmarks or []):
        if len(lm) < 21: continue
        pts = np.array([[p.x, p.y, p.z] for p in lm], np.float32)
        if (pts[:,0].max()-pts[:,0].min()) < HAND_MIN_BOX and \
           (pts[:,1].max()-pts[:,1].min()) < HAND_MIN_BOX:
            continue                                  # 너무 작은 검출 버림
        label = handed[i][0].category_name if i < len(handed) and handed[i] else "Right"
        if label == "Right": right_norm, right_root = normalize_hand(pts), pts[0].copy()
        else:                left_norm,  left_root  = normalize_hand(pts), pts[0].copy()

    if right_norm is None and left_norm is None:
        return None

    def rel_ms(root):
        if root is None or not has_body or sw <= 0.0:
            return np.zeros(3, np.float32)
        return ((root - ms) / sw).astype(np.float32)

    return {
        "hand_mask": np.array([right_norm is not None, left_norm is not None], np.float32),
        "right": right_norm if right_norm is not None else np.zeros((21,3), np.float32),
        "left":  left_norm  if left_norm  is not None else np.zeros((21,3), np.float32),
        "body_mask": float(has_body), "ms": ms.astype(np.float32), "sw": np.float32(sw),
        "right_rel_ms": rel_ms(right_root), "left_rel_ms": rel_ms(left_root),
    }

def features_from_records(records):
    """record 리스트 → (T,139). train.py build_frame_features 와 동일한 순서."""
    rows = []
    for r in records:
        rows.append(np.concatenate([
            r["right"].reshape(-1), r["left"].reshape(-1), r["hand_mask"],
            [r["body_mask"]], r["ms"], [r["sw"]], r["right_rel_ms"], r["left_rel_ms"],
        ]).astype(np.float32))
    feat = np.stack(rows)
    assert feat.shape[1] == INPUT_DIM, f"특징 차원 {feat.shape[1]} != {INPUT_DIM}"
    np.nan_to_num(feat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return feat

# ----------------- 수어 분류기 (train.py 와 동일 구조) -----------------
class SignRecognizer:
    def __init__(self, ckpt=CKPT):
        import torch
        from torch import nn
        self.torch, self.nn = torch, nn
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = blob["config"]
        self.label2id = blob["label2id"]
        self.id2label = {v: k for k, v in self.label2id.items()}

        class BiLSTMClassifier(nn.Module):
            def __init__(s, input_dim, num_classes, hidden, layers, dropout, bidir):
                super().__init__()
                s.lstm = nn.LSTM(input_dim, hidden, num_layers=layers,
                                 dropout=dropout if layers > 1 else 0.0,
                                 bidirectional=bidir, batch_first=True)
                out_dim = hidden * (2 if bidir else 1)
                s.head = nn.Sequential(nn.LayerNorm(out_dim), nn.Dropout(dropout),
                                       nn.Linear(out_dim, num_classes))
            def forward(s, x):
                _, (hn, _) = s.lstm(x)
                L, D = s.lstm.num_layers, (2 if s.lstm.bidirectional else 1)
                hn = hn.view(L, D, x.size(0), s.lstm.hidden_size)
                return s.head(hn[-1].transpose(0, 1).contiguous().view(x.size(0), -1))

        self.model = BiLSTMClassifier(cfg["input_dim"], len(self.label2id), cfg["hidden"],
                                      cfg["layers"], cfg["dropout"], cfg["bidir"])
        self.model.load_state_dict(blob["model"])
        self.model.eval()

    def predict(self, records, topk=3):
        """[(라벨, 확률), ...] 상위 topk."""
        feat = features_from_records(records)
        with self.torch.no_grad():
            x = self.torch.from_numpy(feat).unsqueeze(0)          # (1,T,139)
            prob = self.torch.softmax(self.model(x), dim=1)[0].numpy()
        order = np.argsort(-prob)[:topk]
        return [(self.id2label[int(i)], float(prob[i])) for i in order]

# ----------------- 학습용 영상 녹화 -----------------
def next_video_index(outdir: Path) -> int:
    """outdir 안의 001.mp4, 002.mp4 ... 다음 번호."""
    mx = 0
    if outdir.exists():
        for f in outdir.glob("*.mp4"):
            if f.stem.isdigit():
                mx = max(mx, int(f.stem))
    return mx + 1

class ClipRecorder:
    """R 키 → 카운트다운 → 고정 길이/고정 fps 로 아바타 캔버스만 mp4 저장.

    실제 렌더가 REC_FPS 보다 느려도 벽시계 기준으로 프레임을 채워 넣으므로
    저장된 영상의 재생 시간은 항상 RECORD_SEC 로 맞는다.
    """
    def __init__(self, outdir=VIDEO_DIR, countdown=COUNTDOWN_SEC,
                 length=RECORD_SEC, fps=REC_FPS, size=(W, H)):
        self.outdir, self.countdown, self.length = Path(outdir), countdown, length
        self.fps, self.size = fps, size
        self.total = max(1, int(round(length * fps)))
        self.state = "idle"          # idle | countdown | recording
        self.writer = None
        self.path = None
        self.t_start = 0.0           # 녹화 시작 시각
        self.next_write = 0.0
        self.written = 0

    def arm(self, now):
        """R: 대기중이면 카운트다운 시작, 진행중이면 취소."""
        if self.state != "idle":
            self.cancel(); return
        self.state = "countdown"
        self.t_start = now + self.countdown
        print(f"[REC] {self.countdown:.0f}초 뒤 시작 · {self.length:.0f}초 녹화")

    def _open(self):
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.path = self.outdir / f"{next_video_index(self.outdir):03d}.mp4"
        for cc in ("avc1", "mp4v"):
            w = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*cc),
                                self.fps, self.size)
            if w.isOpened():
                self.writer = w; return True
            w.release()
        print("[ERR] VideoWriter 열기 실패 → 녹화 취소")
        self.state, self.path = "idle", None
        return False

    def update(self, now, clean_frame):
        """매 루프 호출. clean_frame = HUD 없는 아바타 캔버스."""
        if self.state == "countdown":
            if now >= self.t_start:
                if not self._open(): return
                self.state = "recording"
                self.next_write, self.written = self.t_start, 0
                print(f"[REC] 녹화 시작 → {self.path.name}")
            else:
                return
        if self.state != "recording":
            return
        # 벽시계 기준 고정 간격으로 프레임을 채운다(렌더가 느리면 직전 프레임 복제)
        step = 1.0 / self.fps
        while self.written < self.total and now >= self.next_write:
            self.writer.write(clean_frame)
            self.written += 1
            self.next_write += step
        if self.written >= self.total:
            self._close()
            print(f"[REC] 저장 완료 {self.path.name}  "
                  f"({self.total}프레임 / {self.length:.0f}초 @ {self.fps:.0f}fps)")
            self.state = "idle"

    def _close(self):
        if self.writer is not None:
            self.writer.release(); self.writer = None

    def cancel(self):
        was = self.state
        self._close()
        if was == "recording" and self.path and self.path.exists():
            self.path.unlink()
            print(f"[REC] 취소 — {self.path.name} 삭제")
        elif was == "countdown":
            print("[REC] 카운트다운 취소")
        self.state, self.path = "idle", None

    def hud(self, img, now):
        """카운트다운 숫자 / 녹화 표시 (화면에만, 저장 영상엔 안 들어감)."""
        if self.state == "countdown":
            left = self.t_start - now
            put_text(img, str(max(1, int(math.ceil(left)))),
                     (W//2 - 40, H//2 - 110), 190, (90, 120, 220))
            put_text(img, "준비", (W//2 - 34, H//2 + 110), 34, (120, 130, 190))
        elif self.state == "recording":
            remain = max(0.0, self.length - self.written / self.fps)
            cv2.circle(img, (W - 44, 30), 11, (60, 60, 235), -1, cv2.LINE_AA)
            put_text(img, f"REC {remain:.1f}s", (W - 168, 16), 24, (60, 60, 235))
            cv2.rectangle(img, (0, H-6), (int(W * self.written / self.total), H),
                          (60, 60, 235), -1)

# ----------------- 웹캠 실시간 + 인식 -----------------
def draw_hud(img, gate, buflen, result, hint, conf_min):
    """아바타 캔버스 위에 인식 상태/결과 HUD."""
    cv2.rectangle(img, (0, 0), (W, 74), (250, 248, 252), -1)
    color = (80, 180, 80) if gate == "REC" else (150, 150, 160)
    cv2.circle(img, (26, 30), 9, color, -1, cv2.LINE_AA)
    put_text(img, f"{gate}   frames {buflen}", (46, 16), 22, (110, 100, 130))
    put_text(img, hint, (46, 44), 17, (160, 155, 175))

    if result:
        word, prob = result[0]
        ok = prob >= conf_min
        cv2.rectangle(img, (0, H-136), (W, H), (250, 248, 252), -1)
        put_text(img, word if ok else "?", (26, H-128), 58,
                 (70, 140, 70) if ok else (140, 140, 150))
        put_text(img, f"{prob*100:.0f}%", (26, H-56), 26, (110, 100, 130))
        put_text(img, "   ".join(f"{w} {p*100:.0f}%" for w, p in result[1:]),
                 (150, H-50), 20, (165, 160, 180))

def run_live(show_cam=False, conf_min=0.55, recognize=True, recorder=None):
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision as mpv

    rec_model = None
    if recognize:
        if not CKPT.exists():
            print(f"[WARN] 체크포인트 없음 → 인식 끔: {CKPT}")
        else:
            print("[..] 수어 모델 로딩")
            rec_model = SignRecognizer()
            print(f"[OK] 라벨 {len(rec_model.label2id)}개: {list(rec_model.label2id)}")

    ho = mpv.HandLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=str(MODELS/"hand_landmarker.task")),
                                   running_mode=mpv.RunningMode.IMAGE, num_hands=NUM_HANDS,
                                   min_hand_detection_confidence=DET_CONF,
                                   min_hand_presence_confidence=PRESENCE_CONF,
                                   min_tracking_confidence=TRK_CONF)
    po = mpv.PoseLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=str(MODELS/"pose_landmarker.task")),
                                   running_mode=mpv.RunningMode.IMAGE,
                                   min_pose_detection_confidence=DET_CONF,
                                   min_pose_presence_confidence=PRESENCE_CONF,
                                   min_tracking_confidence=TRK_CONF)
    hl = mpv.HandLandmarker.create_from_options(ho)
    pl = mpv.PoseLandmarker.create_from_options(po)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    if not cap.isOpened():
        print("[ERR] 웹캠을 열 수 없음"); return

    dt = 1.0 / FPS_TARGET
    next_t = 0.0
    buffer, started, start_run, stop_run = [], False, 0, 0
    result, result_at = None, 0.0
    last_pose = (np.zeros(3), np.zeros((21,3)), False, np.zeros(3), np.zeros((21,3)), False)
    hint = ("R 영상녹화 · SPACE 즉시판정 · C 초기화 · M 카메라 · Q 종료"
            if recorder else "손을 들면 자동 인식 · SPACE 즉시판정 · C 초기화 · M 카메라 · Q 종료")
    if recorder:
        print(f"[REC] 저장 폴더 {recorder.outdir}  다음 파일 "
              f"{next_video_index(recorder.outdir):03d}.mp4")

    def judge(force=False):
        """버퍼를 하나의 제스처로 보고 판정. 판정했으면 True."""
        nonlocal buffer, result, result_at, started, start_run, stop_run
        cut = 0 if force else min(TRIM_TAIL, len(buffer))
        clip = buffer[:-cut] if cut > 0 else buffer[:]
        done = False
        if len(clip) >= MIN_PRED_FRAMES and rec_model is not None:
            result = rec_model.predict(clip)
            result_at = time.time()
            print(f"[PRED] {len(clip)}프레임 → " +
                  "  ".join(f"{w} {p*100:.1f}%" for w, p in result))
            done = True
        elif len(clip) < MIN_PRED_FRAMES:
            print(f"[SKIP] 너무 짧음 ({len(clip)}<{MIN_PRED_FRAMES})")
        buffer, started, start_run, stop_run = [], False, 0, 0
        return done

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERR] 프레임 읽기 실패"); break
        frame = cv2.flip(frame, 1)                  # 거울(collect.py 동일)

        now = time.time()
        if now >= next_t:                           # 15fps 로 상한 → 학습 데이터와 동일 속도
            next_t = now + dt
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            rec = build_record(hl.detect(mpimg), pl.detect(mpimg))

            # ---- 아바타 포즈 ----
            if rec is not None:
                last_pose = (rec["right_rel_ms"], rec["right"], bool(rec["hand_mask"][0]),
                             rec["left_rel_ms"],  rec["left"],  bool(rec["hand_mask"][1]))
            else:
                last_pose = (np.zeros(3), np.zeros((21,3)), False,
                             np.zeros(3), np.zeros((21,3)), False)

            # ---- 제스처 구간 자동 분할 ----
            if not started:
                if rec is not None:
                    buffer.append(rec); start_run += 1
                    if start_run >= START_CONSEC:
                        started = True                 # 시작 확정 (앞부분도 버퍼에 남김)
                else:
                    buffer.clear(); start_run = 0
            else:
                if rec is not None:
                    buffer.append(rec); stop_run = 0
                    if len(buffer) >= MAX_SEQ_FRAMES:
                        judge(force=True)
                else:
                    stop_run += 1
                    if stop_run >= STOP_CONSEC:
                        judge()                        # 손 사라짐 → 구간 끝, 판정

        img = draw_avatar(*last_pose)
        if recorder:
            # HUD 를 얹기 전의 깨끗한 아바타 캔버스만 영상으로 저장
            recorder.update(time.time(), img)
        gate = "REC" if started else "WAIT"
        shown = result if (result and now - result_at < HOLD_SEC) else None
        draw_hud(img, gate, len(buffer), shown, hint, conf_min)
        if recorder:
            recorder.hud(img, time.time())
        if show_cam:
            panel = np.full((H, 360, 3), 30, np.uint8)
            small = cv2.resize(frame, (360, int(360*frame.shape[0]/frame.shape[1])))
            y0 = (H - small.shape[0]) // 2
            panel[y0:y0+small.shape[0]] = small
            img = np.hstack([img, panel])
        cv2.imshow("SLingo Avatar (live)", img)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        elif k == ord(' '): judge(force=True)
        elif k == ord('c'):
            buffer, started, start_run, stop_run = [], False, 0, 0
            result = None
            print("[CLEAR] 버퍼 비움")
        elif k == ord('m'): show_cam = not show_cam
        elif k == ord('r') and recorder: recorder.arm(time.time())

    if recorder: recorder.cancel()      # 녹화 중 종료 → 잘린 파일 남기지 않음
    cap.release(); cv2.destroyAllWindows()

# ----------------- main -----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="안녕")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--shot", default="")
    ap.add_argument("--cam", action="store_true", help="라이브: 웹캠 원본도 옆에 표시")
    ap.add_argument("--no-recog", action="store_true", help="라이브: 인식 끄고 아바타만")
    ap.add_argument("--conf", type=float, default=0.55, help="이 확률 미만이면 '?' 표시")
    ap.add_argument("--record", action="store_true",
                    help="라이브: R 키로 학습용 아바타 영상(mp4) 녹화")
    ap.add_argument("--outdir", default=str(VIDEO_DIR), help="녹화 저장 폴더")
    ap.add_argument("--countdown", type=float, default=COUNTDOWN_SEC, help="R 누른 뒤 준비 시간(초)")
    ap.add_argument("--seconds", type=float, default=RECORD_SEC, help="녹화 길이(초)")
    ap.add_argument("--fps", type=float, default=REC_FPS, help="저장 영상 fps")
    args = ap.parse_args()

    if args.shot:
        o = load_clip(args.clip); frames = frames_from_clip(o)
        f = frames[len(frames)//2]
        cv2.imwrite(args.shot, draw_avatar(*f, label=args.clip))
        print("shot saved:", args.shot); sys.exit(0)
    if args.live:
        rec = ClipRecorder(args.outdir, args.countdown, args.seconds, args.fps) \
              if args.record else None
        run_live(show_cam=args.cam, conf_min=args.conf,
                 recognize=not args.no_recog, recorder=rec)
    else:
        play_clips(args.clip)
