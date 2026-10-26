import json
import time
import re
from pathlib import Path
from collections import deque

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import cv2
import mediapipe as mp
import ollama

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"


# ===== Ollama 문장 생성 유틸 =====
def _simple_tokens(s: str):
    """소문자 + 알파벳/숫자만 남기고 나눈 토큰 (가드용)"""
    s = re.sub(r"[^a-zA-Z0-9']+", " ", s.lower())
    return [t for t in s.split() if t]


def make_sentence(words: str) -> str:
    base = " ".join(words.split()).strip()
    if not base:
        return ""

    orig_tokens = _simple_tokens(base)
    if not orig_tokens:
        return base

    prompt = (
        "You are polishing a sequence of English keywords into ONE natural sentence.\n"
        "RULES:\n"
        "1. Preserve the core meaning of ALL given words.\n"
        "2. You may add small function words (articles, prepositions, pronouns, auxiliaries)\n"
        "   and adjust tense or word order to make it fluent.\n"
        "3. Do NOT invent completely new events or concrete details that are not implied.\n"
        "4. Output exactly ONE sentence and NOTHING else.\n\n"
        f"KEYWORDS:\n{base}"
    )

    try:
        resp = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": prompt}],
        )
        sent = resp["message"]["content"].strip()
    except Exception as e:
        print("[Ollama Error]", e)
        return base

    out_tokens = _simple_tokens(sent)

    # --- 문장 안정성 보장 ---
    # 원래 토큰의 일정 비율 이상은 유지
    covered = sum(1 for t in orig_tokens if t in out_tokens)
    cov_ratio = covered / len(orig_tokens)

    # 길이가 너무 폭발하면 막기
    if cov_ratio < 0.6 or len(out_tokens) > len(orig_tokens) * 2 + 6:
        print("[Ollama Guard] coverage/length check fail → fallback to raw")
        return base

    return sent


# ===== 모델 =====
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

        L = self.lstm.num_layers
        D = 2 if self.lstm.bidirectional else 1

        hn = hn.view(L, D, x.size(0), self.lstm.hidden_size)
        last = hn[-1].transpose(0, 1).contiguous().view(x.size(0), -1)

        return self.head(last)


def load_model():
    ckpt_path = CKPT_DIR / "best_lstm.pt"
    jpath = CKPT_DIR / "label2id.json"

    if not ckpt_path.exists() or not jpath.exists():
        raise FileNotFoundError("checkpoints/best_lstm.pt 또는 label2id.json 없음")

    with open(jpath, "r", encoding="utf-8") as f:
        label2id = json.load(f)

    id2label = {i: lb for lb, i in label2id.items()}

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

    model = BiLSTMClassifier(
        **cfg,
        num_classes=len(label2id),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    return model, id2label


# ===== MediaPipe 유틸 =====
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


def bbox_from_landmarks(lm_xy):
    x_min, y_min = np.min(lm_xy, axis=0)
    x_max, y_max = np.max(lm_xy, axis=0)
    return x_min, y_min, x_max, y_max


def normalize_hand(lm_xyz):
    lm = lm_xyz.copy()
    wrist = lm[0]
    lm -= wrist
    scale = np.linalg.norm(lm[9]) + 1e-6
    lm /= scale
    return lm


def build_feature_from_frame(frame_bgr, hands, pose):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Pose
    res_pose = pose.process(rgb)
    has_body = False

    ms = np.zeros(3, np.float32)
    sw = 0.0
    lsh = np.zeros(3, np.float32)
    rsh = np.zeros(3, np.float32)

    if res_pose.pose_landmarks:
        pl = res_pose.pose_landmarks.landmark
        lsh = np.array([pl[11].x, pl[11].y, pl[11].z], np.float32)
        rsh = np.array([pl[12].x, pl[12].y, pl[12].z], np.float32)
        ms = (lsh + rsh) / 2.0
        sw = float(np.linalg.norm(lsh - rsh) + 1e-6)
        has_body = True

    # Hands
    res = hands.process(rgb)
    right_norm, left_norm = None, None
    right_root, left_root = None, None

    if res.multi_hand_landmarks and res.multi_handedness:
        for lm_obj, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
            pts = np.array(
                [[lm.x, lm.y, lm.z] for lm in lm_obj.landmark],
                dtype=np.float32,
            )
            pts_xy = pts[:, :2]
            x0, y0, x1, y1 = bbox_from_landmarks(pts_xy)

            if (x1 - x0) < 0.03 and (y1 - y0) < 0.03:
                continue

            wrist_root = pts[0].copy()
            pts_norm = normalize_hand(pts)
            label = hd.classification[0].label

            if label == "Right":
                right_norm, right_root = pts_norm, wrist_root
            else:
                left_norm, left_root = pts_norm, wrist_root

    if right_norm is None and left_norm is None:
        return None

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

    return feat


# ===== 시각화 유틸 =====
def draw_label_banner(img, text, subtext=None, color=(0, 200, 0)):
    h, w = img.shape[:2]
    pad = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs_main = 1.1
    fs_sub = 0.7

    (tw, th), _ = cv2.getTextSize(text, font, fs_main, 2)

    x0 = max(10, (w - (tw + pad * 2)) // 2)
    y0 = 60

    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (x0, y0 - th - pad),
        (x0 + tw + pad * 2, y0 + pad),
        color,
        -1,
    )
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

    cv2.putText(
        img,
        text,
        (x0 + pad, y0),
        font,
        fs_main,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if subtext:
        cv2.putText(
            img,
            subtext,
            (x0 + pad, y0 + 30),
            font,
            fs_sub,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


# ===== 메인 루프 =====
def main():
    win_len = 30
    pred_step = 5
    min_prob = 0.6       # 관성 줄이려면 0.5~0.7 권장
    hold_sec = 2.0

    # === 안정 유지(홀드) 파라미터 ===
    stable_hold = 0.6    # 같은 라벨이 이 시간 이상 유지되면 단어 추가
    stable_reset = 0.4   # 이 시간 이상 "인식 없음"이면 안정 상태 리셋

    # === 옵션 A/B ===
    tail_n = 30          # A) 최근 꼬리 프레임만 사용 (8~20 권장)
    flush_n = 10         # B) 휴지기 감지 시 최근 이 개수만 남기고 플러시

    model, id2label = load_model()

    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0,
    )
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    rec = False
    ring = deque(maxlen=win_len)
    frame_cnt = 0

    # 화면 표시 / 문장 누적
    last_text = None
    last_prob = 0.0
    show_until = 0.0
    sentence = []

    # 반복/안정상태 관리
    prev_word = None
    last_add_t = 0.0
    last_no_det = 0.0
    repeat_cd = 0.5

    stable_word = None
    stable_t = 0.0
    stable_done = False

    print(
        "키: r=녹화 토글, s=즉시 예측(강제추가), d=이전 단어 삭제, "
        "k=문장 확정(ollama), q=종료"
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        vis = frame.copy()

        # HUD
        hud = f"REC: {'ON' if rec else 'OFF'}  T={len(ring)}"
        cv2.putText(
            vis,
            hud,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        now = time.time()

        if rec:
            feat = build_feature_from_frame(frame, hands, pose)

            # === B) 손이 아예 안 잡힘 → 휴지기 플러시 ===
            if feat is None:
                last_no_det = now

                if len(ring) > flush_n:
                    ring = deque(
                        list(ring)[-flush_n:],
                        maxlen=ring.maxlen,
                    )

                # 안정상태 리셋 조건
                if (now - stable_t) > stable_reset:
                    stable_word = None
                    stable_done = False

            else:
                ring.append(feat)
                frame_cnt += 1

                # 주기적 예측
                if len(ring) >= 6 and (frame_cnt % pred_step == 0):
                    # === A) 꼬리만 사용 ===
                    x = np.concatenate(list(ring), axis=0)

                    if x.shape[0] > tail_n:
                        x = x[-tail_n:]

                    x_t = torch.from_numpy(x).unsqueeze(0)
                    lengths = torch.tensor([x.shape[0]], dtype=torch.long)

                    with torch.no_grad():
                        logits = model(x_t, lengths)
                        prob = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                    i = int(prob.argmax())
                    p = float(prob[i])
                    word = id2label[i]

                    # 배너
                    last_text = word
                    last_prob = p
                    show_until = now + 0.5

                    if p >= min_prob:
                        # === 안정 유지 로직 ===
                        if stable_word == word:
                            if (now - stable_t) >= stable_hold and not stable_done:
                                cond_new = word != prev_word
                                cond_rep = (
                                    last_no_det > last_add_t
                                    and (now - last_add_t) > repeat_cd
                                )
                                if cond_new or cond_rep:
                                    sentence.append(word)
                                    prev_word = word
                                    last_add_t = now
                                    stable_done = True
                        else:
                            stable_word = word
                            stable_t = now
                            stable_done = False

                    else:
                        # === B) 확률 미달 → 휴지기 플러시 ===
                        last_no_det = now
                        if len(ring) > flush_n:
                            ring = deque(
                                list(ring)[-flush_n:],
                                maxlen=ring.maxlen,
                            )

                        if (now - stable_t) > stable_reset:
                            stable_word = None
                            stable_done = False

        else:
            last_no_det = now
            stable_word = None
            stable_done = False

        # 배너
        if last_text and time.time() <= show_until:
            draw_label_banner(
                vis,
                last_text,
                f"{last_prob:.2%}",
                color=(0, 200, 0),
            )

        # 누적 문장 (raw 단어 나열)
        full_sentence = " ".join(sentence)
        cv2.putText(
            vis,
            full_sentence,
            (20, vis.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Live Sign Recognition", vis)

        k = cv2.waitKey(1) & 0xFF

        if k == ord("q"):
            break

        elif k == ord("r"):
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

        elif k == ord("d"):
            if sentence:
                removed = sentence.pop()
                print(f"[DEL] '{removed}' 삭제됨")

            prev_word = None
            last_add_t = 0.0

        elif k == ord("s"):
            # 수동 즉시 예측(강제 추가) — A/B 적용
            if len(ring) == 0:
                print("[PRED] buffer empty")
                continue

            x = np.concatenate(list(ring), axis=0)

            if x.shape[0] > tail_n:  # A)
                x = x[-tail_n:]

            x_t = torch.from_numpy(x).unsqueeze(0)
            lengths = torch.tensor([x.shape[0]], dtype=torch.long)

            with torch.no_grad():
                logits = model(x_t, lengths)
                prob = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            i = int(prob.argmax())
            p = float(prob[i])
            word = id2label[i]

            last_text = word
            last_prob = p
            show_until = time.time() + hold_sec

            if p < min_prob:
                # B) 수동 예측이라도 너무 낮으면 휴지기 취급(추가는 하지 않고 flush만)
                last_no_det = time.time()
                if len(ring) > flush_n:
                    ring = deque(
                        list(ring)[-flush_n:],
                        maxlen=ring.maxlen,
                    )
                print(f"[PRED] (low conf) {word}: {p:.3f}")
            else:
                sentence.append(word)
                prev_word = word
                last_add_t = time.time()
                print(f"[PRED] {word}: {p:.3f}")

                # 현재 안정상태와 동일 단어라면 중복 커밋 방지
                if stable_word == word:
                    stable_done = True

        elif k == ord("k"):
            # === Ollama로 문장 확정 ===
            if not sentence:
                print("[SENTENCE] 문장이 비어 있습니다.")
            else:
                raw = " ".join(sentence)
                natural = make_sentence(raw)

                print(f"[RAW] {raw}")
                print(f"[SENTENCE] {natural}")

                last_text = natural
                last_prob = 1.0
                show_until = time.time() + 3.0

                sentence.clear()
                prev_word = None
                last_add_t = 0.0

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    pose.close()


if __name__ == "__main__":
    main()