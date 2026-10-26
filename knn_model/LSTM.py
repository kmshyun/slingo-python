import os, json, glob, argparse, random
import numpy as np
from typing import List, Dict, Tuple

# ====== 0) 공통 설정 ======
SEED = 42
random.seed(SEED); np.random.seed(SEED)

MAX_LEN = 64          # 프레임 윈도우 길이
STRIDE = 8            # 윈도우 이동 간격
FEATURES = 21 * 3     # 손 랜드마크 (x,y,z) * 21
BATCH = 64
EPOCHS = 20

# ====== 1) 데이터 적재 (LEARNED/*.json) ======
def load_learned_dir(learned_dir: str) -> Dict[str, np.ndarray]:
    """
    각 파일명(라벨).json을 읽어 frames(T, 21, 3) -> (T, 63)로 변환해 반환
    """
    data = {}
    for path in glob.glob(os.path.join(learned_dir, "*.json")):
        label = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r") as f:
            obj = json.load(f)
        arr = np.array(obj["data"], dtype=np.float32)  # (T, 21, 3)
        if arr.ndim != 3 or arr.shape[1:] != (21, 3):
            raise ValueError(f"{path}: 예상한 (T,21,3)이 아님.")
        flat = arr.reshape(arr.shape[0], -1)           # (T, 63)
        data[label] = flat
    if not data:
        raise FileNotFoundError(f"{learned_dir} 에 json이 없음.")
    return data

def make_sliding_windows(arr: np.ndarray, label_idx: int, max_len=MAX_LEN, stride=STRIDE) -> Tuple[np.ndarray, np.ndarray]:
    """
    한 제스처 시퀀스(T,63)에서 슬라이딩 윈도우로 학습 샘플 생성
    """
    xs, ys = [], []
    T = arr.shape[0]
    if T < max_len:
        # 패딩해서 1개 샘플 만들기
        pad = np.zeros((max_len, arr.shape[1]), dtype=np.float32)
        pad[-T:] = arr
        xs.append(pad); ys.append(label_idx)
    else:
        for s in range(0, T - max_len + 1, stride):
            xs.append(arr[s:s+max_len])
            ys.append(label_idx)
        # 끝쪽 남은 꼬리도 하나 만들어줌
        tail = arr[max(0, T - max_len):T]
        if tail.shape[0] < max_len:
            pad = np.zeros((max_len, arr.shape[1]), dtype=np.float32)
            pad[-tail.shape[0]:] = tail
            tail = pad
        xs.append(tail); ys.append(label_idx)
    return np.stack(xs), np.array(ys)

def build_gesture_dataset(learned_dir: str):
    data = load_learned_dir(learned_dir)
    labels = sorted(list(data.keys()))
    label2id = {lb:i for i,lb in enumerate(labels)}
    Xs, Ys = [], []
    for lb, arr in data.items():
        x, y = make_sliding_windows(arr, label2id[lb])
        Xs.append(x); Ys.append(y)
    X = np.concatenate(Xs, axis=0)   # (N, L, 63)
    y = np.concatenate(Ys, axis=0)   # (N,)
    # 셔플
    idx = np.arange(len(X)); np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    # train/val split
    n_val = max(1, int(0.1 * len(X)))
    X_train, y_train = X[n_val:], y[n_val:]
    X_val,   y_val   = X[:n_val], y[:n_val]
    return (X_train, y_train, X_val, y_val, labels)


import tensorflow as tf
from tensorflow.keras import layers, models

class SimpleAttention(layers.Layer):
    def __init__(self, units=128, **kwargs):
        super().__init__(**kwargs)
        self.W = layers.Dense(units, activation="tanh")
        self.v = layers.Dense(1)

    def call(self, h, mask=None):
        # h: (B, T, H)
        score = self.v(self.W(h))  # (B,T,1)
        if mask is not None:
            mask = tf.cast(mask[:, :, None], tf.float32)  # (B,T,1)
            score = score - 1e9 * (1. - mask)
        attn = tf.nn.softmax(score, axis=1)               # (B,T,1)
        context = tf.reduce_sum(attn * h, axis=1)         # (B,H)
        return context

def build_gesture_model(n_classes: int, max_len=MAX_LEN, feat=FEATURES):
    inp = layers.Input(shape=(max_len, feat))
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    context = SimpleAttention(128)(x)
    x = layers.Dense(256, activation="relu")(context)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# ====== 3) 제스처 시퀀스 → 라벨 추론 ======
def predict_gesture_tokens(model_path: str, labels_path: str, seq_path: str) -> List[str]:
    """
    seq_path: 새 영상에서 뽑아 저장한 json(위 포맷 동일, {"data": [...]} )
    출력: 제스처 라벨 토큰 시퀀스 (윈도우 단위 예측을 합쳐 압축)
    """
    model = tf.keras.models.load_model(model_path, custom_objects={"SimpleAttention": SimpleAttention})
    with open(labels_path, "r") as f:
        labels = json.load(f)
    flat = np.array(json.load(open(seq_path))["data"], dtype=np.float32).reshape(-1, 21*3)

    # 윈도우 단위 예측
    preds = []
    T = flat.shape[0]
    if T < MAX_LEN:
        pad = np.zeros((MAX_LEN, flat.shape[1]), dtype=np.float32)
        pad[-T:] = flat
        X = pad[None, ...]
        p = model.predict(X, verbose=0)[0]
        preds.append(np.argmax(p))
    else:
        for s in range(0, T - MAX_LEN + 1, STRIDE):
            X = flat[s:s+MAX_LEN][None, ...]
            p = model.predict(X, verbose=0)[0]
            preds.append(np.argmax(p))
    # 간단한 연속 중복 압축
    compressed = []
    for idx in preds:
        if not compressed or compressed[-1] != idx:
            compressed.append(idx)
    return [labels[i] for i in compressed]

# ====== 4) 번역 단계 ======
# (A) 규칙기반: 바로 동작
RULES = {
    # 예시 규칙: 필요에 맞게 계속 추가/수정
    ("Wave",): "안녕하세요.",
    ("17",): "열일곱입니다.",
    ("Wave", "17"): "안녕하세요, 열일곱입니다.",
}

def rule_translate(tokens: List[str]) -> str:
    # 가장 긴 매칭부터 찾는 간단한 규칙 번역
    i = 0; out = []
    while i < len(tokens):
        matched = False
        for L in range(min(4, len(tokens)-i), 0, -1):
            key = tuple(tokens[i:i+L])
            if key in RULES:
                out.append(RULES[key])
                i += L; matched = True; break
        if not matched:
            out.append(tokens[i])  # 미정의는 토큰 그대로 노출
            i += 1
    return " ".join(out)

# (B) 선택: 병렬 코퍼스가 있을 때 seq2seq 학습/사용
# pairs.jsonl: {"gestures":["Wave","17"], "text":"안녕하세요, 열일곱입니다."}
def load_pairs(pairs_path: str):
    pairs = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pairs.append((obj["gestures"], obj["text"]))
    return pairs

def build_vocab_from_pairs(pairs):
    g_vocab = {"<PAD>":0, "<BOS>":1, "<EOS>":2, "<UNK>":3}
    w_vocab = {"<PAD>":0, "<BOS>":1, "<EOS>":2, "<UNK>":3}
    # 제스처 토큰
    for gseq, _ in pairs:
        for g in gseq:
            if g not in g_vocab: g_vocab[g] = len(g_vocab)
    # 텍스트 토큰(띄어쓰기 기준 토큰화)
    for _, sent in pairs:
        for w in sent.strip().split():
            if w not in w_vocab: w_vocab[w] = len(w_vocab)
    return g_vocab, w_vocab

def encode_pairs(pairs, g_vocab, w_vocab, max_g_len=32, max_w_len=32):
    G, W_in, W_out = [], [], []
    for gseq, sent in pairs:
        g_ids = [g_vocab.get(g, g_vocab["<UNK>"]) for g in gseq][:max_g_len]
        g_ids = g_ids + [g_vocab["<PAD>"]] * (max_g_len - len(g_ids))
        words = sent.strip().split()
        w_in = [w_vocab["<BOS>"]] + [w_vocab.get(w, w_vocab["<UNK>"]) for w in words]
        w_out= [w_vocab.get(w, w_vocab["<UNK>"]) for w in words] + [w_vocab["<EOS>"]]
        w_in  = (w_in + [w_vocab["<PAD>"]]*max_w_len)[:max_w_len]
        w_out = (w_out+ [w_vocab["<PAD>"]]*max_w_len)[:max_w_len]
        G.append(g_ids); W_in.append(w_in); W_out.append(w_out)
    return np.array(G), np.array(W_in), np.array(W_out)

def build_seq2seq(g_vocab_size, w_vocab_size, d_model=256):
    # 간단한 LSTM 인코더-디코더
    g_in  = layers.Input(shape=(None,), name="g_in")
    w_in  = layers.Input(shape=(None,), name="w_in")

    g_embed = layers.Embedding(g_vocab_size, d_model, mask_zero=True)(g_in)
    enc_out, state_h, state_c = layers.LSTM(d_model, return_state=True)(g_embed)

    w_embed = layers.Embedding(w_vocab_size, d_model, mask_zero=True)(w_in)
    dec_lstm = layers.LSTM(d_model, return_sequences=True)
    dec_out = dec_lstm(w_embed, initial_state=[state_h, state_c])

    # 간단한 어텐션
    enc_exp = layers.RepeatVector(tf.shape(dec_out)[1])(enc_out)  # (B,T,D)
    concat  = layers.Concatenate()([dec_out, enc_exp])
    logits  = layers.TimeDistributed(layers.Dense(w_vocab_size))(concat)

    model = models.Model([g_in, w_in], logits)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model

def greedy_decode(model, g_ids, w_vocab, inv_w_vocab, max_len=32):
    bos = w_vocab["<BOS>"]; eos = w_vocab["<EOS>"]
    w = [bos]
    for _ in range(max_len):
        logits = model.predict([g_ids[None, :], np.array(w)[None, :]], verbose=0)
        next_id = int(np.argmax(logits[0, -1]))
        if next_id == eos: break
        w.append(next_id)
    words = [inv_w_vocab.get(t, "<UNK>") for t in w[1:]]
    return " ".join(words)

# ====== 5) CLI ======
def cmd_train_gesture(args):
    X_tr, y_tr, X_va, y_va, labels = build_gesture_dataset(args.learned)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "labels.json"), "w") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    model = build_gesture_model(n_classes=len(labels))
    cb = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(args.out, "gesture_lstm.h5"),
                                           monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    ]
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
              epochs=EPOCHS, batch_size=BATCH, callbacks=cb)
    model.save(os.path.join(args.out, "gesture_lstm.h5"))
    print("✅ 제스처 인식 모델 저장 완료.")

def cmd_predict_tokens(args):
    tokens = predict_gesture_tokens(args.model, args.labels, args.seq)
    print("TOKENS:", tokens)
    if args.translate_rules:
        print("RULE_TRANSLATION:", rule_translate(tokens))

def cmd_train_translate(args):
    pairs = load_pairs(args.pairs)
    g_vocab, w_vocab = build_vocab_from_pairs(pairs)
    G, W_in, W_out = encode_pairs(pairs, g_vocab, w_vocab, max_g_len=args.max_g, max_w_len=args.max_w)
    model = build_seq2seq(len(g_vocab), len(w_vocab), d_model=256)
    model.fit([G, W_in], W_out, batch_size=32, epochs=30, validation_split=0.1)
    os.makedirs(args.out, exist_ok=True)
    model.save(os.path.join(args.out, "seq2seq.h5"))
    json.dump({"g_vocab": g_vocab, "w_vocab": w_vocab},
              open(os.path.join(args.out, "vocabs.json"), "w"), ensure_ascii=False, indent=2)
    print("✅ 번역 모델 저장 완료.")

def cmd_translate_tokens(args):
    # tokens.txt: 공백 구분 토큰들(예: Wave 17 Wave)
    tokens = open(args.tokens, "r", encoding="utf-8").read().strip().split()
    voc = json.load(open(args.vocabs, "r", encoding="utf-8"))
    g_vocab, w_vocab = voc["g_vocab"], voc["w_vocab"]
    inv_w_vocab = {v:k for k,v in w_vocab.items()}

    g_ids = [g_vocab.get(t, g_vocab["<UNK>"]) for t in tokens]
    g_ids = (g_ids + [g_vocab["<PAD>"]]*args.max_g)[:args.max_g]

    model = tf.keras.models.load_model(args.model, compile=False)
    sent = greedy_decode(model, np.array(g_ids), w_vocab, inv_w_vocab, max_len=args.max_w)
    print("SEQ2SEQ:", sent)

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    # 학습(제스처 인식)
    ap_tr = sub.add_parser("train-gesture")
    ap_tr.add_argument("--learned", default="LEARNED", help="라벨별 json이 모인 폴더")
    ap_tr.add_argument("--out", default="artifacts", help="모델/라벨 저장 폴더")
    ap_tr.set_defaults(func=cmd_train_gesture)

    # 추론(토큰 예측)
    ap_pr = sub.add_parser("predict-tokens")
    ap_pr.add_argument("--model", default="artifacts/gesture_lstm.h5")
    ap_pr.add_argument("--labels", default="artifacts/labels.json")
    ap_pr.add_argument("--seq", required=True, help="새 json 시퀀스 파일 경로")
    ap_pr.add_argument("--translate-rules", action="store_true")
    ap_pr.set_defaults(func=cmd_predict_tokens)

    # 번역 학습(seq2seq)
    ap_tt = sub.add_parser("train-translate")
    ap_tt.add_argument("--pairs", default="pairs.jsonl")
    ap_tt.add_argument("--out", default="artifacts_trans")
    ap_tt.add_argument("--max-g", type=int, default=32)
    ap_tt.add_argument("--max-w", type=int, default=32)
    ap_tt.set_defaults(func=cmd_train_translate)

    # 번역 추론(seq2seq)
    ap_it = sub.add_parser("translate-tokens")
    ap_it.add_argument("--model", default="artifacts_trans/seq2seq.h5")
    ap_it.add_argument("--vocabs", default="artifacts_trans/vocabs.json")
    ap_it.add_argument("--tokens", required=True)
    ap_it.add_argument("--max-g", type=int, default=32)
    ap_it.add_argument("--max-w", type=int, default=32)
    ap_it.set_defaults(func=cmd_translate_tokens)

    args = ap.parse_args()
    if not hasattr(args, "func"):
        ap.print_help(); return
    args.func(args)

if __name__ == "__main__":
    main()
