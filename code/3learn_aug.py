# -*- coding: utf-8 -*-
"""
2learn.py 와 같은 모델/특징을 쓰되, 학습 때 augment.py 의 증강을 적용한다.

왜 필요한가:
  dataset 657개 클립이 전부 하루 한 세션 녹화라 같은 클래스의 클립끼리 거의
  복사본이다. 정직한 5-fold CV 가 657/657 (100%) 이 나오는데, 이건 모델이 잘
  배웠다는 뜻이 아니라 허용 범위가 극단적으로 좁다는 뜻이다. 실행할 때 조금만
  다르게 하면 떨어진다. 증강은 재녹화 없이 그 범위를 넓힌다.

출력: checkpoints/best_lstm_aug.pt  (final.py 가 읽는 형식과 동일)
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from augment import augment

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "dataset"
SAVE_DIR = ROOT / "checkpoints"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
NUM_EPOCHS = 60          # 증강이 있으면 더 오래 학습해야 한다
LR = 1e-3
HIDDEN, LAYERS, DROPOUT, BIDIR = 192, 2, 0.2, True
VAL_RATIO = 0.15
SEED = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def build_frame_features(pack) -> np.ndarray:
    N = pack["right"].shape[0]
    feat = np.concatenate([
        pack["right"].reshape(N, -1).astype(np.float32),
        pack["left"].reshape(N, -1).astype(np.float32),
        pack["hand_mask"].astype(np.float32),
        pack["body_mask"].astype(np.float32).reshape(N, 1),
        pack["ms"].astype(np.float32),
        pack["sw"].astype(np.float32).reshape(N, 1),
        pack["right_rel_ms"].astype(np.float32),
        pack["left_rel_ms"].astype(np.float32),
    ], axis=1)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=192, layers=2, dropout=0.2, bidir=True):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers,
                            dropout=dropout if layers > 1 else 0.0,
                            bidirectional=bidir, batch_first=True)
        out_dim = hidden * (2 if bidir else 1)
        self.head = nn.Sequential(nn.LayerNorm(out_dim), nn.Dropout(dropout),
                                  nn.Linear(out_dim, num_classes))

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        D = 2 if self.lstm.bidirectional else 1
        hn = hn.view(self.lstm.num_layers, D, x.size(0), self.lstm.hidden_size)
        return self.head(hn[-1].transpose(0, 1).contiguous().view(x.size(0), -1))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clips = []
    for ldir in sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()]):
        for f in sorted(ldir.glob("clip_*.npy")):
            clips.append((build_frame_features(np.load(f, allow_pickle=True).item()), ldir.name))
    if not clips:
        raise RuntimeError(f"클립 없음: {DATA_ROOT}")

    labels = sorted({lb for _, lb in clips})
    label2id = {lb: i for i, lb in enumerate(labels)}
    with open(SAVE_DIR / "label2id.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    # 라벨 보존 train/val 분할
    by = {}
    for i, (_, lb) in enumerate(clips):
        by.setdefault(lb, []).append(i)
    train_idx, val_idx = [], []
    for lb, idxs in by.items():
        rnd = random.Random(SEED + hash(lb) % (10 ** 6))
        rnd.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * VAL_RATIO)))
        val_idx += idxs[:n_val]
        train_idx += idxs[n_val:]
    print(f"train={len(train_idx)}  val={len(val_idx)}  classes={len(labels)}")

    model = BiLSTMClassifier(139, len(labels), HIDDEN, LAYERS, DROPOUT, BIDIR).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=NUM_EPOCHS)
    crit = nn.CrossEntropyLoss()
    arng = np.random.default_rng(SEED)

    best_val, best_path = -1.0, SAVE_DIR / "best_lstm_aug.pt"
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        random.shuffle(train_idx)
        tr_loss = n_b = 0
        for b in range(0, len(train_idx), BATCH_SIZE):
            bi = train_idx[b:b + BATCH_SIZE]
            xs = [torch.from_numpy(np.ascontiguousarray(augment(clips[i][0], arng))) for i in bi]
            lens = torch.tensor([len(x) for x in xs])
            y = torch.tensor([label2id[clips[i][1]] for i in bi], device=device)
            x = pad_sequence(xs, batch_first=True).to(device)
            optim.zero_grad()
            loss = crit(model(x, lens), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tr_loss += loss.item(); n_b += 1
        sched.step()

        model.eval(); ok = 0
        with torch.no_grad():
            for i in val_idx:
                xx = torch.from_numpy(clips[i][0])[None].to(device)
                p = int(model(xx, torch.tensor([xx.shape[1]])).argmax(1))
                ok += int(p == label2id[clips[i][1]])
        val_acc = ok / len(val_idx)
        print(f"[{epoch:02d}/{NUM_EPOCHS}] train loss {tr_loss/max(1,n_b):.4f} | val acc {val_acc:.3f}")

        if val_acc >= best_val:
            best_val = val_acc
            torch.save({"model": model.state_dict(), "label2id": label2id,
                        "config": {"input_dim": 139, "hidden": HIDDEN, "layers": LAYERS,
                                   "dropout": DROPOUT, "bidir": BIDIR}}, best_path)
    print(f"\n저장: {best_path}  (val acc={best_val:.3f})")
    print("final.py 에서 쓰려면 checkpoints/best_lstm.pt 를 백업하고 이 파일로 교체하세요.")


if __name__ == "__main__":
    main()
