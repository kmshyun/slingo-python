# sign_lstm_train_test.py
# - Dataset:
#     train/val: dataset/<label>/clip_XXXX.npy
#     test: (선택) dataset_test/<label>/clip_XXXX.npy  가 있으면 그걸 사용.
#           없으면 dataset에서 라벨 보존 분할로 test 추출.
# - Feature dim = 139 (코드 작성자 제공 저장 포맷 기준)
# - Model: BiLSTM 분류기

import os, json, math, random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader

# sklearn은 선택(없어도 동작)
try:
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# -------------------------
# Config
# -------------------------
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "dataset"
TEST_ROOT = ROOT / "dataset_test"  # 있으면 사용
BATCH_SIZE = 16
NUM_EPOCHS = 30
LR = 1e-3
HIDDEN = 192
LAYERS = 2
DROPOUT = 0.2
BIDIR = True
VAL_RATIO = 0.15
TEST_RATIO = 0.15  # TEST_ROOT 없을 때만 사용
SEED = 42

SAVE_DIR = ROOT / "checkpoints"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------
# Data utils
# -------------------------
def list_label_dirs(root: Path):
    return sorted([p for p in root.iterdir() if p.is_dir()])

def list_clips(root: Path):
    """return [(path,label_name)]"""
    items = []
    for ldir in list_label_dirs(root):
        for f in sorted(ldir.glob("clip_*.npy")):
            items.append((f, ldir.name))
    return items

def stratified_split(items, val_ratio, test_ratio, seed=42):
    """
    items: list of (path, label_name)
    라벨별로 셔플 후 val/test를 분할. test_ratio가 0이면 test는 비움.
    """
    random.Random(seed).shuffle(items)
    by_label = {}
    for p, lb in items:
        by_label.setdefault(lb, []).append(p)

    train, val, test = [], [], []
    for lb, paths in by_label.items():
        rnd = random.Random(seed + hash(lb) % (10**6))
        rnd.shuffle(paths)
        n = len(paths)
        n_test = int(round(n * test_ratio))
        n_val  = int(round(n * val_ratio))
        # 누수가 없도록 순서: test, val, train
        test_paths = paths[:n_test]
        val_paths  = paths[n_test:n_test+n_val]
        train_paths= paths[n_test+n_val:]

        train += [(p, lb) for p in train_paths]
        val   += [(p, lb) for p in val_paths]
        test  += [(p, lb) for p in test_paths]
    return train, val, test

def load_pack(path: Path):
    return np.load(path, allow_pickle=True).item()

def build_frame_features(pack) -> np.ndarray:
    right = pack["right"]          # (N,21,3)
    left  = pack["left"]           # (N,21,3)
    mask  = pack["hand_mask"]      # (N,2)
    bodym = pack.get("body_mask")  # (N,)
    ms    = pack.get("ms")         # (N,3)
    sw    = pack.get("sw")         # (N,)
    rrel  = pack.get("right_rel_ms")  # (N,3)
    lrel  = pack.get("left_rel_ms")   # (N,3)

    N = right.shape[0]
    feat = np.concatenate([
        right.reshape(N, -1).astype(np.float32),   # 63
        left.reshape(N, -1).astype(np.float32),    # 63
        mask.astype(np.float32),                   # 2
        bodym.astype(np.float32).reshape(N,1),     # 1
        ms.astype(np.float32),                     # 3
        sw.astype(np.float32).reshape(N,1),        # 1
        rrel.astype(np.float32),                   # 3
        lrel.astype(np.float32)                    # 3
    ], axis=1)
    np.nan_to_num(feat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return feat  # (T,139)

# -------------------------
# Dataset
# -------------------------
class SignSeqDataset(Dataset):
    def __init__(self, items, label2id):
        """
        items: list[(Path, label_name)]
        label2id: dict
        """
        self.items = items
        self.label2id = label2id

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, lb = self.items[idx]
        y = self.label2id[lb]
        pack = load_pack(path)
        x = build_frame_features(pack)  # (T, 139)
        T = x.shape[0]
        x = torch.from_numpy(x)         # float32
        y = torch.tensor(y, dtype=torch.long)
        length = torch.tensor(T, dtype=torch.long)
        return x, y, length, str(path)

def make_label_map(all_items):
    labels = sorted(list({lb for _, lb in all_items}))
    return {lb: i for i, lb in enumerate(labels)}

def collate_batch(batch):
    xs, ys, lens, paths = zip(*batch)
    # 정렬(길이 내림차순)
    order = sorted(range(len(xs)), key=lambda i: lens[i].item(), reverse=True)
    xs   = [xs[i] for i in order]
    ys   = torch.stack([ys[i] for i in order], dim=0)
    lens = torch.stack([lens[i] for i in order], dim=0)
    paths= [paths[i] for i in order]

    padded = pad_sequence(xs, batch_first=True, padding_value=0.0)
    B, Tmax, _ = padded.shape
    mask = torch.zeros((B, Tmax), dtype=torch.bool)
    for i, L in enumerate(lens):
        mask[i, :L] = True
    return padded, ys, lens, mask, paths

# -------------------------
# Model
# -------------------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int,
                 hidden: int = 192, layers: int = 2, dropout: float = 0.2, bidir: bool = True):
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
            nn.Linear(out_dim, num_classes)
        )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)  # hn: (L*D, B, H)
        L = self.lstm.num_layers
        D = 2 if self.lstm.bidirectional else 1
        hn = hn.view(L, D, x.size(0), self.lstm.hidden_size)
        last = hn[-1].transpose(0,1).contiguous().view(x.size(0), -1)  # (B, D*H)
        logits = self.head(last)
        return logits

def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()

# -------------------------
# Train / Eval
# -------------------------
def evaluate(model, loader, device, label2id, title="eval"):
    model.eval()
    crit = nn.CrossEntropyLoss()
    tot_loss = 0.0
    tot_acc = 0.0
    n_batch = 0

    all_pred, all_true, all_paths = [], [], []

    with torch.no_grad():
        for x, y, lengths, mask, paths in loader:
            x = x.to(device); y = y.to(device); lengths = lengths.to(device)
            logits = model(x, lengths)
            loss = crit(logits, y)
            tot_loss += loss.item()
            tot_acc  += accuracy(logits, y)
            n_batch += 1

            all_pred += logits.argmax(dim=1).cpu().tolist()
            all_true += y.cpu().tolist()
            all_paths+= list(paths)

    loss = tot_loss / max(1, n_batch)
    acc  = tot_acc / max(1, n_batch)
    print(f"[{title}] loss {loss:.4f} acc {acc:.3f}")

    # 리포트/혼동행렬(선택)
    if SKLEARN_OK:
        id2label = {i: lb for lb, i in label2id.items()}
        target_names = [id2label[i] for i in range(len(id2label))]
        print(classification_report(all_true, all_pred, target_names=target_names, digits=3))
        cm = confusion_matrix(all_true, all_pred, labels=list(range(len(id2label))))
        print("Confusion Matrix:\n", cm)

    return acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- 데이터 로딩 / 분할 --------
    base_items = list_clips(DATA_ROOT)  # [(path,label)]
    if len(base_items) == 0:
        raise RuntimeError(f"No clips under {DATA_ROOT}")

    # 외부 테스트셋 존재 여부
    test_items_ext = list_clips(TEST_ROOT) if TEST_ROOT.exists() else []
    use_external_test = len(test_items_ext) > 0

    # 라벨 맵(모든 라벨을 union으로): train 영역 라벨과 test 라벨이 동일하도록
    all_items_for_labels = base_items + test_items_ext
    label2id = make_label_map(all_items_for_labels)
    with open(SAVE_DIR / "label2id.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    print("Label map:", label2id)

    # 내부 분할 또는 외부 테스트셋 사용
    if use_external_test:
        # train/val은 base에서만 분할, test는 외부 폴더 고정
        train_items, val_items, _ = stratified_split(base_items, VAL_RATIO, 0.0, seed=SEED)
        test_items = test_items_ext
        print(f"Use external test set at {TEST_ROOT}")
    else:
        # 라벨 보존 분할로 train/val/test 생성
        train_items, val_items, test_items = stratified_split(base_items, VAL_RATIO, TEST_RATIO, seed=SEED)

    print(f"train={len(train_items)}  val={len(val_items)}  test={len(test_items)}")

    train_ds = SignSeqDataset(train_items, label2id)
    val_ds   = SignSeqDataset(val_items,   label2id)
    test_ds  = SignSeqDataset(test_items,  label2id)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_batch, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_batch, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_batch, num_workers=0, pin_memory=True)

    # -------- 모델/학습 --------
    model = BiLSTMClassifier(input_dim=139, num_classes=len(label2id),
                             hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT, bidir=BIDIR).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=NUM_EPOCHS)
    crit  = nn.CrossEntropyLoss()

    best_val = -1.0
    best_path = SAVE_DIR / "best_lstm.pt"

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        tr_loss = 0.0; tr_acc = 0.0; n_batch = 0
        for x, y, lengths, mask, _ in train_loader:
            x = x.to(device); y = y.to(device); lengths = lengths.to(device)
            optim.zero_grad()
            logits = model(x, lengths)
            loss = crit(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tr_loss += loss.item()
            tr_acc  += accuracy(logits.detach(), y)
            n_batch += 1
        sched.step()
        tr_loss /= max(1, n_batch); tr_acc /= max(1, n_batch)

        # val
        val_acc = evaluate(model, val_loader, device, label2id, title=f"val@{epoch:02d}")

        print(f"[{epoch:02d}/{NUM_EPOCHS}] train loss {tr_loss:.4f} acc {tr_acc:.3f} | val acc {val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "model": model.state_dict(),
                "label2id": label2id,
                "config": {"input_dim":139,"hidden":HIDDEN,"layers":LAYERS,"dropout":DROPOUT,"bidir":BIDIR}
            }, best_path)
            print(f"  -> saved best to {best_path} (val acc={best_val:.3f})")

    # -------- 테스트 평가(최고모델) --------
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print("\n[TEST] evaluating best checkpoint ...")
    else:
        print("\n[TEST] best checkpoint not found, using last weights ...")

    _ = evaluate(model, test_loader, device, label2id, title="test")

if __name__ == "__main__":
    main()
