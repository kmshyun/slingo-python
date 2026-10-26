# -*- coding: utf-8 -*-
"""
best_lstm.pt (PyTorch BiLSTMClassifier) -> 안드로이드 앱 asset 2개로 내보내기.
  - sign_model.json : config / labels(클래스순서, NFC) / order(텐서 배치) / ref(자체검증)
  - sign_lstm.bin   : order 순서대로 이어붙인 raw float32 (little-endian)

앱 SignModel.kt 가 기대하는 포맷과 정확히 일치하도록 만든다.
  * bias 는 PyTorch 의 bias_ih + bias_hh 를 합쳐 b_l{l}[_reverse] 로 저장
  * gate 순서(i,f,g,o) 는 PyTorch=Kotlin 동일
  * head.0=LayerNorm(2H) -> ln_w/ln_b, head.2=Linear(2H,C) -> fc_w/fc_b
"""
import json, struct, unicodedata
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints" / "best_lstm.pt"
ASSETS = Path("/Users/shkim/Desktop/SLingo/app/src/main/assets")
OUT_JSON = ASSETS / "sign_model.json"
OUT_BIN = ASSETS / "sign_lstm.bin"


# 2learn.py 의 모델과 동일 구조(ref 로짓 계산용)
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=192, layers=2, dropout=0.2, bidir=True):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers,
                            dropout=dropout if layers > 1 else 0.0,
                            bidirectional=bidir, batch_first=True)
        out_dim = hidden * (2 if bidir else 1)
        self.head = nn.Sequential(nn.LayerNorm(out_dim), nn.Dropout(dropout),
                                  nn.Linear(out_dim, num_classes))

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        L = self.lstm.num_layers
        D = 2 if self.lstm.bidirectional else 1
        hn = hn.view(L, D, x.size(0), self.lstm.hidden_size)
        last = hn[-1].transpose(0, 1).contiguous().view(x.size(0), -1)
        return self.head(last)


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["model"]
    cfg = ck["config"]
    label2id = ck["label2id"]
    H = cfg["hidden"]
    inp = cfg["input_dim"]
    layers = cfg["layers"]

    # 클래스 index 순서대로 라벨 배열(NFC 정규화)
    id2label = {v: k for k, v in label2id.items()}
    labels = [unicodedata.normalize("NFC", id2label[i]) for i in range(len(id2label))]
    C = len(labels)
    print(f"labels({C}):", labels)

    # 텐서를 (이름, np.float32 flat) 순서 리스트로 구성 — bin 기록 순서 = order 순서
    tensors = []  # (name, shape, flat_array)

    def add(name, t):
        a = t.detach().cpu().numpy().astype("<f4")  # little-endian float32
        tensors.append((name, list(a.shape), a.reshape(-1)))

    for l in range(layers):
        for suf in ("", "_reverse"):
            wih = sd[f"lstm.weight_ih_l{l}{suf}"]          # (4H, inDim)
            whh = sd[f"lstm.weight_hh_l{l}{suf}"]          # (4H, H)
            b = sd[f"lstm.bias_ih_l{l}{suf}"] + sd[f"lstm.bias_hh_l{l}{suf}"]  # (4H,) 합산
            add(f"wih_l{l}{suf}", wih)
            add(f"whh_l{l}{suf}", whh)
            add(f"b_l{l}{suf}", b)
    add("ln_w", sd["head.0.weight"])   # (2H,)
    add("ln_b", sd["head.0.bias"])     # (2H,)
    add("fc_w", sd["head.2.weight"])   # (C, 2H)
    add("fc_b", sd["head.2.bias"])     # (C,)

    # bin 기록
    ASSETS.mkdir(parents=True, exist_ok=True)
    with open(OUT_BIN, "wb") as f:
        for _, _, flat in tensors:
            f.write(flat.tobytes())
    order = [{"name": n, "shape": shp} for (n, shp, _) in tensors]

    # ref(자체검증용): feats[t][d] = sin(t*0.1 + d*0.01), T=20 → PyTorch 로짓
    model = BiLSTMClassifier(inp, C, hidden=H, layers=layers,
                             dropout=cfg["dropout"], bidir=cfg["bidir"])
    model.load_state_dict(sd, strict=True)
    model.eval()
    T = 20
    feats = np.zeros((T, inp), dtype=np.float32)
    for t in range(T):
        for d in range(inp):
            feats[t, d] = np.sin(t * 0.1 + d * 0.01)
    with torch.no_grad():
        x = torch.from_numpy(feats).unsqueeze(0)  # (1,T,inp)
        logits = model(x, torch.tensor([T])).squeeze(0).numpy()

    manifest = {
        "config": {"input_dim": inp, "hidden": H, "layers": layers,
                   "dropout": cfg["dropout"], "bidir": cfg["bidir"]},
        "labels": labels,
        "order": order,
        "ref": {"T": T,
                "formula": "feats[t][d]=sin(t*0.1+d*0.01)",
                "logits": [float(v) for v in logits]},
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)

    total_floats = sum(int(np.prod(shp)) for _, shp, _ in tensors)
    print(f"wrote {OUT_BIN} ({OUT_BIN.stat().st_size} bytes, {total_floats} floats)")
    print(f"wrote {OUT_JSON} ({OUT_JSON.stat().st_size} bytes)")
    print("ref logits[:5]:", manifest["ref"]["logits"][:5])


if __name__ == "__main__":
    main()
