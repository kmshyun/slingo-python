# -*- coding: utf-8 -*-
"""
검증 스크립트: 저장된 clip .npy 를 2learn.py 와 '똑같은 방식'으로 139차원 특징으로
복원해서, 형태(T,139)와 통계가 정상인지 확인한다.

앱 SignFeatureExtractor.kt 의 특징 순서:
  right(63) + left(63) + hand_mask(2) + has_body(1) + ms(3) + sw(1)
  + right_rel_ms(3) + left_rel_ms(3) = 139

사용:  python verify_parity.py dataset/안녕/clip_0001.npy
       python verify_parity.py dataset            # 폴더 전체 스캔
"""
import sys
from pathlib import Path
import numpy as np


def build_frame_features(pack) -> np.ndarray:
    right = pack["right"]          # (N,21,3)
    left = pack["left"]            # (N,21,3)
    mask = pack["hand_mask"]       # (N,2)
    bodym = pack["body_mask"]      # (N,)
    ms = pack["ms"]                # (N,3)
    sw = pack["sw"]                # (N,)
    rrel = pack["right_rel_ms"]    # (N,3)
    lrel = pack["left_rel_ms"]     # (N,3)
    N = right.shape[0]
    feat = np.concatenate([
        right.reshape(N, -1).astype(np.float32),
        left.reshape(N, -1).astype(np.float32),
        mask.astype(np.float32),
        bodym.astype(np.float32).reshape(N, 1),
        ms.astype(np.float32),
        sw.astype(np.float32).reshape(N, 1),
        rrel.astype(np.float32),
        lrel.astype(np.float32),
    ], axis=1)
    np.nan_to_num(feat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return feat


def check_one(path: Path):
    pack = np.load(path, allow_pickle=True).item()
    feat = build_frame_features(pack)
    ok = feat.shape[1] == 139
    nan = np.isnan(feat).any()
    print(f"{'OK ' if ok and not nan else 'BAD'} {path}  shape={feat.shape}  "
          f"min={feat.min():.3f} max={feat.max():.3f} "
          f"nan={bool(nan)} frames={feat.shape[0]}")
    return ok and not nan


def main():
    if len(sys.argv) < 2:
        print("사용: python verify_parity.py <clip.npy | dataset폴더>")
        return
    target = Path(sys.argv[1])
    if target.is_file():
        check_one(target)
    else:
        files = sorted(target.rglob("clip_*.npy"))
        if not files:
            print(f"clip 파일 없음: {target}")
            return
        good = sum(check_one(f) for f in files)
        print(f"\n총 {len(files)}개 중 {good}개 정상 (139차원)")


if __name__ == "__main__":
    main()
