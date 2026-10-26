# -*- coding: utf-8 -*-
"""
139차원 시퀀스 특징에 대한 증강.

전제: dataset 657개 클립이 전부 하루 한 세션 녹화라, 같은 클래스의 클립끼리
거의 복사본이다(정직한 5-fold CV 가 657/657). 그래서 모델의 허용 범위가
극단적으로 좁고, 실행 시 조금만 다르게 해도 떨어진다.
여기서는 '실행할 때 실제로 벌어지는 변형'만 골라서 학습 때 미리 보여준다.

특징 레이아웃 (2learn.py / final.py 와 동일):
  right 0:63 | left 63:126 | hand_mask 126:128 | body 128
  ms 129:132 | sw 132 | right_rel_ms 133:136 | left_rel_ms 136:139
"""
import numpy as np

R_SL = slice(0, 63)
L_SL = slice(63, 126)
MASK_SL = slice(126, 128)
BODY_I = 128
MS_SL = slice(129, 132)
SW_I = 132
RREL_SL = slice(133, 136)
LREL_SL = slice(136, 139)


def _resample(x, T_new):
    """시퀀스를 T_new 길이로 선형 보간 (동작 속도 변화)."""
    T = len(x)
    if T_new < 2 or T < 2:
        return x.copy()
    idx = np.linspace(0, T - 1, T_new)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (idx - lo)[:, None].astype(np.float32)
    out = x[lo] * (1 - w) + x[hi] * w
    # mask/body 는 보간하면 안 되므로 최근접으로 되돌린다
    near = np.rint(idx).astype(int)
    out[:, MASK_SL] = x[near][:, MASK_SL]
    out[:, BODY_I] = x[near][:, BODY_I]
    return out.astype(np.float32)


def _rot_z(hand63, deg):
    """21x3 정규화 손 좌표를 z축 기준으로 회전 (손목 각도 차이)."""
    h = hand63.reshape(-1, 21, 3).copy()
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    x, y = h[..., 0].copy(), h[..., 1].copy()
    h[..., 0] = c * x - s * y
    h[..., 1] = s * x + c * y
    return h.reshape(len(hand63), 63)


def augment(x, rng):
    """(T,139) -> 증강된 (T',139)."""
    x = x.copy().astype(np.float32)

    # 1) 동작 속도 (0.7~1.4배)
    if rng.random() < 0.8:
        s = rng.uniform(0.7, 1.4)
        x = _resample(x, max(6, int(round(len(x) * s))))

    # 2) 프레임 누락 (검출 실패 / fps 흔들림)
    if rng.random() < 0.5 and len(x) > 8:
        keep = rng.random(len(x)) > rng.uniform(0.03, 0.15)
        if keep.sum() >= 6:
            x = x[keep]

    # 3) 앞뒤 잘림 (링버퍼가 동작을 부분만 담는 상황)
    if rng.random() < 0.4 and len(x) > 8:
        a = rng.integers(0, 3)
        b = rng.integers(0, 3)
        if len(x) - a - b >= 6:
            x = x[a:len(x) - b if b else None]

    # 4) 한 손 검출 실패 (측정치: 클래스에 따라 8~34% 프레임에서 발생)
    if rng.random() < 0.3 and len(x) >= 5:
        n = rng.integers(1, 4)
        st = rng.integers(0, len(x) - n + 1)
        if rng.random() < 0.5:
            x[st:st + n, R_SL] = 0.0
            x[st:st + n, 126] = 0.0
            x[st:st + n, RREL_SL] = 0.0
        else:
            x[st:st + n, L_SL] = 0.0
            x[st:st + n, 127] = 0.0
            x[st:st + n, LREL_SL] = 0.0

    # 5) 손 모양 회전/스케일 (손목 각도, 손 크기 개인차)
    if rng.random() < 0.7:
        deg = rng.uniform(-12, 12)
        sc = rng.uniform(0.92, 1.08)
        x[:, R_SL] = _rot_z(x[:, R_SL], deg) * sc
        x[:, L_SL] = _rot_z(x[:, L_SL], rng.uniform(-12, 12)) * rng.uniform(0.92, 1.08)

    # 6) 랜드마크 지터 (MediaPipe 검출 노이즈)
    if rng.random() < 0.8:
        x[:, R_SL] += rng.normal(0, 0.025, (len(x), 63)).astype(np.float32)
        x[:, L_SL] += rng.normal(0, 0.025, (len(x), 63)).astype(np.float32)
        # z 는 MediaPipe 에서 특히 불안정하므로 노이즈를 더 준다
        zi = [i for i in range(63) if i % 3 == 2]
        x[:, zi] += rng.normal(0, 0.05, (len(x), 21)).astype(np.float32)
        x[:, 63 + np.array(zi)] += rng.normal(0, 0.05, (len(x), 21)).astype(np.float32)

    # 7) 몸 기준 위치/거리 (카메라 거리, 서 있는 위치)
    if rng.random() < 0.7:
        x[:, RREL_SL] += rng.normal(0, 0.04, (len(x), 3)).astype(np.float32)
        x[:, LREL_SL] += rng.normal(0, 0.04, (len(x), 3)).astype(np.float32)
        x[:, MS_SL] += rng.normal(0, 0.02, (1, 3)).astype(np.float32)
        x[:, SW_I] *= rng.uniform(0.9, 1.1)

    # 검출 안 된 손은 0 이어야 한다 (지터가 0을 깨뜨리지 않도록 복원)
    off_r = x[:, 126] < 0.5
    off_l = x[:, 127] < 0.5
    x[off_r, R_SL] = 0.0
    x[off_r, RREL_SL] = 0.0
    x[off_l, L_SL] = 0.0
    x[off_l, LREL_SL] = 0.0

    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
