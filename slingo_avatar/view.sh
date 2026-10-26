#!/bin/bash
# SLingo 수어 아바타 실시간 뷰어.
# 사용: bash view.sh <단어> [--depth 0.8 --reach 0.85 --mirror 1 ...]
# 예:   bash view.sh 사랑
#       bash view.sh 안녕 --mirror 0        # 좌우반전 꺼서 비교
WORD="${1:-안녕}"; shift 2>/dev/null
BL="/Applications/Blender.app/Contents/MacOS/Blender"
DIR="/Users/shkim/Desktop/2026 전국전람회/code/slingo_avatar"
VRM="$DIR/avatar/AvatarSample_B.vrm"
echo "[$WORD] 씬 만드는 중..."
"$BL" --background --python "$DIR/tools/bl_render_sign.py" -- \
    --blend "$DIR/preview/$WORD.blend" --vrm "$VRM" \
    --json "$DIR/motion_json/$WORD.json" --depth 0.8 --reach 0.85 "$@" \
    2>&1 | grep -E "blend saved|Error:|Traceback" | tail -2
pkill -x Blender 2>/dev/null; sleep 1
"$BL" "$DIR/preview/$WORD.blend" >/dev/null 2>&1 &
echo "✅ [$WORD] 창 열림 — Space=재생/정지, 아래 타임라인 드래그=한프레임씩, 마우스휠=줌"
