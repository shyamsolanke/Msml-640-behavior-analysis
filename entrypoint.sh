#!/usr/bin/env bash
# entrypoint.sh — download missing assets then launch Streamlit
set -euo pipefail

echo "============================================"
echo " Behavior Analysis — startup"
echo "============================================"

# ---------------------------------------------------------------------------
# 1. YOLOv8m weights
# ---------------------------------------------------------------------------
YOLO_WEIGHTS="weights/yolov8m.pt"
if [[ ! -f "$YOLO_WEIGHTS" ]]; then
    echo "[weights] Downloading yolov8m.pt (~50 MB)..."
    python - <<'EOF'
from ultralytics import YOLO
import shutil, os
m = YOLO("yolov8m.pt")          # downloads to ~/.ultralytics cache
cache = m.ckpt_path
os.makedirs("weights", exist_ok=True)
shutil.copy(cache, "weights/yolov8m.pt")
print(f"[weights] Saved to weights/yolov8m.pt")
EOF
else
    echo "[weights] yolov8m.pt already present."
fi

# ---------------------------------------------------------------------------
# 2. MobileSAM weights
# ---------------------------------------------------------------------------
SAM_WEIGHTS="weights/mobile_sam.pt"
if [[ ! -f "$SAM_WEIGHTS" ]]; then
    echo "[weights] Downloading mobile_sam.pt (~39 MB)..."
    python - <<'EOF'
from core.segmenter import _ensure_weights
_ensure_weights("weights/mobile_sam.pt")
EOF
else
    echo "[weights] mobile_sam.pt already present."
fi

# ---------------------------------------------------------------------------
# 3. Demo clip
# ---------------------------------------------------------------------------
DEMO_CLIP="data/demo_clip/source.mp4"
DEMO_URL="https://assets.mixkit.co/videos/4437/4437-720.mp4"
if [[ ! -f "$DEMO_CLIP" ]]; then
    echo "[data] Downloading demo clip (~12 MB)..."
    mkdir -p data/demo_clip
    curl -L --fail --progress-bar -o "$DEMO_CLIP" "$DEMO_URL"
    echo "[data] Demo clip saved to $DEMO_CLIP"
else
    echo "[data] Demo clip already present."
fi

echo "============================================"
echo " Launching Streamlit on port 8501 ..."
echo "============================================"

exec streamlit run app.py
