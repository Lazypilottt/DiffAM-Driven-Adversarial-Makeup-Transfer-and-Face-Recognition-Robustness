#!/usr/bin/env bash
# Fetch the upstream DiffAM code at the commit used for the experiments.
# Usage: scripts/setup_diffam.sh [target_dir]
set -euo pipefail

DIFFAM_REPO="https://github.com/HansSunY/DiffAM.git"
DIFFAM_COMMIT="323aef8c4f7834ff098f97b6457f17925e9f7e43"
TARGET="${1:-third_party/DiffAM}"

if [ ! -d "$TARGET/.git" ]; then
  git clone "$DIFFAM_REPO" "$TARGET"
fi

if ! git -C "$TARGET" cat-file -e "${DIFFAM_COMMIT}^{commit}" 2>/dev/null; then
  git -C "$TARGET" fetch origin
fi
git -C "$TARGET" checkout --quiet "$DIFFAM_COMMIT"

cat <<MSG
DiffAM checked out at $TARGET (${DIFFAM_COMMIT:0:7}).

Remaining manual steps (files are hosted on Google Drive by the DiffAM authors):
  1. pretrained/  celeba_hq.ckpt, makeup.pt, model_ir_se50.pth,
                  shape_predictor_68_face_landmarks.dat
  2. assets/      target FR models, MT dataset, makeup pairs (assets.zip)
  3. assets/datasets/CelebAMask-HQ
Links are in $TARGET/README.md. Then:
  pip install -r $TARGET/requirements.txt
  pip install git+https://github.com/openai/CLIP.git
MSG
