#!/usr/bin/env bash
# Full automated setup for Wybe Voice NO on a RunPod GPU pod.
# Usage: bash scripts/runpod_setup.sh [--port PORT]
#
# Prerequisites:
#   - RunPod GPU pod (RTX A5000 24GB recommended)
#   - HF_TOKEN environment variable set
#
# This script is idempotent — safe to re-run after a pod restart.
# Auto-detects RunPod persistent volume at /runpod for cache & repo storage.

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────

REPO_URL="https://github.com/Usoramara/wybe-voice-no.git"
PORT=8080

# ─── Auto-detect RunPod volume ───────────────────────────────────────────────

if [[ -d "/runpod" ]]; then
    VOLUME_DIR="/runpod"
else
    VOLUME_DIR="/root"
fi
REPO_DIR="$VOLUME_DIR/wybe-voice-no"
CACHE_DIR="$VOLUME_DIR/huggingface_cache"

# ─── Usage ───────────────────────────────────────────────────────────────────

usage() {
    cat <<EOF
Wybe Voice NO — RunPod Setup

Usage: bash $0 [OPTIONS]

Options:
  --port PORT       Server port (default: 8080)
  --help            Show this help message

Environment:
  HF_TOKEN          Required. Hugging Face token.
                    (auto-sourced from container env if set via RunPod UI)

Volume:
  If /runpod exists, repo and model cache are stored there (persists across
  pod termination). Otherwise falls back to /root.
EOF
    exit 0
}

# ─── Parse args ──────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)  PORT="$2"; shift 2 ;;
        --help)  usage ;;
        *)       echo "Unknown option: $1"; usage ;;
    esac
done

# ─── Helpers ─────────────────────────────────────────────────────────────────

info()  { echo -e "\033[1;34m[INFO]\033[0m $*"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

# ─── Step 1: Validate HF_TOKEN (auto-source from container env) ─────────────

if [[ -z "${HF_TOKEN:-}" ]] && [[ -f /proc/1/environ ]]; then
    HF_TOKEN="$(tr '\0' '\n' < /proc/1/environ | grep '^HF_TOKEN=' | cut -d= -f2- || true)"
    export HF_TOKEN
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    error "HF_TOKEN is not set."
    echo "  Get a token at https://huggingface.co/settings/tokens"
    echo "  Then: export HF_TOKEN=hf_..."
    exit 1
fi
info "HF_TOKEN is set."

# ─── Step 2: Install system dependencies ─────────────────────────────────────

if command -v ffmpeg &>/dev/null; then
    info "System deps already installed (ffmpeg found)."
else
    info "Installing system dependencies..."
    apt-get update -qq
    apt-get install -y --no-install-recommends ffmpeg
    info "System dependencies installed."
fi

# ─── Step 3: Clone the repo ─────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/../server/app.py" ]]; then
    REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
    info "Running from inside the repo at $REPO_DIR — skipping clone."
elif [[ -d "$REPO_DIR/server" ]]; then
    info "Repo already cloned at $REPO_DIR — pulling latest..."
    git -C "$REPO_DIR" pull --ff-only || true
else
    info "Cloning Wybe Voice NO to $REPO_DIR..."
    git clone "$REPO_URL" "$REPO_DIR"
    info "Clone complete."
fi

# ─── Step 4: Install Python package ─────────────────────────────────────────

info "Installing wybe-voice-no Python package..."
pip install --quiet "$REPO_DIR/."
info "Python package installed."

# Install chatterbox-tts separately (large dependency)
info "Installing chatterbox-tts..."
pip install --quiet chatterbox-tts
info "chatterbox-tts installed."

# ─── Step 5: Pre-download model weights ─────────────────────────────────────

export HF_HOME="$CACHE_DIR"

info "Pre-downloading model weights..."
info "  Cache directory: $CACHE_DIR"
info "  (This may take several minutes on first run — skips if already cached)"

# ASR model
python -c "
from huggingface_hub import snapshot_download
import os
token = os.environ.get('HF_TOKEN', '')
print('[INFO] Downloading NB-Whisper ASR model...')
snapshot_download('NbAiLab/nb-whisper-large-distil-turbo-beta', token=token)
print('[INFO] Downloading NorMistral LLM (GGUF)...')
from huggingface_hub import hf_hub_download
hf_hub_download('norallm/normistral-7b-warm-instruct', filename='normistral-7b-warm-instruct.Q4_K_M.gguf', token=token)
print('[INFO] Models cached.')
"
info "Model weights cached."

# ─── Step 6: Launch the server ───────────────────────────────────────────────

export HF_HOME="$CACHE_DIR"

echo ""
info "========================================="
info " Starting Wybe Voice NO"
info "========================================="
info "  Port: $PORT"
info "  Volume: $VOLUME_DIR"
info "  Cache: $CACHE_DIR"
echo ""
info "Access via RunPod proxy:"
info "  https://<POD_ID>-${PORT}.proxy.runpod.net/"
echo ""

cd "$REPO_DIR"
exec python -m server
