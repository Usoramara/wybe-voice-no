#!/usr/bin/env bash
# Minimal restart script for Wybe Voice NO on RunPod.
# Use this after runpod_setup.sh has already run (models are cached).
#
# Auto-detects RunPod persistent volume at /runpod for cached models.
# Auto-sources HF_TOKEN from container environment if set via RunPod UI.

set -euo pipefail

PORT=8080

# ─── Auto-detect RunPod volume ───────────────────────────────────────────────

if [[ -d "/runpod/huggingface_cache" ]]; then
    CACHE_DIR="/runpod/huggingface_cache"
elif [[ -d "/runpod" ]]; then
    CACHE_DIR="/runpod/huggingface_cache"
else
    CACHE_DIR="/root/.cache"
fi

if [[ -d "/runpod/wybe-voice-no" ]]; then
    REPO_DIR="/runpod/wybe-voice-no"
else
    REPO_DIR="/root/wybe-voice-no"
fi

# ─── Parse args ──────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --help) echo "Usage: bash $0 [--port PORT]"; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ─── Helpers ─────────────────────────────────────────────────────────────────

info()  { echo -e "\033[1;34m[INFO]\033[0m $*"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

# ─── Validate HF_TOKEN ──────────────────────────────────────────────────────

if [[ -z "${HF_TOKEN:-}" ]] && [[ -f /proc/1/environ ]]; then
    HF_TOKEN="$(tr '\0' '\n' < /proc/1/environ | grep '^HF_TOKEN=' | cut -d= -f2- || true)"
    export HF_TOKEN
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    error "HF_TOKEN is not set."
    exit 1
fi

export HF_HOME="$CACHE_DIR"

info "Starting Wybe Voice NO on port $PORT..."
info "  Cache: $CACHE_DIR"
info "  Repo: $REPO_DIR"

cd "$REPO_DIR"
exec python -m server
