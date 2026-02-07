#!/bin/bash
# Auto-start Wybe Voice NO on RunPod pod boot.
# Called by RunPod's /start.sh as /post_start.sh.
# Models and repo live on the persistent volume at /runpod.

LOG="/var/log/wybe-voice-no.log"

echo "[Wybe Voice NO] post_start.sh triggered at $(date)" | tee "$LOG"

# Clone repo to volume if not present
if [[ ! -d /runpod/wybe-voice-no/server ]]; then
    echo "[Wybe Voice NO] Cloning repo to /runpod/wybe-voice-no..." | tee -a "$LOG"
    git clone https://github.com/Usoramara/wybe-voice-no.git /runpod/wybe-voice-no >> "$LOG" 2>&1
fi

# Install deps + launch server (idempotent, uses cached models from volume)
bash /runpod/wybe-voice-no/scripts/runpod_setup.sh >> "$LOG" 2>&1 &

echo "[Wybe Voice NO] Server launching in background. Logs: tail -f $LOG"
