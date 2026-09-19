#!/usr/bin/env bash
# ============================================================================
#  install_autostart.sh — make the animal-deterrent dashboard start on boot
# ============================================================================
#  Installs a systemd service that:
#    * starts automatically when the Pi powers on (no login, no monitor)
#    * waits for the network so the dashboard is reachable immediately
#    * restarts itself if the script crashes or the camera is unplugged
#    * logs everything to the journal so you can debug a headless box
#
#  Run it ON THE PI:
#      chmod +x install_autostart.sh
#      ./install_autostart.sh
#
#  It auto-detects the venv, the script and the model; override if needed:
#      VENV=~/Downloads/.venv_tflite MODEL=~/deterrent/yolo26n.onnx ./install_autostart.sh
#
#  Extra flags for the detector (e.g. run without audio while testing):
#      EXTRA_ARGS="--no-audio" ./install_autostart.sh
# ============================================================================
set -euo pipefail

SERVICE_NAME="animal-dashboard"
PORT="${PORT:-8000}"
# Detection defaults that were tuned on the bench. Override with e.g.
#   CONF=0.30 VOTE=9 ./install_autostart.sh
CONF="${CONF:-0.35}"
VOTE="${VOTE:-7}"
# Capture size. The USB camera offers MJPG 640x480@30 natively; anything larger
# just costs CPU because inference letterboxes down to 320px anyway.
CAM_W="${CAM_W:-640}"
CAM_H="${CAM_H:-480}"
# Audio output device index (PortAudio numbering, NOT the ALSA card number).
# Leave empty to let the script auto-detect the I2S DAC. Find it with:
#   python3 rpi5_dashboard.py --list-audio
AUDIO_DEVICE="${AUDIO_DEVICE:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] && die "Do NOT run this with sudo. Run it as your normal user; it will ask for sudo when needed."

RUN_USER="$(id -un)"
RUN_GROUP="$(id -gn)"

# --------------------------------------------------------------------------- #
#  1. Locate the dashboard script                                             #
# --------------------------------------------------------------------------- #
say "Locating rpi5_dashboard.py ..."
if [[ -n "${SCRIPT:-}" ]]; then
    DASH="$(readlink -f "$SCRIPT")"
else
    HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    DASH=""
    for c in "$HERE/rpi5_dashboard.py" "$HOME/deterrent/rpi5_dashboard.py" \
             "$HOME/Downloads/rpi5_dashboard.py" "$HOME/rpi5_dashboard.py"; do
        [[ -f "$c" ]] && { DASH="$(readlink -f "$c")"; break; }
    done
    [[ -z "$DASH" ]] && DASH="$(find "$HOME" -name rpi5_dashboard.py -not -path '*/.*' 2>/dev/null | head -1)"
fi
[[ -f "$DASH" ]] || die "rpi5_dashboard.py not found. Pass it: SCRIPT=/path/to/rpi5_dashboard.py $0"
WORKDIR="$(dirname "$DASH")"
say "  script : $DASH"

# --------------------------------------------------------------------------- #
#  2. Locate the Python interpreter (prefer a venv with onnxruntime)           #
# --------------------------------------------------------------------------- #
say "Locating a Python with onnxruntime + cv2 ..."
PY=""
if [[ -n "${VENV:-}" ]]; then
    PY="$(readlink -f "$VENV/bin/python")"
    [[ -x "$PY" ]] || die "No python at $VENV/bin/python"
else
    # If this installer was launched from inside an active venv, prefer it.
    [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]] && PY="$VIRTUAL_ENV/bin/python"
    if [[ -z "$PY" ]]; then
        while IFS= read -r cand; do
            [[ -x "$cand" ]] || continue
            if "$cand" -c 'import onnxruntime, cv2' >/dev/null 2>&1; then PY="$cand"; break; fi
        done < <(find "$HOME" -maxdepth 4 -path '*/bin/python' -not -path '*/.cache/*' 2>/dev/null)
    fi
    [[ -z "$PY" ]] && command -v python3 >/dev/null && \
        python3 -c 'import onnxruntime, cv2' >/dev/null 2>&1 && PY="$(command -v python3)"
fi
[[ -x "$PY" ]] || die "No Python with onnxruntime+cv2 found. Pass it: VENV=/path/to/venv $0"

# Verify rather than assume — a service that fails at 3am is hard to debug headless.
"$PY" -c 'import onnxruntime, cv2, numpy' 2>/dev/null \
    || die "$PY cannot import onnxruntime/cv2/numpy. Activate the right venv or pass VENV=..."
say "  python : $PY"
"$PY" -c 'import sounddevice' >/dev/null 2>&1 \
    || warn "sounddevice not importable — the deterrent audio will be disabled at runtime."

# --------------------------------------------------------------------------- #
#  3. Locate the ONNX model — absolute path, because systemd's cwd is not yours #
# --------------------------------------------------------------------------- #
say "Locating the ONNX model ..."
if [[ -n "${MODEL:-}" ]]; then
    MODEL_ABS="$(readlink -f "$MODEL")"
else
    MODEL_ABS=""
    for c in "$WORKDIR/onnx_kaggle/yolo26n/320/fp32/best.onnx" \
             "$WORKDIR/Pi-deterrent/onnx_kaggle/yolo26n/320/fp32/best.onnx" \
             "$HOME/Downloads/Pi-deterrent/onnx_kaggle/yolo26n/320/fp32/best.onnx" \
             "$HOME/deterrent/onnx_kaggle/yolo26n/320/fp32/best.onnx" \
             "$WORKDIR/yolo26n.onnx" "$HOME/deterrent/yolo26n.onnx"; do
        [[ -f "$c" ]] && { MODEL_ABS="$(readlink -f "$c")"; break; }
    done
    # Fall back to a search, but EXCLUDE library demo models. onnxruntime ships
    # logreg_iris/mul_1/sigmoid .onnx files in site-packages; picking one of those
    # yields a service that starts and then fails in a totally confusing way.
    # Real YOLO exports are megabytes, the demos are a couple of KB.
    if [[ -z "$MODEL_ABS" ]]; then
        MODEL_ABS="$(find "$HOME" -name '*.onnx' -size +1M \
                        -not -path '*/site-packages/*' -not -path '*/dist-packages/*' \
                        -not -path '*/.*' 2>/dev/null | grep yolo26n | head -1)"
        [[ -z "$MODEL_ABS" ]] && MODEL_ABS="$(find "$HOME" -name '*.onnx' -size +1M \
                        -not -path '*/site-packages/*' -not -path '*/dist-packages/*' \
                        -not -path '*/.*' 2>/dev/null | head -1)"
    fi
fi
[[ -f "$MODEL_ABS" ]] || die "No .onnx model found. Pass it: MODEL=/path/to/best.onnx $0"
say "  model  : $MODEL_ABS"
say "  tuning : --conf $CONF --vote $VOTE ${EXTRA_ARGS:+$EXTRA_ARGS}"

# --------------------------------------------------------------------------- #
#  4. Camera sanity check                                                     #
# --------------------------------------------------------------------------- #
if compgen -G '/dev/video*' >/dev/null; then
    say "  camera : $(echo /dev/video* | tr ' ' ',')"
else
    warn "No /dev/video* right now. The service will keep retrying until one appears."
fi
# The camera and audio groups must be held by the service user, since there is
# no desktop session to grant them on boot.
for grp in video audio plugdev; do
    if ! id -nG "$RUN_USER" | tr ' ' '\n' | grep -qx "$grp"; then
        say "Adding $RUN_USER to '$grp' group ..."
        sudo usermod -aG "$grp" "$RUN_USER"
        warn "Group '$grp' added — takes effect after the next reboot."
    fi
done

# --------------------------------------------------------------------------- #
#  4b. Stop anything already holding the camera / port                        #
# --------------------------------------------------------------------------- #
# A USB camera allows exactly ONE reader. A manually-launched copy of the
# dashboard will keep /dev/video0 and the port, and the service will then
# restart-loop forever with "can't open camera by index" — invisible on a
# headless box. Clear the decks before installing.
say "Checking for processes already holding the camera or port $PORT ..."
STRAY=0
if pgrep -af 'rpi5_dashboard\.py' | grep -v "$$" >/dev/null 2>&1; then
    warn "Found a running rpi5_dashboard.py:"
    pgrep -af 'rpi5_dashboard\.py' | sed 's/^/     /'
    pkill -f 'rpi5_dashboard\.py' 2>/dev/null || true
    sleep 2
    pkill -9 -f 'rpi5_dashboard\.py' 2>/dev/null || true
    say "  stopped."
    STRAY=1
fi
if command -v fuser >/dev/null 2>&1; then
    if fuser "${PORT}/tcp" >/dev/null 2>&1; then
        warn "Port $PORT still held; freeing it."
        fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
        sleep 1
        STRAY=1
    fi
fi
[[ $STRAY -eq 0 ]] && say "  nothing in the way."

# --------------------------------------------------------------------------- #
#  5. Write the unit                                                          #
# --------------------------------------------------------------------------- #
UNIT="/etc/systemd/system/${SERVICE_NAME}.service"
say "Writing $UNIT ..."

# Only pass --audio-device if the caller set one; otherwise the script's own
# auto-detection picks the I2S DAC (and warns if it lands on HDMI).
AUDIO_ARG=""
if [[ -n "$AUDIO_DEVICE" ]]; then
    AUDIO_ARG=" --audio-device $AUDIO_DEVICE"
    say "  audio  : device $AUDIO_DEVICE (explicit)"
else
    say "  audio  : auto-detect I2S DAC"
fi
say "  camera : ${CAM_W}x${CAM_H}"

sudo tee "$UNIT" >/dev/null <<EOF
[Unit]
Description=Animal Deterrent Dashboard (YOLO + ultrasonic, headless web UI)
Documentation=file://$WORKDIR/README_DASHBOARD.md
# Start once the network is configured so the dashboard is reachable right away.
# NetworkManager-wait-online matches Raspberry Pi OS Bookworm; the plain
# network-online.target keeps this working on older images too.
Wants=network-online.target
After=network-online.target NetworkManager-wait-online.service

[Service]
Type=simple
User=$RUN_USER
Group=$RUN_GROUP
WorkingDirectory=$WORKDIR

# Absolute paths everywhere: systemd does not inherit your shell's cwd or PATH.
ExecStart=$PY $DASH --model $MODEL_ABS --port $PORT --conf $CONF --vote $VOTE --cam-width $CAM_W --cam-height $CAM_H${AUDIO_ARG} $EXTRA_ARGS

# Kill any hand-started copy FIRST: a USB camera has exactly one reader, so a
# stray instance would make this service restart-loop forever. Safe because
# ExecStartPre runs before our own ExecStart exists, and --full/-f matching is
# scoped to other users' shells too. '-' = "nothing to kill" is not a failure.
ExecStartPre=-/usr/bin/pkill -f rpi5_dashboard\\.py
# Then let the USB camera settle; it can enumerate a few seconds after boot.
ExecStartPre=/bin/sleep 5

# Keep it alive: restart on crash, on camera unplug, on anything.
Restart=always
RestartSec=10
# Never give up. Without this, >5 restarts in 10s permanently disables the unit
# — fatal on a headless box with nobody to run 'systemctl reset-failed'.
StartLimitIntervalSec=0

# Unbuffered so logs reach the journal immediately instead of sitting in a pipe.
Environment=PYTHONUNBUFFERED=1

StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

[Install]
WantedBy=multi-user.target
EOF

# --------------------------------------------------------------------------- #
#  6. Enable + start                                                          #
# --------------------------------------------------------------------------- #
say "Enabling and starting the service ..."
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME" >/dev/null
sudo systemctl restart "$SERVICE_NAME"

sleep 8
echo
if systemctl is-active --quiet "$SERVICE_NAME"; then
    IP="$(hostname -I | awk '{print $1}')"
    printf '\033[1;32m'
    echo "============================================================"
    echo "  RUNNING — and it will now start automatically on boot."
    echo "============================================================"
    printf '\033[0m'
    echo "  Dashboard : http://${IP}:${PORT}"
    echo "  By name   : http://$(hostname).local:${PORT}   (survives IP changes)"
    echo
    echo "  Logs      : journalctl -u ${SERVICE_NAME} -f"
    echo "  Restart   : sudo systemctl restart ${SERVICE_NAME}"
    echo "  Stop      : sudo systemctl stop ${SERVICE_NAME}"
    echo "  Disable   : sudo systemctl disable ${SERVICE_NAME}"
    echo
    echo "  Test it now:  sudo reboot     (then browse to the URL above)"
else
    printf '\033[1;31m'
    echo "============================================================"
    echo "  Service did NOT start. Last 30 log lines:"
    echo "============================================================"
    printf '\033[0m'
    sudo journalctl -u "$SERVICE_NAME" -n 30 --no-pager
    exit 1
fi
