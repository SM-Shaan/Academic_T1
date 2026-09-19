# Academic T1 — Raspberry Pi Animal Deterrent

This repository contains the Raspberry Pi 5 code for an animal-detection and
deterrent system. A USB camera feeds an ONNX object-detection model; detected
animals are shown in a browser dashboard and can trigger class-specific audio
patterns through a connected speaker or amplifier.

The accompanying thesis evaluates whether a low-cost, CPU-only embedded device
can provide useful real-time detection without cloud connectivity. The
experimental system compares seven lightweight YOLO-family detectors on a
seven-class dataset and deploys the selected model with an acoustic deterrent.

## Repository layout

- [`Code/rpi5_dashboard.py`](Code/rpi5_dashboard.py) — headless HTTP dashboard,
  MJPEG video stream, detection state, and audio control.
- [`Code/install_autostart.sh`](Code/install_autostart.sh) — installs and starts
  the dashboard as a systemd service on Raspberry Pi OS.
- [`Code/README_AUTOSTART.md`](Code/README_AUTOSTART.md) — autostart setup,
  configuration options, and troubleshooting.
- [`Code/rpi5_speed_benchmark.py`](Code/rpi5_speed_benchmark.py) — camera and
  inference performance benchmark.
- [`Code/rpi5_tflite_speed_all.py`](Code/rpi5_tflite_speed_all.py) — TFLite
  model speed benchmark.

Model weights are not included in this repository. Provide a compatible ONNX
model on the Pi and pass its path with `--model`.

## Quick start

Install the runtime dependencies in a Python virtual environment on the Pi:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy opencv-python onnxruntime sounddevice
```

Run the dashboard manually while testing:

```bash
python3 Code/rpi5_dashboard.py \
  --model /absolute/path/to/best.onnx
```

Open `http://<pi-ip>:8000` from a computer on the same network. Use
`hostname -I` on the Pi to find its address. For a detection-only test, add
`--no-audio`.

The detector recognizes the model's fixed class order: `cat`, `cow`, `dog`,
`fox`, `goat`, `human`, and `snake`. Humans are displayed but do not trigger
the deterrent audio.

## Reported benchmark

The deployed YOLO26n INT8-TFLite build reaches approximately 104 FPS at
`320x320` on a Raspberry Pi 5 CPU, compared with approximately 34.8 FPS for the
FP32-ONNX reference. The reported INT8 model occupies about 2.87 MB and reaches
approximately 0.888 mAP@0.5 on the evaluation set. These figures depend on the
hardware, runtime, model export, and test protocol; reproduce them on the target
device before treating them as acceptance criteria.

## Start on boot

After confirming that the camera, model, and audio device work manually, run:

```bash
cd Code
chmod +x install_autostart.sh
./install_autostart.sh
```

The installer detects the Python environment and model, creates the systemd
service, and prints the dashboard URL. See
[`Code/README_AUTOSTART.md`](Code/README_AUTOSTART.md) for overrides such as
`MODEL=`, `PORT=`, `CONF=`, and `AUDIO_DEVICE=`.

## Hardware assumptions

The scripts are intended for a Raspberry Pi 5 with a USB camera and an audio
output suitable for the deterrent speaker. Camera index, capture dimensions,
audio device, confidence threshold, and HTTP port are configurable through the
dashboard command-line options.

## Thesis and reproducibility

The LaTeX thesis source is maintained separately from this code-only repository.
Its main document assembles the introduction, literature review, methodology,
results, and conclusion chapters. For a reproducible evaluation, keep the
dataset split, class order, input resolution, confidence threshold, model
precision, warm-up iterations, timed iterations, and Raspberry Pi software
environment fixed and record them with every benchmark.

## Safety and scope

This project is a research prototype. Validate audio levels, enclosure safety,
battery autonomy, false activations, and effects on people, livestock, and
wildlife before field deployment. Detection output should be treated as an
assistance signal rather than a replacement for human supervision.

## License

No license has been added yet. Add one before redistributing this project.
