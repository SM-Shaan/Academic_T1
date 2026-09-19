# Autostart on Boot — systemd Service

Makes the dashboard launch automatically when the Pi powers on. No login, no
monitor, no keyboard. Plug in power, wait ~30s, open the dashboard from your laptop.

> **Before installing, verify the dashboard runs manually.** The service inherits
> whatever is broken. Confirm the camera streams and a detection fires the tweeter
> first — debugging that through `journalctl` on a headless box is far harder.

## The one-reader rule

A USB camera allows **exactly one** process to read it. If a hand-started copy of
`rpi5_dashboard.py` is running when the service starts, the service cannot open the
camera and will restart-loop forever — silently, with no screen to show you.

The installer handles this in two places: it kills stray instances before installing,
and the unit runs `pkill -f rpi5_dashboard.py` as an `ExecStartPre`. But if you SSH in
later and launch a copy by hand, **stop the service first**:

```bash
sudo systemctl stop animal-dashboard
python3 rpi5_dashboard.py ...        # your manual run
sudo systemctl start animal-dashboard
```

---

## Install

Copy `install_autostart.sh` next to `rpi5_dashboard.py` on the Pi, then:

```bash
chmod +x install_autostart.sh
./install_autostart.sh
```

Run it as your normal user (**not** `sudo`) — it asks for sudo only where needed.

It auto-detects your venv, script and model, verifies the imports actually work,
writes the service, enables it, starts it, and prints the dashboard URL.

### If auto-detection picks the wrong thing

```bash
VENV=~/Downloads/.venv_tflite MODEL=~/deterrent/yolo26n.onnx ./install_autostart.sh
```

| Variable | Purpose |
|---|---|
| `VENV` | venv root (expects `$VENV/bin/python`) |
| `MODEL` | absolute path to `best.onnx` |
| `SCRIPT` | absolute path to `rpi5_dashboard.py` |
| `PORT` | HTTP port (default `8000`) |
| `CONF` | confidence threshold (default `0.35`) |
| `VOTE` | class-vote smoothing window (default `7`) |
| `CAM_W` / `CAM_H` | capture size (default `640`/`480`) |
| `AUDIO_DEVICE` | PortAudio output index; empty = auto-detect the I2S DAC |
| `EXTRA_ARGS` | any other detector flags, e.g. `"--no-audio"` |

**On this Pi**, the MAX98357A is PortAudio device **1** (ALSA card 2). Auto-detection
finds it, but you can pin it explicitly:

```bash
AUDIO_DEVICE=1 ./install_autostart.sh
```

Confirm the index first — it can shift if you add or remove USB audio hardware:

```bash
python3 rpi5_dashboard.py --list-audio
```

Example — detection-only on port 5000 with a lower threshold:

```bash
PORT=5000 CONF=0.30 EXTRA_ARGS="--no-audio" ./install_autostart.sh
```

The installer skips library demo models (`logreg_iris.onnx`, `mul_1.onnx`,
`sigmoid.onnx` ship inside onnxruntime) by requiring a >1MB file outside
`site-packages`. If it still picks the wrong one, pass `MODEL=` explicitly.

---

## Verify it survives a reboot

This is the only test that matters:

```bash
sudo reboot
```

Wait ~30 seconds, then load `http://<pi-ip>:8000` from your laptop. If it appears,
you're done — the Pi now needs nothing but power.

---

## Daily commands

```bash
journalctl -u animal-dashboard -f          # live logs (detections, sweeps, errors)
journalctl -u animal-dashboard -n 50       # last 50 lines
journalctl -u animal-dashboard -b          # everything since this boot
systemctl status animal-dashboard          # is it running?
sudo systemctl restart animal-dashboard    # restart (after editing the script)
sudo systemctl stop animal-dashboard       # stop until next boot
sudo systemctl disable animal-dashboard    # stop starting on boot
sudo systemctl enable animal-dashboard     # start on boot again
```

To run the script by hand for debugging, stop the service first — otherwise two
processes fight over the camera and port 8000:

```bash
sudo systemctl stop animal-dashboard
python3 rpi5_dashboard.py --no-audio
```

---

## Why systemd (and not `rc.local` or `@reboot` cron)

Three things matter on a headless box, and only systemd gives all three:

1. **Ordering** — the unit waits for `network-online.target`, so the dashboard is
   reachable the moment it starts. A cron `@reboot` fires before the network is up.
2. **Restart on failure** — `Restart=always` brings it back if the script crashes
   or the camera is unplugged and replugged. With no screen, nobody is there to
   notice and restart it manually.
3. **Logs** — everything lands in the journal, survives reboots, and is timestamped.
   `rc.local` output goes nowhere useful.

Details worth knowing about the unit:

- **`StartLimitIntervalSec=0`** — disables systemd's default "give up after 5
  restarts in 10 seconds". Without it, a camera that's slow to enumerate can put
  the service into a permanently `failed` state that needs a manual
  `systemctl reset-failed`. On a deployed box with no monitor, that's fatal.
- **`ExecStartPre=/bin/sleep 5`** — USB cameras often enumerate a few seconds
  after the network comes up.
- **Absolute paths for the interpreter, script and model** — systemd does not
  inherit your shell's working directory or `PATH`. Relative paths are the single
  most common reason a service that works by hand fails on boot.
- **`video` and `audio` groups** — the installer adds your user to both. There's
  no desktop session at boot to grant that access. *Takes effect after a reboot.*

---

## Troubleshooting

**Service won't start**
```bash
journalctl -u animal-dashboard -n 40 --no-pager
```
The dashboard prints a clear message for a missing model or camera, and lists what
it did find.

**`ModuleNotFoundError` in the logs**
The wrong Python was picked. Find the right one and reinstall:
```bash
which python                       # with your venv activated
VENV=/path/to/venv ./install_autostart.sh
```

**Camera not found on boot, fine when run by hand**
The USB camera is enumerating late. Raise the delay:
```bash
sudo systemctl edit --full animal-dashboard   # change: ExecStartPre=/bin/sleep 15
sudo systemctl daemon-reload && sudo systemctl restart animal-dashboard
```

**Permission denied on the camera**
The group change needs a reboot to apply:
```bash
groups                             # should list: video audio
sudo reboot
```

**Dashboard runs but the laptop can't reach it**
That's the AP-isolation issue from before, not the service. Confirm on the Pi:
```bash
curl -sI http://localhost:8000/ | head -1     # HTTP/1.0 200 OK = service is fine
```

---

## Finding the Pi after a reboot

DHCP can hand the Pi a different IP. Two options that don't require a screen:

**Use the hostname** (works out of the box on Raspberry Pi OS):
```
http://raspberrypi.local:8000
```

**Or give the Pi a static IP** so it never moves:
```bash
nmcli con show                                     # find your connection name
sudo nmcli con mod "<name>" ipv4.addresses 192.168.0.101/24 \
     ipv4.gateway 192.168.0.1 ipv4.dns 8.8.8.8 ipv4.method manual
sudo reboot
```
For a permanent deployment, a static IP or a DHCP reservation in the router is
worth setting up — it means the dashboard URL never changes.

---

## Uninstall

```bash
sudo systemctl disable --now animal-dashboard
sudo rm /etc/systemd/system/animal-dashboard.service
sudo systemctl daemon-reload
```
