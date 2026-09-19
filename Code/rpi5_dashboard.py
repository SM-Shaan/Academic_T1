"""
rpi5_dashboard.py
=================
Headless web dashboard for the Pi 5 animal deterrent.

Same detection pipeline as rpi5_animal_deterrent.py (identical letterbox, decode,
NMS and audio state machine) but instead of cv2.imshow() on a local monitor, the
annotated frames are served as an MJPEG stream over HTTP, plus a JSON state
endpoint for the live stats. No screen, no X server, no VNC needed.

Architecture (deliberately simple — no Flask/websockets to install):
    capture thread  -> grabs frames from the camera as fast as it can
    detect thread   -> letterbox + ONNX + NMS + audio + annotate  (the real work)
    HTTP server     -> ThreadingHTTPServer, serves the page / MJPEG / JSON
The detect thread never blocks on HTTP clients: each client reads the latest
annotated frame from a shared slot, so a slow browser cannot stall detection.

Usage on the Pi:
    python3 rpi5_dashboard.py                               # 0.0.0.0:8000, audio on
    python3 rpi5_dashboard.py --port 8080
    python3 rpi5_dashboard.py --no-audio                    # detection-only dry run
    python3 rpi5_dashboard.py --model onnx_kaggle/yolo26n/320/fp32/best.onnx
    python3 rpi5_dashboard.py --jpeg-quality 60 --stream-fps 10   # slower networks

Then from a laptop on the SAME network:
    http://<pi-ip>:8000          (find the ip with:  hostname -I)

Press Ctrl+C on the Pi to stop.

--------------------------------------------------------------------------------
CLASS INDEX ORDER (baked into the ONNX output — do NOT reorder):
    0=cat  1=cow  2=dog  3=fox  4=goat  5=human  6=snake
--------------------------------------------------------------------------------
"""

import argparse
import json
import os
import socket
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import onnxruntime as ort

# --------------------------------------------------------------------------- #
#  Detection config — kept identical to rpi5_animal_deterrent.py               #
# --------------------------------------------------------------------------- #
CLASS_NAMES = ['cat', 'cow', 'dog', 'fox', 'goat', 'human', 'snake']

IGNORE_FOR_DETERRENT = {'human'}

FREQ_DEFAULT = 22000

# --------------------------------------------------------------------------- #
#  PER-CLASS SOUND PROFILES                                                    #
#                                                                              #
#  Each animal gets a distinct waveform, not just a distinct end frequency.    #
#  Species differ in BOTH hearing range and what startles them, and pulsed     #
#  tones resist habituation far better than one continuous sweep — an animal   #
#  tunes out a steady tone within a few exposures.                             #
#                                                                              #
#  Fields:                                                                     #
#    f0, f1     sweep start/end in Hz (f1 < f0 gives a descending sweep)       #
#    pulses     number of separate bursts inside SWEEP_DURATION                #
#    duty       fraction of each pulse slot that sounds (rest is silence)      #
#    shape      'chirp'  linear sweep f0 -> f1                                 #
#               'warble' fast sinusoidal wobble around the f0..f1 midpoint     #
#               'am'     steady tone amplitude-modulated at `am_hz`            #
#    am_hz      modulation rate for 'am' / wobble rate for 'warble'            #
#    gain       0..1 output level                                              #
#                                                                              #
#  HEARING RANGES (why these values):                                          #
#    dog    67 Hz - 45 kHz, most sensitive 8-25 kHz                            #
#    cat    48 Hz - 85 kHz, most sensitive 8-32 kHz (widest of all)            #
#    cow    23 Hz - 35 kHz     goat  78 Hz - 37 kHz                            #
#    fox    reaches ~65 kHz, hunts by low-frequency rustling                   #
#    snake  no eardrum — feels ~50-1000 Hz substrate vibration, not air        #
# --------------------------------------------------------------------------- #
SOUND_PROFILES = {
    # Dog — rapid rising chirps in the 15-25 kHz band a dog hears sharply.
    # 6 short pulses read as "urgent" and avoid habituation.
    'dog':   {'f0': 15000, 'f1': 25000, 'pulses': 6, 'duty': 0.55,
              'shape': 'chirp',  'am_hz': 0,  'gain': 0.90},

    # Cat — higher band, slow warble. Cats hear to 85 kHz but are most
    # reactive to a wavering tone; a steady one they simply ignore.
    'cat':   {'f0': 22000, 'f1': 30000, 'pulses': 3, 'duty': 0.80,
              'shape': 'warble', 'am_hz': 14, 'gain': 0.85},

    # Cow — large animal, lower band, long slow pulses.
    'cow':   {'f0': 12000, 'f1': 20000, 'pulses': 2, 'duty': 0.85,
              'shape': 'chirp',  'am_hz': 0,  'gain': 0.95},

    # Goat — mid band, buzzy amplitude modulation.
    'goat':  {'f0': 14000, 'f1': 22000, 'pulses': 4, 'duty': 0.70,
              'shape': 'am',     'am_hz': 30, 'gain': 0.90},

    # Fox — DESCENDING sweep (f1 < f0). Falling pitch reads as a predator
    # cue rather than prey; foxes are wary and respond to novelty.
    'fox':   {'f0': 26000, 'f1': 16000, 'pulses': 5, 'duty': 0.60,
              'shape': 'chirp',  'am_hz': 0,  'gain': 0.90},

    # Snake — snakes have NO eardrum and are effectively deaf to airborne
    # sound. This is a low-frequency buzz meant to couple into the ground
    # as vibration, which is what they actually sense. Expect limited effect
    # from a tweeter; a surface transducer would work far better.
    'snake': {'f0': 200,   'f1': 800,   'pulses': 8, 'duty': 0.50,
              'shape': 'am',     'am_hz': 60, 'gain': 1.00},
    # 'human' intentionally absent — never deterred
}

# Fallback for any class without an explicit profile.
DEFAULT_PROFILE = {'f0': 10000, 'f1': FREQ_DEFAULT, 'pulses': 3, 'duty': 0.75,
                   'shape': 'chirp', 'am_hz': 0, 'gain': 0.90}

# Kept so anything referencing the old map still works; the dashboard shows this
# as the headline "target frequency" for a class.
CLASS_TARGET_FREQ = {k: v['f1'] for k, v in SOUND_PROFILES.items()}

SWEEP_DURATION = 3.0     # seconds — one full pulse train
LOST_THRESHOLD = 1.0     # seconds — grace after last sighting before stopping
SAMPLE_RATE    = 96000   # Hz — Nyquist 48 kHz, enough for every profile above
SWEEP_START_FREQ = 10000 # Hz — legacy default start for the fallback profile

CONF_THRESHOLD = 0.50
IOU_THRESHOLD  = 0.45

EVENT_LOG_MAX = 60       # how many recent detection events the dashboard keeps


def parse_args():
    p = argparse.ArgumentParser(description="RPi5 animal deterrent — web dashboard")
    p.add_argument("--model", default="onnx_kaggle/yolo26n/320/fp32/best.onnx",
                   help="Path to the exported ONNX file. yolo26n performs BEST at "
                        "imgsz=320 — its confidence collapses at 640. Use the 320 export.")
    p.add_argument("--imgsz", type=int, default=None,
                   help="Override inference size (else taken from the ONNX input shape)")
    p.add_argument("--cam", type=int, default=0, help="USB camera index (default 0)")
    p.add_argument("--camera-backend", default="auto",
                   choices=["auto", "v4l2", "picamera2"],
                   help="auto = probe every /dev/video* node, then Picamera2 (CSI). "
                        "Force one if auto-detection picks the wrong device.")
    p.add_argument("--cam-width", type=int, default=640, help="Camera capture width")
    p.add_argument("--cam-height", type=int, default=480, help="Camera capture height")
    p.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=IOU_THRESHOLD, help="NMS IoU threshold")
    p.add_argument("--vote", type=int, default=7,
                   help="Smooth the deterrent class over the last N frames "
                        "(confidence-weighted majority). Stops the frequency "
                        "flipping dog/goat/fox on a hard subject. 1 disables "
                        "(default 7)")
    p.add_argument("--persist", type=int, default=5,
                   help="Hold a box on screen for N frames after it stops being "
                        "detected. Smooths flicker from confidence hovering near "
                        "the threshold. 0 disables (default 5)")
    p.add_argument("--threads", type=int, default=4,
                   help="onnxruntime intra-op threads (Pi 5 has 4 cores)")
    p.add_argument("--no-audio", action="store_true",
                   help="Disable the frequency generator (detection-only dry run)")
    p.add_argument("--audio-device", type=int, default=None,
                   help="sounddevice output device id for the MAX98357A (I2S DAC)")
    p.add_argument("--list-audio", action="store_true",
                   help="List audio devices and exit")
    p.add_argument("--test-sound", nargs="?", const="all", default=None,
                   metavar="CLASS",
                   help="Play each class's deterrent sound and exit. "
                        "--test-sound dog plays just the dog profile.")
    p.add_argument("--audible", action="store_true",
                   help="Shift every profile into 1-8kHz so YOU can hear the "
                        "difference between classes. For bench testing only — "
                        "not for deployment.")
    p.add_argument("--no-fit-hardware", action="store_true",
                   help="Use the raw species-hearing frequencies instead of "
                        "fitting them to the amplifier's usable band. Only "
                        "useful with a genuine ultrasonic amp (>25kHz).")
    p.add_argument("--amp-bandwidth", type=int, default=AMP_BANDWIDTH_HZ,
                   metavar="HZ",
                   help=f"Upper frequency the amplifier can reproduce "
                        f"(default {AMP_BANDWIDTH_HZ}, correct for the "
                        f"MAX98357A). Raise it if you fit a wideband amp.")
    # --- dashboard-specific ---
    p.add_argument("--host", default="0.0.0.0",
                   help="Bind address. 0.0.0.0 = reachable from the LAN (default)")
    p.add_argument("--port", type=int, default=8000, help="HTTP port (default 8000)")
    p.add_argument("--jpeg-quality", type=int, default=75,
                   help="MJPEG quality 1-100. Lower = less bandwidth (default 75)")
    p.add_argument("--stream-fps", type=float, default=15.0,
                   help="Max frames/sec pushed to each browser (default 15)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Audio — unchanged from rpi5_animal_deterrent.py                             #
# --------------------------------------------------------------------------- #
def audible_profile(profile):
    """Transpose a profile into 1-8 kHz so a human can actually hear it.

    The deployment tones are mostly ultrasonic; to a person every class sounds
    the same (i.e. silent, or just the click of the envelope). This maps the
    band down while preserving the pattern — pulse count, duty, shape and sweep
    direction all survive, so classes stay clearly distinguishable by ear.
    """
    p = dict(profile)
    lo, hi = 1000.0, 8000.0
    src_lo, src_hi = 200.0, 30000.0
    def m(f):
        frac = (float(f) - src_lo) / (src_hi - src_lo)
        return lo + max(0.0, min(1.0, frac)) * (hi - lo)
    p['f0'], p['f1'] = m(profile['f0']), m(profile['f1'])
    return p


def clamp_profile(profile, sample_rate):
    """Scale a profile's frequencies under the Nyquist limit of the real device.

    Playing a 30 kHz tone at 48 kHz would alias down to an audible 18 kHz
    whine — the opposite of what we want. Cap at 90% of Nyquist and shift the
    whole band down together so the sweep keeps its shape and direction.
    """
    limit = 0.90 * (sample_rate / 2.0)
    p = dict(profile)
    hi = max(p['f0'], p['f1'])
    if hi > limit:
        k = limit / hi
        p['f0'] = p['f0'] * k
        p['f1'] = p['f1'] * k
        p['_clamped'] = True
    return p


def _render_pulse(profile, seconds, sample_rate):
    """Render ONE pulse of a profile as float32 in [-1, 1]."""
    n = max(int(sample_rate * seconds), 1)
    t = np.linspace(0, seconds, n, False)
    f0, f1 = float(profile['f0']), float(profile['f1'])
    shape = profile['shape']

    if shape == 'chirp':
        # Linear chirp. Works ascending (f1>f0) and descending (f1<f0).
        phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * seconds) * t ** 2)
        wave = np.sin(phase)

    elif shape == 'warble':
        # Carrier at the band midpoint, pitch wobbling across the full band.
        centre = (f0 + f1) / 2.0
        dev = abs(f1 - f0) / 2.0
        rate = max(profile.get('am_hz', 10), 1)
        # integrate instantaneous frequency to get phase (FM synthesis)
        inst = centre + dev * np.sin(2 * np.pi * rate * t)
        phase = 2 * np.pi * np.cumsum(inst) / sample_rate
        wave = np.sin(phase)

    elif shape == 'am':
        # Steady carrier, amplitude modulated -> a buzzy, "rough" texture.
        centre = (f0 + f1) / 2.0
        rate = max(profile.get('am_hz', 30), 1)
        carrier = np.sin(2 * np.pi * centre * t)
        env = 0.5 * (1.0 + np.sin(2 * np.pi * rate * t))
        wave = carrier * env

    else:
        wave = np.sin(2 * np.pi * f0 * t)

    # 5 ms raised-cosine fade in/out. Without this the abrupt start/stop makes
    # a broadband click that both wastes amplifier headroom and is audible to
    # humans as a "tick" on every pulse.
    edge = min(int(0.005 * sample_rate), n // 2)
    if edge > 0:
        ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, edge)))
        wave[:edge] *= ramp
        wave[-edge:] *= ramp[::-1]

    return wave


AUDIBLE_MODE = False   # set from --audible; bench testing only
FIT_HARDWARE = True    # set from --no-fit-hardware; see fit_to_hardware()

# Upper frequency the output stage can actually reproduce, in Hz.
# The MAX98357A is a Class-D amp with a ~20 kHz output filter: past that its
# response falls off a cliff regardless of what we synthesise. Squeezing every
# class into 13-21 kHz also makes them overlap and stop being distinguishable.
# Setting a realistic ceiling spreads them back out across the usable band.
AMP_BANDWIDTH_HZ = 20000

# Lowest frequency worth using — below this a small tweeter produces almost
# nothing, and it starts being unpleasant for people nearby.
AMP_MIN_HZ = 2000


def fit_to_hardware(profile, ceiling=None, floor=AMP_MIN_HZ):
    """Re-map a profile's band into what the amplifier can actually output.

    Profiles are authored around each species' hearing range (up to 30 kHz).
    Clamping those to a 20 kHz ceiling compresses them all into the same
    narrow strip, so cat and dog stop sounding different — the opposite of
    what we want. This instead spreads the authored 2-30 kHz span across the
    reproducible floor..ceiling range, preserving BOTH the ordering between
    classes and each sweep's direction.
    """
    # Read the module global at call time so --amp-bandwidth takes effect.
    if ceiling is None:
        ceiling = AMP_BANDWIDTH_HZ
    p = dict(profile)
    src_lo, src_hi = 2000.0, 30000.0
    span = ceiling - floor

    def m(f):
        frac = (float(f) - src_lo) / (src_hi - src_lo)
        return floor + max(0.0, min(1.0, frac)) * span

    # Snake is deliberately low-frequency (substrate vibration, not hearing);
    # remapping it upward would defeat the point.
    if max(profile['f0'], profile['f1']) <= floor:
        return p

    p['f0'], p['f1'] = m(profile['f0']), m(profile['f1'])
    p['_fitted'] = True
    return p


def generate_sound_for_class(class_name, duration=SWEEP_DURATION,
                             sample_rate=SAMPLE_RATE):
    """Build the full pulse train for one class. float32 in [-1, 1]."""
    p = SOUND_PROFILES.get(class_name, DEFAULT_PROFILE)
    if AUDIBLE_MODE:
        p = audible_profile(p)
    elif FIT_HARDWARE:
        p = fit_to_hardware(p)
    p = clamp_profile(p, sample_rate)
    pulses = max(int(p.get('pulses', 1)), 1)
    duty = min(max(float(p.get('duty', 0.75)), 0.05), 1.0)

    slot = duration / pulses          # one pulse + its trailing silence
    on = slot * duty
    off_n = max(int(sample_rate * (slot - on)), 0)

    pulse = _render_pulse(p, on, sample_rate)
    silence = np.zeros(off_n, dtype=np.float64)

    train = np.concatenate([np.concatenate([pulse, silence])
                            for _ in range(pulses)])
    train *= float(p.get('gain', 0.9))   # headroom so the DAC doesn't clip
    return train.astype(np.float32)


def generate_sweep_array(target_frequency, duration, sample_rate):
    """Legacy single-chirp generator, kept for backwards compatibility."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    phase = 2 * np.pi * (
        SWEEP_START_FREQ * t
        + (target_frequency - SWEEP_START_FREQ) / (2 * duration) * t ** 2
    )
    return (0.9 * np.sin(phase)).astype(np.float32)


def describe_profile(class_name, sample_rate=SAMPLE_RATE):
    """One-line human description, for the console and the dashboard."""
    raw = SOUND_PROFILES.get(class_name, DEFAULT_PROFILE)
    if AUDIBLE_MODE:
        raw = audible_profile(raw)
    elif FIT_HARDWARE:
        raw = fit_to_hardware(raw)
    p = clamp_profile(raw, sample_rate)
    base = f"{p['f0']/1000:.1f}k -> {p['f1']/1000:.1f}kHz"
    kind = {'chirp': 'sweep', 'warble': 'warble', 'am': f"buzz@{p['am_hz']}Hz"}
    txt = f"{base}, {p['pulses']}x {kind.get(p['shape'], p['shape'])}"
    if p.get('_clamped'):
        txt += " (capped to device Nyquist)"
    return txt


class AudioPlayer:
    """Wraps sounddevice so the script still runs with --no-audio or if
    sounddevice/the DAC isn't present."""

    def __init__(self, enabled, device, want_rate=SAMPLE_RATE):
        self.enabled = enabled
        self.sd = None
        self.status = "disabled"
        self.rate = want_rate
        self.device = device
        self.channels = 1
        if not enabled:
            print("Audio DISABLED (--no-audio): detections will be logged only.")
            return
        try:
            import sounddevice as sd
            if device is not None:
                sd.default.device = device
            self.sd = sd

            if device is None:
                device = self._autodetect_device(sd)
                if device is not None:
                    sd.default.device = device

            dev_info = sd.query_devices(device if device is not None
                                        else sd.default.device[1], 'output')
            self.channels = min(2, max(1, dev_info.get('max_output_channels', 1)))
            print(f"Audio device: {dev_info['name']} "
                  f"({dev_info.get('max_output_channels')} ch, "
                  f"default {dev_info.get('default_samplerate')} Hz)")

            # The MAX98357A (and most I2S DACs) top out at 48 kHz — asking for
            # 96 kHz makes PortAudio refuse the stream, which previously failed
            # silently. Probe downwards and keep the best rate that works.
            self.rate = self._negotiate_rate(
                [want_rate, 48000, 44100, 32000, 22050], device)
            if self.rate is None:
                raise RuntimeError("no supported sample rate on this device")

            if self.rate < want_rate:
                print(f"NOTE: device rejected {want_rate} Hz; using {self.rate} Hz. "
                      f"Max reproducible tone is {self.rate // 2} Hz "
                      f"(Nyquist) — ultrasonic profiles will be capped.")
            self.status = f"ready @ {self.rate}Hz"
            print(f"Audio ENABLED at {self.rate} Hz, {self.channels} ch.")

            # ALSA's 'default' accepts everything and routes it nowhere useful.
            # Warn loudly rather than let a silent setup look successful.
            dname = dev_info.get('name', '').lower()
            if any(k in dname for k in ('hdmi', 'vc4')):
                print("\n  " + "!" * 58)
                print("  WARNING: audio is going to HDMI, not your DAC.")
                print("  Your tweeter will stay SILENT — the sound is leaving")
                print("  down the HDMI cable instead.")
                print("  Find the MAX98357A:   --list-audio   (or aplay -l)")
                print("  Then force it:        --audio-device N")
                print("  " + "!" * 58 + "\n")
                self.status = "WRONG DEVICE (HDMI)"
            elif dname in ('default', 'sysdefault', 'pulse'):
                print("\n  WARNING: playing to ALSA's generic '" + dname + "' device.")
                print("  On a Pi this normally routes to HDMI (card 0), so the")
                print("  tweeter stays silent while playback reports success.")
                print("  List real devices:  aplay -l   /   --list-audio")
                print("  Then force it:      --audio-device N\n")
            if dev_info.get('max_output_channels', 0) > 8:
                print(f"  WARNING: device reports "
                      f"{dev_info['max_output_channels']} output channels — that is "
                      f"an ALSA alias, not real hardware. Use --audio-device N.\n")

            # An I2S DAC that was never enabled in config.txt produces exactly
            # this state: no real card anywhere, only ALSA's catch-all. Say so
            # explicitly — otherwise everything "succeeds" and stays silent.
            if not self._any_real_card(sd):
                self.status = "no I2S card — check config.txt"
                print("  " + "!" * 58)
                print("  NO REAL SOUND CARD FOUND.")
                print("  If you are using a MAX98357A, it needs a device-tree")
                print("  overlay before Linux creates an audio device for it:")
                print("")
                print("      sudo bash setup_i2s_audio.sh")
                print("      sudo reboot")
                print("")
                print("  Until then, playback goes to ALSA's dummy device and")
                print("  you will hear NOTHING even though no error is raised.")
                print("  " + "!" * 58 + "\n")
        except Exception as e:
            print(f"WARNING: could not initialise sounddevice ({e}). "
                  f"Continuing WITHOUT audio.")
            self.enabled = False
            self.status = f"unavailable ({e})"

    @staticmethod
    def _any_real_card(sd):
        """True if any output device looks like actual hardware.

        ALSA always offers 'default'/'sysdefault'/'pulse' even with no sound
        card at all, so their presence proves nothing about whether audio can
        physically leave the board.
        """
        # HDMI counts as real hardware but is never the deterrent speaker, so
        # it is excluded here — a Pi with only HDMI has no usable output.
        # NOTE: the HDMI devices are themselves named "...i2s-hifi...", so a
        # naive substring test for 'i2s' matches them too. Check the DAC names
        # FIRST and only then fall through to the exclusions.
        known_dac = ('max98357', 'hifiberry', 'pcm5102', 'pcm512', 'sndrpi',
                     'iqaudio', 'justboom')
        skip = ('default', 'sysdefault', 'pulse', 'dmix', 'dsnoop', 'null',
                'jack', 'samplerate', 'speexrate', 'upmix', 'downmix',
                'surround', 'lavrate', 'oss', 'plughw', 'hdmi', 'vc4',
                'front', 'iec958', 'spdif')
        try:
            for d in sd.query_devices():
                if d.get('max_output_channels', 0) < 1:
                    continue
                nm = d.get('name', '').lower()
                if any(k in nm for k in known_dac):
                    return True
                if any(a in nm for a in skip):
                    continue
                if d.get('max_output_channels', 0) > 8:
                    continue          # channel counts like 128 mean an alias
                return True
        except Exception:
            pass
        return False

    @staticmethod
    def _autodetect_device(sd):
        """Find the real DAC rather than ALSA's 'default' alias.

        On the Pi, 'default' is a plug/dmix alias that reports absurd values
        (e.g. 128 channels) and happily ACCEPTS any sample rate while routing
        the audio nowhere useful. That makes a broken setup look healthy. Prefer
        a device whose name looks like the I2S hat.
        """
        # Real I2S DAC hats, in priority order.
        # Specific DAC part/hat names only. Deliberately NOT 'i2s': the Pi's
        # HDMI outputs are named "vc4-hdmi-0: MAI PCM i2s-hifi-0" and would
        # match it, which is exactly the trap that sends audio to the monitor.
        prefer = ('max98357', 'hifiberry', 'pcm5102', 'pcm512',
                  'sndrpi', 'iqaudio', 'justboom')
        # ALSA plumbing and outputs that are NOT the deterrent speaker.
        # 'hdmi'/'vc4' matter most here: on a Pi they are card 0, so ALSA's
        # 'default' sends audio down the HDMI cable and the tweeter stays
        # silent while everything reports success.
        avoid = ('default', 'sysdefault', 'pulse', 'dmix', 'dsnoop', 'null',
                 'jack', 'hdmi', 'vc4', 'headphone', 'samplerate', 'speexrate',
                 'upmix', 'downmix', 'surround', 'lavrate', 'oss', 'front',
                 'iec958', 'spdif')

        devices = list(sd.query_devices())

        for i, d in enumerate(devices):
            if d.get('max_output_channels', 0) < 1:
                continue
            name = d.get('name', '').lower()
            if any(k in name for k in prefer) and not any(k in name for k in avoid):
                print(f"Auto-selected audio device #{i}: {d['name']}")
                return i

        # No known DAC name — take the first plausible real device.
        for i, d in enumerate(devices):
            if d.get('max_output_channels', 0) < 1:
                continue
            name = d.get('name', '').lower()
            if any(k in name for k in avoid):
                continue
            if d.get('max_output_channels', 0) > 8:
                continue          # channel counts like 128 mean an ALSA alias
            print(f"Auto-selected audio device #{i}: {d['name']}")
            return i

        print("WARNING: could not identify a real DAC — falling back to ALSA "
              "'default', which on a Pi usually means HDMI. Pass --audio-device N.")
        return None

    def _negotiate_rate(self, candidates, device):
        """Return the first sample rate the output device actually accepts."""
        for r in candidates:
            try:
                self.sd.check_output_settings(
                    device=device, samplerate=r, channels=self.channels,
                    dtype='float32')
                return r
            except Exception:
                continue
        return None

    def play(self, wave, sample_rate=None):
        """Play a mono float32 array. Never raises — a dead speaker must not
        take down the detector."""
        if not (self.enabled and self.sd is not None):
            return False
        try:
            data = wave
            # Duplicate mono -> stereo when the DAC insists on 2 channels.
            if self.channels == 2 and data.ndim == 1:
                data = np.column_stack([data, data])
            self.sd.play(data, samplerate=self.rate, blocking=False)
            return True
        except Exception as e:
            print(f"AUDIO ERROR: playback failed ({e})", flush=True)
            STATE.add_event(f"Audio error: {e}")
            self.status = f"error: {e}"
            return False

    def stop(self):
        if self.enabled and self.sd is not None:
            try:
                self.sd.stop()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
#  ONNX inference + decode — unchanged from rpi5_animal_deterrent.py           #
# --------------------------------------------------------------------------- #
def resolve_model(model_path):
    """Turn --model into an existing absolute path, or exit with a useful message.

    Relative paths are resolved against the script's own directory first, so the
    dashboard works the same whether it's started from a shell or by systemd
    (which gives the process an arbitrary working directory)."""
    from pathlib import Path
    p = Path(model_path).expanduser()
    candidates = [p] if p.is_absolute() else [
        Path.cwd() / p,
        Path(__file__).resolve().parent / p,
    ]
    for c in candidates:
        if c.is_file():
            return str(c.resolve())

    here = Path(__file__).resolve().parent
    print(f"\nERROR: model not found: {model_path}")
    print("Looked in:")
    for c in dict.fromkeys(str(x) for x in candidates):   # de-dup when cwd==script dir
        print(f"  {c}")

    # Search widely, then DISCARD anything that obviously isn't a detector.
    # site-packages ships demo models (logreg_iris.onnx, mul_1.onnx, sigmoid.onnx);
    # suggesting one of those sends the user straight into a baffling crash.
    # Bounded-depth walk: an unbounded rglob over $HOME can take minutes on a
    # Pi with a large SD card, and the user is staring at a hung prompt.
    def scan(root, max_depth=6):
        root = Path(root)
        hits, stack = [], [(root, 0)]
        while stack:
            d, depth = stack.pop()
            if depth > max_depth:
                continue
            try:
                for e in os.scandir(d):
                    if e.is_dir(follow_symlinks=False):
                        if e.name.startswith('.') and e.name not in ('.venv',):
                            continue
                        if e.name in ('node_modules', '__pycache__', 'proc', 'sys'):
                            continue
                        stack.append((e.path, depth + 1))
                    elif e.name.endswith('.onnx'):
                        hits.append(Path(e.path))
            except (PermissionError, OSError):
                continue
        return hits

    found = set()
    for root in {here, Path.cwd(), Path.home()}:
        found.update(scan(root))

    def is_plausible_detector(f):
        s = str(f)
        if "site-packages" in s or "/.venv" in s or "/dist-packages" in s:
            return False
        # A YOLO detector is megabytes; the bundled demo models are a few KB.
        try:
            return f.stat().st_size > 1_000_000
        except OSError:
            return False

    usable = sorted(f for f in found if is_plausible_detector(f))
    if usable:
        print("\nDetection models found on this Pi:")
        for f in usable[:10]:
            print(f"  {f}  ({f.stat().st_size / 1e6:.1f} MB)")
        # Prefer a yolo26n 320 fp32 export if one is present — that's the tuned default.
        best = next((f for f in usable
                     if "yolo26n" in str(f) and "320" in str(f) and "fp32" in str(f)),
                    usable[0])
        print(f"\nRe-run with:\n  python3 {Path(__file__).name} --model {best}")
    else:
        skipped = len(found) - len(usable)
        print("\nNo detection model found"
              + (f" ({skipped} library demo .onnx files ignored)." if skipped else "."))
        print("Copy it to the Pi, e.g.:")
        print("  scp best.onnx pi@<pi-ip>:~/deterrent/onnx_kaggle/yolo26n/320/fp32/")
    raise SystemExit(2)


def build_session(model_path, threads):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ort.get_available_providers()
    ep = ['XnnpackExecutionProvider', 'CPUExecutionProvider'] \
        if 'XnnpackExecutionProvider' in providers else ['CPUExecutionProvider']
    sess = ort.InferenceSession(model_path, sess_options=opts, providers=ep)
    print(f"ONNX providers in use: {sess.get_providers()}")
    return sess


def letterbox(frame_bgr, size):
    """Resize with unchanged aspect ratio + padding to (size,size). Returns the
    padded RGB float tensor plus the scale/pad needed to map boxes back."""
    h0, w0 = frame_bgr.shape[:2]
    r = min(size / h0, size / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = rgb.transpose(2, 0, 1)[None, ...]  # (1,3,size,size)
    return tensor, r, left, top


def decode(output, conf_thres):
    """Decode an ONNX detection output into (boxes_xyxy, class_ids, scores) in
    letterboxed-pixel space, plus a flag saying whether NMS still needs running.

    Handles BOTH export formats:
      * mainline ultralytics >=8.4 embedded-NMS: (1, 300, 6) = [x1,y1,x2,y2,conf,cls]
      * raw YOLOv8 head: (1, 4+nc, N) -> transpose, argmax classes, cx,cy,w,h
    """
    o = output
    if o.ndim == 3 and o.shape[2] == 6:
        rows = o[0]
        keep = rows[:, 4] >= conf_thres
        rows = rows[keep]
        return rows[:, :4], rows[:, 5].astype(int), rows[:, 4], False
    pred = o[0].transpose()            # (N, 4+nc)
    cxcywh = pred[:, :4]
    scores_all = pred[:, 4:]
    class_ids = scores_all.argmax(axis=1)
    scores = scores_all[np.arange(scores_all.shape[0]), class_ids]
    keep = scores >= conf_thres
    cxcywh = cxcywh[keep]
    boxes_xyxy = np.empty_like(cxcywh)
    boxes_xyxy[:, 0] = cxcywh[:, 0] - cxcywh[:, 2] / 2
    boxes_xyxy[:, 1] = cxcywh[:, 1] - cxcywh[:, 3] / 2
    boxes_xyxy[:, 2] = cxcywh[:, 0] + cxcywh[:, 2] / 2
    boxes_xyxy[:, 3] = cxcywh[:, 1] + cxcywh[:, 3] / 2
    return boxes_xyxy, class_ids[keep], scores[keep], True


def nms(boxes_xyxy, scores, iou_thres):
    """Standard NMS on x1,y1,x2,y2 boxes. Returns kept indices."""
    if len(boxes_xyxy) == 0:
        return []
    x1, y1 = boxes_xyxy[:, 0], boxes_xyxy[:, 1]
    x2, y2 = boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return keep


def pick_target_freq(class_name):
    return CLASS_TARGET_FREQ.get(class_name, FREQ_DEFAULT)


# --------------------------------------------------------------------------- #
#  Shared state between the detect thread and the HTTP handlers                #
# --------------------------------------------------------------------------- #
class SharedState:
    """One annotated JPEG slot + a stats dict, guarded by a lock.

    Browsers read the newest frame; they never queue up behind the detector, so
    a slow client degrades its own framerate but not the detection loop.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg = None            # latest annotated frame, JPEG bytes
        self.frame_seq = 0          # bumped on every new frame
        self.cond = threading.Condition(self.lock)
        self.stats = {
            "fps": 0.0,
            "infer_ms": 0.0,
            "detections": [],
            "deterrent_active": False,
            "deterrent_class": None,
            "target_freq": None,
            "frames": 0,
            "uptime_s": 0.0,
            "camera_ok": False,
            "camera_source": "starting",
            "conf": CONF_THRESHOLD,   # live-tunable from the dashboard slider
            "raw_detections": [],     # everything above conf/2, for threshold tuning
            "audio": "unknown",
            "model": "",
            "imgsz": 0,
            "events": [],
        }
        self.events = deque(maxlen=EVENT_LOG_MAX)
        self.running = True

    def publish_frame(self, jpeg_bytes):
        with self.cond:
            self.jpeg = jpeg_bytes
            self.frame_seq += 1
            self.cond.notify_all()

    def wait_for_frame(self, last_seq, timeout=5.0):
        """Block until a frame newer than last_seq exists. Returns (jpeg, seq)."""
        with self.cond:
            if self.frame_seq == last_seq:
                self.cond.wait(timeout)
            return self.jpeg, self.frame_seq

    def update_stats(self, **kw):
        with self.lock:
            self.stats.update(kw)

    def add_event(self, text):
        stamp = time.strftime("%H:%M:%S")
        with self.lock:
            self.events.appendleft({"t": stamp, "msg": text})

    def snapshot(self):
        with self.lock:
            s = dict(self.stats)
            s["events"] = list(self.events)
            return s


STATE = SharedState()


# --------------------------------------------------------------------------- #
#  Capture thread — keeps only the newest frame                                #
# --------------------------------------------------------------------------- #
class Camera(threading.Thread):
    """Grabs continuously so the detector always works on a fresh frame.

    OpenCV's VideoCapture buffers internally; if the detector is slower than the
    camera, read() returns progressively staler frames. Draining in a dedicated
    thread keeps latency low.
    """

    daemon = True

    def __init__(self, index, width, height, backend="auto"):
        super().__init__(name="camera")
        self.want_index = index
        self.width = width
        self.height = height
        self.backend = backend
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.stopped = False
        self.source = "not opened"
        self.cap = None       # cv2.VideoCapture
        self.picam = None     # Picamera2, for CSI ribbon cameras

    # ---- opening -------------------------------------------------------- #
    def _try_v4l2(self, index):
        """Open /dev/videoN via V4L2 and PROVE it delivers a frame.

        isOpened() lies on the Pi: the ISP/codec nodes (video19-35 on a Pi 5)
        open successfully but never produce an image, so we require a real
        successful read() before accepting a device.
        """
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        except Exception:
            return None
        if not cap.isOpened():
            cap.release()
            return None

        # Request MJPG BEFORE the resolution. USB webcams expose YUYV only at
        # low framerates (often 5fps at 1080p) and reserve their usable modes
        # for MJPG; OpenCV otherwise picks YUYV and either crawls or fails to
        # negotiate at all. Setting FOURCC first matters — some UVC drivers
        # ignore a format change made after the resolution is fixed.
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        for _ in range(8):            # UVC cameras need a moment to start streaming
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                if (aw, ah) != (self.width, self.height):
                    print(f"  note: camera negotiated {aw}x{ah} "
                          f"(asked for {self.width}x{self.height})")
                return cap
            time.sleep(0.25)

        # MJPG may be unsupported on this device — retry with the driver's
        # own default format before giving up on the node.
        cap.release()
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                for _ in range(5):
                    ok, frame = cap.read()
                    if ok and frame is not None and frame.size > 0:
                        print(f"  note: /dev/video{index} works without MJPG "
                              f"(default format)")
                        return cap
                    time.sleep(0.25)
            cap.release()
        except Exception:
            pass
        return None

    def _try_picamera2(self):
        """CSI ribbon cameras are NOT reachable through V4L2 indices on the
        Pi 5 — they need libcamera via Picamera2."""
        try:
            from picamera2 import Picamera2
        except ImportError:
            print("  Picamera2 not installed — skipping CSI camera check.")
            print("    sudo apt install -y python3-picamera2")
            return None
        try:
            pc = Picamera2()
            cfg = pc.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"})
            pc.configure(cfg)
            pc.start()
            time.sleep(1.0)           # let AE/AWB settle before the first frame
            test = pc.capture_array()
            if test is None or test.size == 0:
                pc.stop()
                pc.close()
                return None
            return pc
        except Exception as e:
            print(f"  Picamera2 attempt failed: {e}")
            return None

    def open(self):
        """Find a working camera. Returns True on success."""
        import glob as _glob

        nodes = []      # defined up front: the picamera2 branch below reads it
        if self.backend in ("auto", "v4l2"):
            # Try the requested index first, then every other /dev/video* node.
            # On a Pi 5, video19 (hevc decoder) and video20-35 (pisp_be ISP)
            # are internal blocks, never cameras — probing each costs ~2s of
            # failed reads, so skip them unless explicitly requested.
            INTERNAL = set(range(19, 36))
            nodes = [self.want_index]
            for p in sorted(_glob.glob("/dev/video*")):
                try:
                    n = int(p.replace("/dev/video", ""))
                    if n not in nodes and n not in INTERNAL:
                        nodes.append(n)
                except ValueError:
                    pass
            for n in nodes:
                cap = self._try_v4l2(n)
                if cap is not None:
                    self.cap = cap
                    self.source = f"/dev/video{n} (V4L2)"
                    self.ok = True
                    print(f"Camera opened: {self.source}")
                    return True

        # Only reach for libcamera if explicitly asked, or if there were no
        # plausible V4L2 nodes at all — on a USB-camera system Picamera2 costs
        # a pointless ~1s of startup on every retry.
        if self.backend == "picamera2" or (
                self.backend == "auto" and len(nodes) <= 1):
            pc = self._try_picamera2()
            if pc is not None:
                self.picam = pc
                self.source = "CSI camera (Picamera2)"
                self.ok = True
                print(f"Camera opened: {self.source}")
                return True

        self.ok = False
        # Distinguish "held by another process" from "not present" — they look
        # identical to OpenCV but need completely different fixes.
        self.source = "camera busy" if self._device_busy() else "no camera"
        return False

    @staticmethod
    def _device_busy():
        """True if a /dev/video* capture node exists but is locked by someone
        else. Opening a busy V4L2 node raises EBUSY."""
        import errno
        import glob as _glob
        import os
        for p in sorted(_glob.glob("/dev/video*")):
            try:
                n = int(p.replace("/dev/video", ""))
            except ValueError:
                continue
            if 19 <= n <= 35:          # Pi 5 ISP/codec blocks, never cameras
                continue
            try:
                fd = os.open(p, os.O_RDWR | os.O_NONBLOCK)
                os.close(fd)
            except OSError as e:
                if e.errno == errno.EBUSY:
                    return True
            except Exception:
                pass
        return False

    # ---- capture loop --------------------------------------------------- #
    def run(self):
        while not self.stopped:
            if not self.ok:
                # Keep retrying: the camera may be plugged in after boot, which
                # matters when there is no screen to notice the failure.
                if self.open():
                    STATE.add_event(f"Camera connected — {self.source}")
                else:
                    time.sleep(2.0)
                continue

            try:
                if self.picam is not None:
                    frame = self.picam.capture_array()
                    ok = frame is not None
                    if ok:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    ok, frame = self.cap.read()
            except Exception:
                ok, frame = False, None

            if not ok or frame is None:
                print("Camera read failed — reconnecting...", flush=True)
                STATE.add_event("Camera lost — reconnecting")
                self._close()
                self.ok = False
                continue

            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def _close(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        if self.picam is not None:
            try:
                self.picam.stop()
                self.picam.close()
            except Exception:
                pass
            self.picam = None

    def release(self):
        self.stopped = True
        time.sleep(0.2)
        self._close()


# --------------------------------------------------------------------------- #
#  Detection thread — the same loop as rpi5_animal_deterrent.py                #
# --------------------------------------------------------------------------- #
def detection_loop(args, cam, audio, sess, inp_name, size):
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]

    # Boxes are held on screen for a few frames after they stop being detected.
    # Confidence on a live feed hovers around the threshold and crosses it back
    # and forth, which reads as flicker; persistence smooths that out without
    # inventing detections that never happened.
    sticky = []          # [{"box","cls","conf","ttl"}]
    STICKY_FRAMES = max(0, args.persist)

    # Class-vote smoothing. A single frame's argmax is noisy — on a hard subject
    # the model can flip dog->goat->fox between consecutive frames. Voting over a
    # short window picks the class the model believes most often, so the deterrent
    # frequency stays stable instead of chasing per-frame noise.
    vote_window = deque(maxlen=max(1, args.vote))

    last_seen_time = 0.0
    next_sweep_start_time = 0.0
    current_target_freq = FREQ_DEFAULT
    last_deterrent_class = None
    is_playing = False

    frame_count = 0
    fps_t0 = time.time()
    fps = 0.0
    start_time = time.time()
    prev_report_key = None
    placeholder_sent = False

    while STATE.running:
        frame = cam.read()
        if frame is None:
            # No camera yet. Publish a "waiting" card once so the browser shows
            # the reason instead of a dead black rectangle.
            if not placeholder_sent:
                ph = np.full((args.cam_height, args.cam_width, 3), 24, np.uint8)
                cv2.putText(ph, "NO CAMERA SIGNAL", (28, args.cam_height // 2 - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 240), 2)
                cv2.putText(ph, f"retrying... ({cam.source})",
                            (28, args.cam_height // 2 + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1)
                ok, buf = cv2.imencode(".jpg", ph, encode_params)
                if ok:
                    STATE.publish_frame(buf.tobytes())
                placeholder_sent = True
            STATE.update_stats(camera_ok=False, camera_source=cam.source)
            time.sleep(0.2)
            continue
        placeholder_sent = False
        frame_count += 1

        # Read the live threshold once per frame — the dashboard slider can move
        # it between frames without a restart.
        with STATE.lock:
            conf_now = float(STATE.stats["conf"])

        # --- inference ---
        # Decode at HALF the active threshold so we can show near-misses in the
        # "what the model nearly saw" panel. That is what makes the slider
        # tunable: you can see a 0.34 dog before deciding to drop conf to 0.30.
        t_inf = time.perf_counter()
        tensor, r, pad_l, pad_t = letterbox(frame, size)
        output = sess.run(None, {inp_name: tensor})[0]
        probe = max(0.05, conf_now / 2.0)
        boxes, class_ids, scores, needs_nms = decode(output, probe)
        keep = nms(boxes, scores, args.iou) if needs_nms else list(range(len(scores)))
        infer_ms = (time.perf_counter() - t_inf) * 1000.0

        # --- pick which detection drives the deterrent ---
        deterrent_class = None
        deterrent_conf = -1.0
        report = []
        near_miss = []
        fresh = []
        for idx in keep:
            cname = CLASS_NAMES[class_ids[idx]]
            sc = float(scores[idx])

            # map box back from letterbox space to the original frame
            bx1, by1, bx2, by2 = boxes[idx]
            x1 = int((bx1 - pad_l) / r)
            y1 = int((by1 - pad_t) / r)
            x2 = int((bx2 - pad_l) / r)
            y2 = int((by2 - pad_t) / r)

            if sc < conf_now:
                # Below the real threshold: report it for tuning, draw it faintly,
                # but never let it trigger the deterrent.
                near_miss.append((cname, sc))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)
                cv2.putText(frame, f"?{cname} {sc:.2f}", (x1, max(15, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
                continue

            report.append((cname, sc))
            fresh.append({"box": (x1, y1, x2, y2), "cls": cname,
                          "conf": sc, "ttl": STICKY_FRAMES})
            if cname not in IGNORE_FOR_DETERRENT and sc > deterrent_conf:
                deterrent_conf = sc
                deterrent_class = cname

        # --- persistence: age out old boxes, keep them briefly after they vanish ---
        if STICKY_FRAMES:
            for s in sticky:
                s["ttl"] -= 1
            # A fresh detection of the same class replaces the stale one.
            fresh_classes = {f["cls"] for f in fresh}
            sticky = [s for s in sticky
                      if s["ttl"] > 0 and s["cls"] not in fresh_classes]
            draw_list = fresh + sticky
            sticky = draw_list
        else:
            draw_list = fresh

        for d in draw_list:
            x1, y1, x2, y2 = d["box"]
            cname, sc = d["cls"], d["conf"]
            stale = d["ttl"] < STICKY_FRAMES
            # humans in red (ignored), deterrable animals in green
            color = (0, 0, 255) if cname in IGNORE_FOR_DETERRENT else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if not stale else 1)
            cv2.putText(frame, f"{cname} {sc:.2f}", (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- class-vote smoothing over the last N frames ---
        # Weight each vote by confidence so a confident dog outweighs a marginal goat.
        if args.vote > 1:
            vote_window.append((deterrent_class, deterrent_conf))
            tally = {}
            for c, w in vote_window:
                if c is not None:
                    tally[c] = tally.get(c, 0.0) + max(w, 0.0)
            if tally:
                winner = max(tally, key=tally.get)
                # Only override when the vote disagrees with this frame; the
                # deterrent should follow the window's consensus, not one frame.
                if deterrent_class is not None:
                    deterrent_class = winner
                    deterrent_conf = max(w for c, w in vote_window if c == winner)

        if deterrent_class is not None:
            current_target_freq = pick_target_freq(deterrent_class)
            # Remember it: during the LOST_THRESHOLD grace window the animal is no
            # longer detected, but the sweep it triggered is still the one running.
            # Reporting "None" there was a bug — the frequency belongs to this class.
            last_deterrent_class = deterrent_class

        current_time = time.time()

        # --- AUDIO STATE MACHINE (identical logic to the original) ---
        if deterrent_class is not None:
            last_seen_time = current_time

        # last_deterrent_class gates the branch so a fresh start (last_seen_time=0)
        # cannot fire a sweep before anything has ever been detected.
        if (last_deterrent_class is not None
                and (current_time - last_seen_time) < LOST_THRESHOLD):
            if current_time >= next_sweep_start_time:
                held = deterrent_class or last_deterrent_class
                msg = f"Tracking {held} — {describe_profile(held, audio.rate)}"
                print(msg, flush=True)
                STATE.add_event(msg)
                # Rendered at the device's ACTUAL rate, not the nominal 96 kHz.
                wave = generate_sound_for_class(held, SWEEP_DURATION, audio.rate)
                audio.play(wave)
                next_sweep_start_time = current_time + SWEEP_DURATION
                is_playing = True
        else:
            if is_playing:
                print("Animal has left the area. Stopping deterrent.", flush=True)
                STATE.add_event("Animal left the area — deterrent stopped")
                audio.stop()
                is_playing = False
                next_sweep_start_time = 0.0

        # --- telemetry ---
        if frame_count % 10 == 0:
            now = time.time()
            fps = 10.0 / (now - fps_t0)
            fps_t0 = now

        # log a human sighting once per appearance, not once per frame
        report_key = tuple(sorted(n for n, _ in report))
        if report and report_key != prev_report_key:
            if deterrent_class is None and any(
                    n in IGNORE_FOR_DETERRENT for n, _ in report):
                STATE.add_event("human detected — deterrent suppressed")
        prev_report_key = report_key

        # --- overlay the same FPS badge the desktop window had ---
        cv2.putText(frame, f"{fps:.1f} FPS", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        ok, buf = cv2.imencode(".jpg", frame, encode_params)
        if ok:
            STATE.publish_frame(buf.tobytes())

        STATE.update_stats(
            fps=round(fps, 1),
            infer_ms=round(infer_ms, 1),
            detections=[{"cls": n, "conf": round(c, 3)}
                        for n, c in sorted(report, key=lambda x: -x[1])],
            raw_detections=[{"cls": n, "conf": round(c, 3)}
                            for n, c in sorted(near_miss, key=lambda x: -x[1])[:6]],
            deterrent_active=is_playing,
            deterrent_class=deterrent_class,
            target_freq=current_target_freq if deterrent_class else None,
            frames=frame_count,
            uptime_s=round(current_time - start_time, 1),
            camera_ok=True,
            camera_source=cam.source,
            audio=audio.status,
        )


# --------------------------------------------------------------------------- #
#  HTTP layer                                                                  #
# --------------------------------------------------------------------------- #
PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Animal Deterrent — Live</title>
<style>
  :root{
    --bg:#0f1115; --panel:#171a21; --line:#262b36; --fg:#e6e9ef;
    --muted:#9aa4b6; --accent:#4ade80; --warn:#f87171; --chip:#1f2430;
  }
  @media (prefers-color-scheme: light){
    :root{ --bg:#f4f6fa; --panel:#fff; --line:#e2e6ee; --fg:#131722;
           --muted:#5c6678; --chip:#eef1f7; }
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
  header{padding:16px 20px;border-bottom:1px solid var(--line);
         display:flex;align-items:center;gap:12px;flex-wrap:wrap}
  h1{font-size:17px;margin:0;font-weight:600}
  .dot{width:9px;height:9px;border-radius:50%;background:var(--muted)}
  .dot.live{background:var(--accent);box-shadow:0 0 0 3px rgba(74,222,128,.18)}
  .dot.dead{background:var(--warn);box-shadow:0 0 0 3px rgba(248,113,113,.18)}
  .wrap{display:grid;grid-template-columns:minmax(0,2fr) minmax(260px,1fr);
        gap:16px;padding:16px 20px;max-width:1400px;margin:0 auto}
  @media (max-width:860px){.wrap{grid-template-columns:1fr}}
  .panel{background:var(--panel);border:1px solid var(--line);
         border-radius:12px;overflow:hidden}
  .panel h2{font-size:12px;letter-spacing:.08em;text-transform:uppercase;
            color:var(--muted);margin:0;padding:12px 14px;border-bottom:1px solid var(--line)}
  #video{display:block;width:100%;height:auto;background:#000}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line)}
  .cell{background:var(--panel);padding:12px 14px}
  .k{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
  .v{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums;margin-top:2px}
  .v.sm{font-size:15px;font-weight:500}
  .body{padding:12px 14px}
  .chip{display:inline-flex;align-items:center;gap:6px;background:var(--chip);
        border:1px solid var(--line);border-radius:999px;padding:4px 10px;
        margin:0 6px 6px 0;font-size:13px}
  .chip b{font-variant-numeric:tabular-nums;font-weight:600}
  .chip.human{border-color:var(--warn);color:var(--warn)}
  .banner{padding:10px 14px;font-weight:600;font-size:14px;display:none}
  .banner.on{display:block;background:rgba(74,222,128,.12);color:var(--accent);
             border-bottom:1px solid var(--line)}
  ul.log{list-style:none;margin:0;padding:0;max-height:300px;overflow-y:auto}
  ul.log li{padding:8px 14px;border-bottom:1px solid var(--line);font-size:13px;
            display:flex;gap:10px}
  ul.log li:last-child{border-bottom:0}
  ul.log time{color:var(--muted);font-variant-numeric:tabular-nums;flex:none}
  .muted{color:var(--muted);font-size:13px;padding:12px 14px}
</style>
</head>
<body>
<header>
  <span id="dot" class="dot"></span>
  <h1>Animal Deterrent System</h1>
  <span class="muted" id="modelinfo" style="padding:0"></span>
</header>

<div class="wrap">
  <div>
    <div class="panel">
      <div id="banner" class="banner"></div>
      <img id="video" src="/stream.mjpg" alt="Live camera feed">
    </div>
  </div>

  <div style="display:flex;flex-direction:column;gap:16px">
    <div class="panel">
      <h2>Live stats</h2>
      <div class="grid">
        <div class="cell"><div class="k">Pipeline FPS</div><div class="v" id="fps">—</div></div>
        <div class="cell"><div class="k">Inference</div><div class="v" id="ms">—</div></div>
        <div class="cell"><div class="k">Frames</div><div class="v sm" id="frames">—</div></div>
        <div class="cell"><div class="k">Uptime</div><div class="v sm" id="uptime">—</div></div>
      </div>
    </div>

    <div class="panel">
      <h2>Current detections</h2>
      <div class="body" id="dets"><span class="muted" style="padding:0">Nothing detected</span></div>
    </div>

    <div class="panel">
      <h2>Sensitivity</h2>
      <div class="body">
        <div style="display:flex;align-items:center;gap:10px">
          <input type="range" id="conf" min="5" max="95" step="1" style="flex:1">
          <b id="confval" style="font-variant-numeric:tabular-nums;min-width:2.6em">—</b>
        </div>
        <div class="muted" style="padding:6px 0 0">
          Lower = detects more, but more false alarms. Drag until animals are
          caught reliably.
        </div>
        <div id="nearwrap" style="display:none;margin-top:10px">
          <div class="k" style="margin-bottom:6px">Below threshold (not triggering)</div>
          <div id="near"></div>
          <div class="muted" style="padding:4px 0 0">
            The model sees these but they're under the cutoff. If a real animal
            shows up here, lower the slider past its number.
          </div>
        </div>
      </div>
    </div>

    <div class="panel">
      <h2>Event log</h2>
      <ul class="log" id="log"></ul>
      <div class="muted" id="logempty">No events yet</div>
    </div>

    <div class="panel">
      <h2>System</h2>
      <div class="body">
        <div style="display:flex;justify-content:space-between;padding:3px 0">
          <span class="k">Camera</span><span id="cam">—</span></div>
        <div style="display:flex;justify-content:space-between;padding:3px 0">
          <span class="k">Audio</span><span id="audio">—</span></div>
      </div>
    </div>
  </div>
</div>

<script>
function fmtUptime(s){
  s = Math.floor(s);
  const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), x = s%60;
  return h ? `${h}h ${m}m` : (m ? `${m}m ${x}s` : `${x}s`);
}

async function tick(){
  try{
    const r = await fetch('/state.json', {cache:'no-store'});
    const d = await r.json();

    document.getElementById('dot').className =
      'dot ' + (d.camera_ok ? 'live' : 'dead');
    document.getElementById('fps').textContent = d.fps.toFixed(1);
    document.getElementById('ms').textContent = d.infer_ms.toFixed(0) + ' ms';
    document.getElementById('frames').textContent = d.frames.toLocaleString();
    document.getElementById('uptime').textContent = fmtUptime(d.uptime_s);
    const camEl = document.getElementById('cam');
    camEl.textContent = d.camera_ok ? (d.camera_source || 'streaming')
                                    : 'no signal — retrying';
    camEl.style.color = d.camera_ok ? '' : 'var(--warn)';
    document.getElementById('audio').textContent = d.audio;
    document.getElementById('modelinfo').textContent = d.model + ' @ ' + d.imgsz + 'px';

    const b = document.getElementById('banner');
    if(d.deterrent_active && d.deterrent_class){
      b.className = 'banner on';
      b.textContent = `Deterrent active — ${d.deterrent_class} · sweeping 10kHz → ${d.target_freq}Hz`;
    } else { b.className = 'banner'; }

    const dets = document.getElementById('dets');
    if(d.detections.length){
      dets.innerHTML = d.detections.map(x =>
        `<span class="chip${x.cls==='human'?' human':''}">${x.cls} <b>${x.conf.toFixed(2)}</b></span>`
      ).join('');
    } else {
      dets.innerHTML = '<span class="muted" style="padding:0">Nothing detected</span>';
    }

    // Don't fight the user while they're dragging the slider.
    const cs = document.getElementById('conf');
    if(!cs.matches(':active') && !window.__confDrag){
      cs.value = Math.round(d.conf * 100);
      document.getElementById('confval').textContent = d.conf.toFixed(2);
    }

    const nw = document.getElementById('nearwrap');
    const near = d.raw_detections || [];
    nw.style.display = near.length ? 'block' : 'none';
    if(near.length){
      document.getElementById('near').innerHTML = near.map(x =>
        `<span class="chip" style="opacity:.65">${x.cls} <b>${x.conf.toFixed(2)}</b></span>`
      ).join('');
    }

    const log = document.getElementById('log');
    document.getElementById('logempty').style.display = d.events.length ? 'none' : 'block';
    log.innerHTML = d.events.map(e =>
      `<li><time>${e.t}</time><span>${e.msg}</span></li>`).join('');
  }catch(e){
    document.getElementById('dot').className = 'dot dead';
  }
}
setInterval(tick, 500);
tick();

// --- sensitivity slider -> POST /set_conf (live, no restart) ---
const confEl = document.getElementById('conf');
confEl.addEventListener('input', () => {
  window.__confDrag = true;
  document.getElementById('confval').textContent = (confEl.value/100).toFixed(2);
});
confEl.addEventListener('change', async () => {
  await fetch('/set_conf', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({conf: confEl.value/100})
  });
  window.__confDrag = false;
});

// If the MJPEG stream drops (Pi reboot, wifi blip), reconnect automatically.
const vid = document.getElementById('video');
vid.addEventListener('error', () => {
  setTimeout(() => { vid.src = '/stream.mjpg?t=' + Date.now(); }, 1500);
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    stream_fps = 15.0

    def log_message(self, fmt, *a):
        pass  # keep the console clean for detection output

    def handle_one_request(self):
        """Swallow the traceback a browser causes when it drops an MJPEG stream.

        Closing a tab, reloading, or navigating away resets the connection
        mid-response. socketserver prints a full traceback for that, which is
        pure noise here — and on a headless box it buries the real errors in
        journalctl."""
        try:
            super().handle_one_request()
        except OSError:
            # Covers ConnectionReset/BrokenPipe/ConnectionAborted/timeout — the
            # exact subclass differs by platform, and all of them just mean
            # "the browser went away".
            self.close_connection = True

    def _no_cache(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send_page()
        elif path == "/stream.mjpg":
            self._send_stream()
        elif path == "/state.json":
            self._send_state()
        elif path == "/snapshot.jpg":
            self._send_snapshot()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/set_conf":
            self.send_error(404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            val = float(json.loads(self.rfile.read(n))["conf"])
            val = max(0.05, min(0.95, val))     # clamp to a sane range
            with STATE.lock:
                STATE.stats["conf"] = val
            body = json.dumps({"ok": True, "conf": val}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self.send_error(400, str(e))

    def _send_page(self):
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._no_cache()
        self.end_headers()
        self.wfile.write(body)

    def _send_state(self):
        body = json.dumps(STATE.snapshot()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._no_cache()
        self.end_headers()
        self.wfile.write(body)

    def _send_snapshot(self):
        with STATE.lock:
            jpeg = STATE.jpeg
        if jpeg is None:
            self.send_error(503, "No frame yet")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self._no_cache()
        self.end_headers()
        self.wfile.write(jpeg)

    def _send_stream(self):
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        self.end_headers()
        last_seq = -1
        min_dt = 1.0 / max(self.stream_fps, 1.0)
        try:
            while STATE.running:
                t0 = time.time()
                jpeg, seq = STATE.wait_for_frame(last_seq)
                if jpeg is None or seq == last_seq:
                    continue
                last_seq = seq
                self.wfile.write(b"--FRAME\r\n")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                # throttle so a fast pipeline doesn't saturate the wifi link
                dt = time.time() - t0
                if dt < min_dt:
                    time.sleep(min_dt - dt)
        except OSError:
            pass  # browser tab closed / network blip — normal, not an error


def lan_ip():
    """Best-effort LAN address, for printing a clickable URL on the console."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))   # no packet is sent; just picks the route
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def all_lan_ips():
    """Every non-loopback IPv4 on the box, so a Pi on both wifi and ethernet
    shows both reachable URLs. Falls back to lan_ip() if `ip` isn't available."""
    ips = []
    try:
        import subprocess
        out = subprocess.run(["ip", "-4", "-o", "addr", "show", "scope", "global"],
                             capture_output=True, text=True, timeout=3).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 4:
                iface, cidr = parts[1], parts[3]
                addr = cidr.split("/")[0]
                if not addr.startswith("127."):
                    ips.append((iface, addr))
    except Exception:
        pass
    if not ips:
        ips = [("net", lan_ip())]
    return ips


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()

    global AUDIBLE_MODE, FIT_HARDWARE, AMP_BANDWIDTH_HZ
    AUDIBLE_MODE = args.audible
    FIT_HARDWARE = not args.no_fit_hardware
    AMP_BANDWIDTH_HZ = args.amp_bandwidth

    if args.list_audio:
        import sounddevice as sd
        print(sd.query_devices())
        print("\n" + "=" * 66)
        print("OUTPUT devices you can pass to --audio-device:")
        print("=" * 66)
        alias = ('default', 'sysdefault', 'pulse', 'dmix', 'dsnoop', 'null', 'jack')
        for i, d in enumerate(sd.query_devices()):
            if d.get('max_output_channels', 0) < 1:
                continue
            nm = d['name']
            low = nm.lower()
            tag = ""
            # Order matters: the Pi's HDMI outputs are themselves named
            # "vc4-hdmi-0: MAI PCM i2s-hifi-0", so they contain "i2s". Rule
            # HDMI out FIRST, otherwise it gets tagged as the DAC.
            if any(k in low for k in ('hdmi', 'vc4')):
                tag = "   <- HDMI, NOT the tweeter"
            elif any(k in low for k in ('max98357', 'hifiberry', 'pcm5102',
                                        'pcm512', 'sndrpi', 'iqaudio',
                                        'justboom')):
                tag = "   <== THIS ONE (your I2S DAC)"
            elif low in alias or d['max_output_channels'] > 8:
                tag = "   <- ALSA alias (usually routes to HDMI)"
            print(f"  --audio-device {i:<3} {nm}  "
                  f"({d['max_output_channels']}ch, "
                  f"{d['default_samplerate']:.0f}Hz){tag}")
        print("\nAlso check:  aplay -l")
        print("Test a card directly (bypasses Python entirely):")
        print("  speaker-test -D hw:2,0 -c2 -t sine -f 1000     # card 2 = MAX98357A")
        return

    if args.test_sound:
        audio = AudioPlayer(enabled=True, device=args.audio_device)
        if not audio.enabled:
            print("\nAudio unavailable — cannot test. See the warning above.")
            return
        if AUDIBLE_MODE:
            print("\n--audible: profiles transposed into 1-8kHz for human ears.\n")
        names = ([args.test_sound] if args.test_sound != "all"
                 else list(SOUND_PROFILES.keys()))
        for name in names:
            if name not in SOUND_PROFILES:
                print(f"Unknown class '{name}'. "
                      f"Choose from: {', '.join(SOUND_PROFILES)}")
                continue
            print(f"  {name:6s} {describe_profile(name, audio.rate)}")
            wave = generate_sound_for_class(name, SWEEP_DURATION, audio.rate)
            peak = float(np.abs(wave).max())
            print(f"         {len(wave)} samples, peak {peak:.2f}")
            if not audio.play(wave):
                print("         PLAYBACK FAILED")
            time.sleep(SWEEP_DURATION + 0.6)
        audio.stop()
        print("\nDone. If you heard nothing, the tones may be ultrasonic —"
              "\nre-run with --audible to hear the per-class patterns.")
        return

    model_path = resolve_model(args.model)
    print(f"Loading ONNX model: {model_path}")
    sess = build_session(model_path, args.threads)
    inp = sess.get_inputs()[0]
    inp_name = inp.name
    onnx_size = inp.shape[2] if isinstance(inp.shape[2], int) else 640
    size = args.imgsz or onnx_size
    print(f"Inference size: {size}  (ONNX input {inp.shape})")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Ignored for deterrent: {sorted(IGNORE_FOR_DETERRENT)}")

    audio = AudioPlayer(enabled=not args.no_audio, device=args.audio_device)

    print("\nPer-class deterrent sounds:")
    for cname in SOUND_PROFILES:
        print(f"  {cname:6s} {describe_profile(cname, audio.rate)}")
    if AUDIBLE_MODE:
        print("  (--audible active: transposed into 1-8kHz for testing)")
    print()

    print("Searching for a camera...")
    cam = Camera(args.cam, args.cam_width, args.cam_height, args.camera_backend)
    if not cam.open():
        if cam.source == "camera busy":
            # A USB camera allows exactly one reader, and a leftover copy of
            # this script is by far the most common cause.
            print("\n" + "!" * 62)
            print("  CAMERA IS BUSY — another process is already holding it.")
            print("  Almost always a leftover instance of this script.")
            print("")
            print("      pkill -f rpi5_dashboard.py")
            print("      fuser -v /dev/video0        # shows who holds it")
            print("")
            print("  Then start this one again.")
            print("!" * 62)
        else:
            print(f"\nWARNING: no working camera found (tried index {args.cam} "
                  f"and every /dev/video* node).")
            print("\n  Confirm the camera is present and readable:")
            print("    v4l2-ctl --list-devices")
            print("    v4l2-ctl -d /dev/video0 --stream-mmap --stream-count=1 \\")
            print("             --stream-to=/tmp/f.raw")
            print("\n  CSI ribbon camera instead? That needs libcamera:")
            print("    rpicam-hello --list-cameras")
            print("    sudo apt install -y python3-picamera2")
        print("\n  Starting the dashboard ANYWAY so you can see status in the")
        print("  browser. The camera is retried every 2s automatically.")
    cam.start()

    # Label the model by its family dir (onnx_kaggle/<name>/<size>/<prec>/best.onnx)
    from pathlib import Path as _P
    parts = _P(model_path).parts
    label = parts[-4] if len(parts) >= 4 else _P(model_path).stem
    STATE.update_stats(model=label, imgsz=size, audio=audio.status,
                       conf=args.conf)
    STATE.add_event("System started")

    # Bind the port BEFORE starting worker threads. If another instance already
    # holds it, we must fail cleanly rather than leave a camera thread running
    # against a half-dead process (which aborts the interpreter on exit).
    Handler.stream_fps = args.stream_fps
    # Deliberately NOT setting allow_reuse_address: SO_REUSEADDR would let this
    # bind succeed while another instance still holds the port, leaving two
    # copies fighting over connections AND over the camera. Failing loudly is
    # the correct behaviour — the second instance must not start.
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as e:
        cam.release()
        audio.stop()
        # EADDRINUSE: 98 Linux, 48 BSD/macOS, 10048 Windows
        if getattr(e, "errno", None) in (48, 98, 10048):
            print(f"\nERROR: port {args.port} is already in use.")
            print("Another copy of this script is almost certainly still")
            print("running — it also holds the camera, which is why the")
            print("camera 'cannot be opened'. Stop it with:\n")
            print("    pkill -f rpi5_dashboard.py")
            print(f"    fuser -k {args.port}/tcp\n")
            print(f"Or start this one on a different port:  --port {args.port + 1}")
        else:
            print(f"\nERROR: could not bind {args.host}:{args.port} — {e}")
        raise SystemExit(1)
    server.daemon_threads = True

    det = threading.Thread(
        target=detection_loop,
        args=(args, cam, audio, sess, inp_name, size),
        name="detect", daemon=True)
    det.start()

    ips = all_lan_ips()
    primary = ips[0][1]
    print("\n" + "=" * 66)
    print("  DASHBOARD IS LIVE — open this from your laptop:")
    print("")
    for iface, addr in ips:
        print(f"        http://{addr}:{args.port}          ({iface})")
    print("")
    print(f"  On the Pi itself:  http://localhost:{args.port}")
    print(f"  MJPEG stream:      http://{primary}:{args.port}/stream.mjpg")
    print(f"  JSON state:        http://{primary}:{args.port}/state.json")
    print("=" * 66)
    if not cam.ok:
        print("  NOTE: no camera yet — the page shows 'NO CAMERA SIGNAL' and")
        print("        starts streaming automatically once one is detected.")
    print("  Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        STATE.running = False
        with STATE.cond:
            STATE.cond.notify_all()   # unblock any streaming clients
        server.shutdown()
        cam.release()
        audio.stop()
        print("Done.")


if __name__ == "__main__":
    main()
