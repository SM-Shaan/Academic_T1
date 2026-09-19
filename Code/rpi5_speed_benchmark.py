"""
rpi5_speed_benchmark.py
=======================
Measure HARDWARE speed metrics for all exported ONNX models ON THE RASPBERRY PI 5:
latency (mean / p50 / p95 ms), throughput (FPS), model size (MB), and param count.

These are the genuinely hardware-specific numbers for the thesis (accuracy is
hardware-independent — get that from bench_accuracy_onnx.py). Runs on pure
onnxruntime, so no torch/ultralytics needed on the Pi.

Protocol (matches thesis methodology / FINDINGS.md §8):
  * 50-frame warmup (Pi 5 CPU throttles ~2 min into sustained load; warmup stabilises it)
  * time N frames of sess.run(); report mean/p50/p95 ms and FPS = 1000/mean
  * format-independent: works for both raw-head (1,4+nc,N) and embedded-NMS (1,300,6) ONNX

Params (M) are read from the sibling accuracy CSV if present, else left blank (params
are a property of the model, not the runtime — no need to recompute on the Pi).

Usage on the Pi (venv active):
    python rpi5_speed_benchmark.py --imgsz 320            # all models @ 320
    python rpi5_speed_benchmark.py --imgsz 320 --frames 500
    python rpi5_speed_benchmark.py --models yolo26n yolo11n

Output: rpi5_speed_<imgsz>.csv
"""

import argparse
import csv
import glob
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ALL_MODELS = ['yolov8n', 'yolov8s', 'yolo11n', 'yolo11s', 'yolo12n', 'yolo12s', 'yolo26n']

# param counts (M) — a model property, same everywhere. From your thesis table.
PARAMS_M = {'yolov8n': 3.01, 'yolov8s': 11.13, 'yolo11n': 2.58, 'yolo11s': 9.42,
            'yolo12n': 2.56, 'yolo12s': 9.08, 'yolo26n': 2.38}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx-dir", default="onnx_kaggle")
    p.add_argument("--img-dir", default="merged_dataset_v3/test/images",
                   help="Folder of real images to stream (representative input)")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--models", nargs="*", default=ALL_MODELS)
    p.add_argument("--precision", default="fp32", choices=["fp32", "int8"])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--frames", type=int, default=300, help="timed frames after warmup")
    p.add_argument("--out", default=None)
    # --- single-file mode (for the space-constrained Pi: one model at a time) ---
    p.add_argument("--model-file", default=None,
                   help="Path to ONE best.onnx to benchmark (bypasses --onnx-dir).")
    p.add_argument("--model-name", default=None,
                   help="Name to record for --model-file (e.g. yolo26n).")
    p.add_argument("--append", action="store_true",
                   help="Append to --out instead of overwriting (build the 7-row CSV "
                        "across separate per-model runs).")
    return p.parse_args()


def letterbox(frame, size):
    h0, w0 = frame.shape[:2]
    r = min(size / h0, size / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)[None, ...]


def build_session(path, threads):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    provs = ort.get_available_providers()
    ep = (['XnnpackExecutionProvider', 'CPUExecutionProvider']
          if 'XnnpackExecutionProvider' in provs else ['CPUExecutionProvider'])
    return ort.InferenceSession(str(path), sess_options=opts, providers=ep)


def bench(onnx_path, images, size, threads, warmup, frames):
    sess = build_session(onnx_path, threads)
    iname = sess.get_inputs()[0].name
    # pre-letterbox a pool of frames so we time inference, not disk/resize
    pool = [letterbox(cv2.imread(p), size) for p in images[:max(frames, warmup)]]
    if not pool:
        raise RuntimeError("no images found to benchmark")

    for i in range(warmup):
        sess.run(None, {iname: pool[i % len(pool)]})

    lat = []
    for i in range(frames):
        t0 = time.perf_counter()
        sess.run(None, {iname: pool[i % len(pool)]})
        lat.append((time.perf_counter() - t0) * 1000.0)
    lat = np.asarray(lat)
    return {
        "lat_mean_ms": round(float(lat.mean()), 3),
        "lat_p50_ms": round(float(np.percentile(lat, 50)), 3),
        "lat_p95_ms": round(float(np.percentile(lat, 95)), 3),
        "fps": round(1000.0 / lat.mean(), 2),
        "size_mb": round(onnx_path.stat().st_size / 1e6, 2),
    }


def main():
    args = parse_args()
    out_csv = args.out or f"rpi5_speed_{args.imgsz}.csv"
    images = sorted(glob.glob(f"{args.img_dir}/*.jpg") + glob.glob(f"{args.img_dir}/*.png"))
    print(f"providers available: {ort.get_available_providers()}")
    print(f"benchmark images: {len(images)} | imgsz {args.imgsz} | "
          f"warmup {args.warmup} | timed {args.frames}")

    # Single-file mode (space-constrained Pi): one (path, name) pair.
    if args.model_file:
        targets = [(args.model_name or Path(args.model_file).stem, Path(args.model_file))]
    else:
        targets = [(n, Path(args.onnx_dir) / n / str(args.imgsz) / args.precision / "best.onnx")
                   for n in args.models]

    rows = []
    for name, onnx_path in targets:
        if not onnx_path.exists():
            print(f"SKIP {name}: {onnx_path} missing")
            continue
        print(f"\n=== {name} ===")
        try:
            res = bench(onnx_path, images, args.imgsz, args.threads,
                        args.warmup, args.frames)
            res = {"model": name, "imgsz": args.imgsz, "precision": args.precision,
                   "params_M": PARAMS_M.get(name), **res}
            print(f"  lat mean={res['lat_mean_ms']}ms p50={res['lat_p50_ms']}ms "
                  f"p95={res['lat_p95_ms']}ms | FPS={res['fps']} | size={res['size_mb']}MB")
            rows.append(res)
        except Exception as e:
            print(f"  FAIL {name}: {e}")

    if rows:
        # append mode: write header only if the file is new/empty
        write_header = True
        if args.append and Path(out_csv).exists() and Path(out_csv).stat().st_size > 0:
            write_header = False
        mode = "a" if args.append else "w"
        with open(out_csv, mode, newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                w.writeheader()
            w.writerows(rows)
        print(f"\n{'appended to' if args.append else 'wrote'} {out_csv} ({len(rows)} row(s))")


if __name__ == "__main__":
    main()
