"""
rpi5_tflite_speed_all.py
========================
Driver that walks the merged tflite_kaggle/ tree and benchmarks EVERY model on the Pi 5
in one go, into a single CSV. Wraps the same measurement logic as rpi5_tflite_speed.py
(letterbox, INT8 quant-aware input, warmup + timed frames, identical CSV columns) so its
output merges with the ONNX-RT results.

Folder layout expected (as produced by the export notebooks):
    tflite_kaggle/<model>/<size>/<prec>/best_<prec>.tflite
      e.g. tflite_kaggle/yolo26n/320/int8/best_int8.tflite

Install on the Pi (venv):
    pip install tflite-runtime opencv-python-headless numpy   # or full tensorflow

Usage on the Pi:
    python rpi5_tflite_speed_all.py --root tflite_kaggle --img-dir imgs \
        --out rpi5_tflite_all.csv --sizes 320 640 --frames 300 --warmup 50

    # only 320, only int8:
    python rpi5_tflite_speed_all.py --root tflite_kaggle --img-dir imgs \
        --sizes 320 --precisions int8 --out rpi5_tflite_320.csv
"""
import argparse
import csv
import glob
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
    def make_interp(path, threads):
        return tflite.Interpreter(model_path=str(path), num_threads=threads)
except ImportError:
    import tensorflow as tf
    def make_interp(path, threads):
        return tf.lite.Interpreter(model_path=str(path), num_threads=threads)

PARAMS_M = {'yolov8n': 3.01, 'yolov8s': 11.13, 'yolo11n': 2.58, 'yolo11s': 9.42,
            'yolo12n': 2.56, 'yolo12s': 9.08, 'yolo26n': 2.38}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="tflite_kaggle", help="tree of <model>/<size>/<prec>/*.tflite")
    p.add_argument("--img-dir", default="imgs")
    p.add_argument("--sizes", nargs="+", type=int, default=[320, 640])
    p.add_argument("--precisions", nargs="+", default=["fp32", "int8"])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--out", default="rpi5_tflite_all.csv")
    return p.parse_args()


def letterbox(frame, size):
    h0, w0 = frame.shape[:2]
    r = min(size / h0, size / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def bench_one(model_file, model_name, precision, imgsz, imgs_raw, a):
    interp = make_interp(model_file, a.threads)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()
    ishape = inp["shape"]            # (1,H,W,3) NHWC  or  (1,3,H,W) NCHW
    idtype = inp["dtype"]
    scale, zp = inp.get("quantization", (0.0, 0))
    # detect channel layout: NHWC has 3 in the last dim; NCHW has 3 in dim 1
    nchw = (len(ishape) == 4 and ishape[1] == 3)
    size = int(ishape[2]) if nchw else int(ishape[1])

    def prep(frame):
        rgb = letterbox(frame, size)                       # HWC, uint8
        if idtype == np.float32:
            x = rgb.astype(np.float32) / 255.0
        elif scale and scale > 0:
            x = (rgb.astype(np.float32) / 255.0 / scale + zp).astype(idtype)
        else:
            x = rgb.astype(idtype)
        if nchw:
            x = np.transpose(x, (2, 0, 1))                 # HWC -> CHW
        return x[None, ...]                                # add batch dim

    pool = [prep(f) for f in imgs_raw[:max(a.frames, a.warmup)]]
    ix = inp["index"]
    for i in range(a.warmup):
        interp.set_tensor(ix, pool[i % len(pool)]); interp.invoke()

    lat = []
    for i in range(a.frames):
        t0 = time.perf_counter()
        interp.set_tensor(ix, pool[i % len(pool)]); interp.invoke()
        _ = interp.get_tensor(out[0]["index"])
        lat.append((time.perf_counter() - t0) * 1000.0)
    lat = np.asarray(lat)

    return {"model": model_name, "imgsz": imgsz, "precision": precision,
            "runtime": "tflite", "params_M": PARAMS_M.get(model_name),
            "lat_mean_ms": round(float(lat.mean()), 3),
            "lat_p50_ms": round(float(np.percentile(lat, 50)), 3),
            "lat_p95_ms": round(float(np.percentile(lat, 95)), 3),
            "fps": round(1000.0 / lat.mean(), 2),
            "size_mb": round(Path(model_file).stat().st_size / 1e6, 2)}


def main():
    a = parse_args()
    root = Path(a.root)
    assert root.is_dir(), f"root not found: {root}"

    img_paths = sorted(glob.glob(f"{a.img_dir}/*.jpg") + glob.glob(f"{a.img_dir}/*.png"))
    assert img_paths, f"no images in {a.img_dir}"
    imgs_raw = [cv2.imread(p) for p in img_paths[:max(a.frames, a.warmup)]]

    # discover all model files matching the requested sizes/precisions
    jobs = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        for sz in a.sizes:
            for prec in a.precisions:
                f = model_dir / str(sz) / prec / f"best_{prec}.tflite"
                if f.is_file():
                    jobs.append((f, model_dir.name, prec, sz))
    if not jobs:
        raise SystemExit(f"no .tflite files under {root} for sizes={a.sizes} precisions={a.precisions}")

    print(f"benchmarking {len(jobs)} models | frames={a.frames} warmup={a.warmup} threads={a.threads}\n")
    rows = []
    for i, (f, name, prec, sz) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {name} {sz} {prec} ...", flush=True)
        try:
            row = bench_one(str(f), name, prec, sz, imgs_raw, a)
            print(f"    lat mean={row['lat_mean_ms']}ms p95={row['lat_p95_ms']}ms "
                  f"FPS={row['fps']} size={row['size_mb']}MB", flush=True)
            rows.append(row)
        except Exception as e:
            print(f"    FAIL: {e}", flush=True)

    if rows:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {a.out}")
        # quick leaderboard
        print("\nfastest (by FPS):")
        for r in sorted(rows, key=lambda r: -r["fps"])[:5]:
            print(f"  {r['model']:8s} {r['imgsz']} {r['precision']:4s}  {r['fps']:6.1f} FPS")


if __name__ == "__main__":
    main()
