#!/usr/bin/env python3
"""Prepare grayscale fundus images and fractal vessel masks for C-DARL.

Does not start training. Original DomainA_UWF / DomainB_FP files are not modified.
"""
import json
import os
import sys
import traceback

import cv2
import numpy as np

ROOT = "/mnt/diskb5/ruining_private/Sunennan/伪标签转回医学图像"
UWF = os.path.join(ROOT, "data", "DomainA_UWF")
FP = os.path.join(ROOT, "data", "DomainB_FP")
DEST = os.path.join(ROOT, "MEDIA_CDARL", "data", "Dataset_Fundus", "train")
TRAIN_B = os.path.join(DEST, "trainB")
TRAIN_A = os.path.join(DEST, "trainA")
REPORT = os.path.join(ROOT, "MEDIA_CDARL", "data", "Dataset_Fundus", "prepare_report.json")
SIZE = 512
N_FRACTAL = 1000
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def log(msg):
    print(msg, flush=True)


def list_images(path):
    names = []
    for name in os.listdir(path):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTS:
            names.append(name)
    return sorted(names)


def to_uint8_gray(image):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype == np.uint8:
        return image
    image = image.astype(np.float32)
    peak = float(image.max())
    if peak <= 0:
        return np.zeros(image.shape, dtype=np.uint8)
    # 16-bit fundus TIFFs are not 0-255. Scale by the file peak, not by 255.
    return np.clip(image / peak * 255.0, 0, 255).astype(np.uint8)


def fractal_mask(rng, size=SIZE):
    """Binary fractal vessel-like mask: background 0, vessel 255."""
    canvas = np.zeros((size, size), dtype=np.float32)
    n_trees = int(rng.integers(2, 5))
    for _ in range(n_trees):
        y = float(rng.integers(size // 5, 4 * size // 5))
        x = float(rng.integers(size // 5, 4 * size // 5))
        angle = float(rng.uniform(0, 2 * np.pi))
        length = float(rng.uniform(size * 0.12, size * 0.28))
        width = float(rng.uniform(2.2, 4.5))
        _branch(canvas, rng, x, y, angle, length, width, depth=0, max_depth=6)
    mask = (canvas > 0).astype(np.uint8) * 255
    yy, xx = np.ogrid[:size, :size]
    disk = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 <= (size * 0.46) ** 2
    mask[~disk] = 0
    return mask


def _branch(canvas, rng, x, y, angle, length, width, depth, max_depth):
    if depth > max_depth or length < 4 or width < 0.4:
        return
    steps = max(int(length), 1)
    for _ in range(steps):
        x += np.cos(angle)
        y += np.sin(angle)
        if not (0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]):
            return
        cv2.circle(canvas, (int(x), int(y)), max(int(round(width)), 1), 1, -1)
        if rng.random() < 0.04:
            angle += float(rng.normal(0, 0.15))
    n_child = 2 if rng.random() < 0.85 else 1
    for _ in range(n_child):
        turn = float(rng.normal(0, 0.45))
        _branch(
            canvas,
            rng,
            x,
            y,
            angle + turn,
            length * float(rng.uniform(0.62, 0.82)),
            width * float(rng.uniform(0.62, 0.84)),
            depth + 1,
            max_depth,
        )


def main():
    os.makedirs(TRAIN_A, exist_ok=True)
    os.makedirs(TRAIN_B, exist_ok=True)
    uwf_names = list_images(UWF)
    fp_names = list_images(FP)
    log("UWF images: %d" % len(uwf_names))
    log("FP images: %d" % len(fp_names))
    if not uwf_names or not fp_names:
        raise RuntimeError("missing fundus images")

    written = 0
    sizes = {}
    for domain, src, names in (("UWF", UWF, uwf_names), ("FP", FP, fp_names)):
        for i, name in enumerate(names):
            src_path = os.path.join(src, name)
            raw = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise RuntimeError("unreadable image: %s" % src_path)
            key = "%s:%s:%s" % (domain, str(raw.shape), str(raw.dtype))
            sizes[key] = sizes.get(key, 0) + 1
            gray = to_uint8_gray(raw)
            if gray.shape != (SIZE, SIZE):
                gray = cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            out_name = "%s_%04d.png" % (domain, i)
            if not cv2.imwrite(os.path.join(TRAIN_B, out_name), gray):
                raise RuntimeError("failed to write %s" % out_name)
            written += 1
            if written % 100 == 0:
                log("wrote images %d" % written)
    log("trainB written: %d" % written)

    rng = np.random.default_rng(20241101)
    vessel_frac = []
    for i in range(N_FRACTAL):
        mask = fractal_mask(rng)
        out_name = "fractal_%04d.png" % i
        if not cv2.imwrite(os.path.join(TRAIN_A, out_name), mask):
            raise RuntimeError("failed to write %s" % out_name)
        vessel_frac.append(float((mask > 0).mean()))
        if (i + 1) % 200 == 0:
            log("wrote masks %d" % (i + 1))

    report = {
        "uwf_count": len(uwf_names),
        "fp_count": len(fp_names),
        "trainB_count": written,
        "trainA_count": N_FRACTAL,
        "image_size": SIZE,
        "channels": 1,
        "vessel_value": 255,
        "background_value": 0,
        "raw_shapes": sizes,
        "fractal_vessel_fraction_mean": float(np.mean(vessel_frac)),
        "fractal_vessel_fraction_min": float(np.min(vessel_frac)),
        "fractal_vessel_fraction_max": float(np.max(vessel_frac)),
        "trainB": TRAIN_B,
        "trainA": TRAIN_A,
        "training_started": False,
    }
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log("report written")
    log(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
