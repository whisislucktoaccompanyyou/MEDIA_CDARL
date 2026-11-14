"""Offline vessel, optic-disc, and fundus-circle maps for the 4-channel condition.

Does not enter the training loop. Writes one PNG per training photo, same stem,
into data/Dataset_Kaggle/conditions/. A missing vessel checkpoint falls back to
a black-hat vessel map and records that source in the report.

Channels stored as one RGB PNG plus a disc PNG, so nothing is packed into a
fourth file the loader has to guess:
    vessels/<stem>.png   vessel pseudo-mask, 0/255
    disc/<stem>.png       optic disc, 0/255
    circle/<stem>.png     fundus circle, 0/255
"""
import argparse
import json
import os

import cv2
import numpy as np


def crop_square(gray, color=None, threshold=10):
    foreground = gray > threshold
    ys, xs = np.where(foreground)
    if len(xs):
        top, bottom = int(ys.min()), int(ys.max()) + 1
        left, right = int(xs.min()), int(xs.max()) + 1
        gray = gray[top:bottom, left:right]
        if color is not None:
            color = color[top:bottom, left:right]
    height, width = gray.shape[:2]
    side = max(height, width)
    pad = (
        (side - height) // 2,
        (side - height + 1) // 2,
        (side - width) // 2,
        (side - width + 1) // 2,
    )
    gray = cv2.copyMakeBorder(gray, *pad, cv2.BORDER_CONSTANT, value=0)
    if color is not None:
        color = cv2.copyMakeBorder(color, *pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return gray, color


def fundus_circle(gray, threshold=12):
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    foreground = blur > threshold
    if int(foreground.sum()) < 32:
        return np.ones(gray.shape, dtype=np.uint8) * 255
    yy, xx = np.nonzero(foreground)
    cy, cx = float(np.median(yy)), float(np.median(xx))
    radius = float(np.percentile(np.hypot(xx - cx, yy - cy), 99.5)) - 1.5
    grid_y, grid_x = np.ogrid[: gray.shape[0], : gray.shape[1]]
    circle = (grid_x - cx) ** 2 + (grid_y - cy) ** 2 <= max(radius, 1.0) ** 2
    return (circle.astype(np.uint8) * 255)


def blackhat_vessels(gray, circle):
    blur = cv2.GaussianBlur(gray, (0, 0), 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return np.where(circle > 0, mask, 0).astype(np.uint8)


def optic_disc(gray, circle):
    """Bright compact peak inside the fundus circle. Empty if none is found."""
    field = circle > 0
    if int(field.sum()) < 32:
        return np.zeros(gray.shape, dtype=np.uint8)
    blur = cv2.GaussianBlur(gray, (0, 0), 5)
    values = blur[field]
    cutoff = float(np.percentile(values, 98.5))
    peak = (blur >= cutoff) & field
    peak = cv2.morphologyEx(peak.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(peak, connectivity=8)
    if count <= 1:
        return np.zeros(gray.shape, dtype=np.uint8)
    height, width = gray.shape
    best = None
    best_score = -1.0
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < 40:
            continue
        component = labels == index
        yy, xx = np.nonzero(component)
        cy, cx = float(yy.mean()), float(xx.mean())
        # A disc is bright and not a thin vessel streak.
        ys, xs = component.nonzero()
        box_h = ys.max() - ys.min() + 1
        box_w = xs.max() - xs.min() + 1
        fill = area / float(box_h * box_w)
        if fill < 0.35 or min(box_h, box_w) / float(max(box_h, box_w)) < 0.4:
            continue
        # Prefer the upper-nasal side of a typical fundus photo, but do not require it.
        score = float(blur[component].mean()) + 0.15 * (1.0 - cy / height)
        if cx < width * 0.15 or cx > width * 0.85:
            score -= 0.2
        if score > best_score:
            best_score = score
            best = component
    if best is None:
        return np.zeros(gray.shape, dtype=np.uint8)
    disc = np.zeros(gray.shape, dtype=np.uint8)
    disc[best] = 255
    return disc


def load_vessel_model(checkpoint, device):
    if not checkpoint:
        return None
    import torch
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    blob = torch.load(checkpoint, map_location=device)
    state = blob.get("state_dict", blob.get("model", blob)) if isinstance(blob, dict) else blob
    if not isinstance(state, dict):
        raise RuntimeError("vessel checkpoint has no state_dict; pass a torch segmentation weight")
    # A generic loader cannot invent an architecture. Callers that have one
    # should set VESSEL_ARCH. Otherwise the black-hat fallback is used.
    arch = os.environ.get("VESSEL_ARCH")
    if not arch:
        return None
    module_name, class_name = arch.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_name)
    model = getattr(module, class_name)()
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def vessel_from_model(model, gray, size, device):
    import torch
    tensor = torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None].to(device)
    with torch.no_grad():
        pred = model(tensor)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if pred.shape[1] > 1:
            pred = pred[:, 1:2]
        pred = torch.nn.functional.interpolate(pred, size=(size, size), mode="bilinear", align_corners=False)
        binary = (pred[0, 0] > 0.5).cpu().numpy()
    return (binary.astype(np.uint8) * 255)


def write_png(path, image):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not cv2.imwrite(path, image):
        raise RuntimeError("cannot write " + path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="../data/kaggle/extracted/train")
    parser.add_argument("--manifest", default="data/Dataset_Kaggle/split_manifest.json")
    parser.add_argument("--out", default="data/Dataset_Kaggle/conditions")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--vessel-checkpoint", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with open(args.manifest) as handle:
        names = json.load(handle)["train"]
    if args.limit:
        names = names[: args.limit]
    device = "cpu"
    model = None
    source_name = "blackhat"
    try:
        import torch
        # This process must already have CUDA_VISIBLE_DEVICES=1, so cuda:0 is physical GPU 1.
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = load_vessel_model(args.vessel_checkpoint, device)
        if model is not None:
            source_name = "checkpoint:" + args.vessel_checkpoint
    except Exception as exc:
        print("vessel model unavailable, using black-hat:", exc, flush=True)
        model = None
        source_name = "blackhat"

    written = 0
    for name in names:
        stem = os.path.splitext(name)[0]
        color = cv2.imread(os.path.join(args.source, name), cv2.IMREAD_COLOR)
        if color is None:
            print("skip unreadable", name, flush=True)
            continue
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        gray, _ = crop_square(gray)
        gray = cv2.resize(gray, (args.size, args.size), interpolation=cv2.INTER_AREA)
        circle = fundus_circle(gray)
        if model is None:
            vessel = blackhat_vessels(gray, circle)
        else:
            vessel = vessel_from_model(model, gray, args.size, device)
            vessel = np.where(circle > 0, vessel, 0).astype(np.uint8)
        disc = optic_disc(gray, circle)
        write_png(os.path.join(args.out, "vessels", stem + ".png"), vessel)
        write_png(os.path.join(args.out, "disc", stem + ".png"), disc)
        write_png(os.path.join(args.out, "circle", stem + ".png"), circle)
        written += 1
        if written % 500 == 0:
            print("wrote", written, flush=True)
    report = {
        "written": written,
        "size": args.size,
        "vessel_source": source_name,
        "note": "Disc and circle come from brightness and contour rules, not from labels.",
    }
    with open(os.path.join(args.out, "report.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
