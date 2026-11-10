"""Deterministic vessel-conditioned synthesis; never modifies training RNG state."""
import os, json
import cv2
import numpy as np
import torch
from test_synthesis import find_pairs, read_gray, to_model_tensor, tensor_to_u8, save_gray

# Previous result folders. A new test must not write back into these.
FORBIDDEN_OUTPUTS = {
    "kaggle_all_100k",
    "kaggle_eyeq_q0_100k（8k优质图像）",
}


def mask_classes(mask, size=256, threshold=0.5):
    values = mask.astype(np.float32)
    if values.max() > 1:
        values /= 255.0
    binary = (values >= threshold).astype(np.uint8)
    return cv2.resize(binary, (size, size), interpolation=cv2.INTER_NEAREST)


def panel(images):
    height, width = images[0].shape
    gap = np.full((height, 8), 128, dtype=np.uint8)
    parts = []
    for image in images:
        if image.shape != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        parts.append(image)
        parts.append(gap)
    return np.concatenate(parts[:-1], axis=1)


def fov_mask(image, threshold=8):
    return image > threshold


def histogram_match(source, reference, fov):
    """Monotonic CDF match. Changes brightness only, not pixel rank."""
    if int(fov.sum()) < 10:
        return source.copy()
    source_hist = np.bincount(source[fov].ravel(), minlength=256).astype(np.float64)
    reference_hist = np.bincount(reference[fov].ravel(), minlength=256).astype(np.float64)
    if source_hist.sum() == 0 or reference_hist.sum() == 0:
        return source.copy()
    source_cdf = np.cumsum(source_hist)
    reference_cdf = np.cumsum(reference_hist)
    source_cdf /= source_cdf[-1]
    reference_cdf /= reference_cdf[-1]
    lut = np.interp(source_cdf, reference_cdf, np.arange(256)).astype(np.uint8)
    return lut[source]


def disc_shape(image, threshold=8):
    fov = (image > threshold).astype(np.uint8)
    if int(fov.sum()) < 10:
        return None, None
    contours, _ = cv2.findContours(fov * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    circularity = 4.0 * np.pi * area / (perimeter * perimeter + 1e-6)
    bulge = None
    if len(contour) >= 5 and area > 0:
        ellipse = np.zeros_like(fov)
        cv2.ellipse(ellipse, cv2.fitEllipse(contour), 1, -1)
        bulge = float(np.logical_and(fov > 0, ellipse == 0).sum()) / float(fov.sum())
    return float(circularity), bulge


def appearance(synthetic, original, mask):
    fov = fov_mask(original)
    vessel = mask > 0
    background = np.logical_and(fov, np.logical_not(vessel))
    vessel_fov = np.logical_and(fov, vessel)
    values = {}
    values["mean"] = float(synthetic.mean())
    if int(fov.sum()) > 10:
        left = synthetic[fov].astype(np.float32)
        right = original[fov].astype(np.float32)
        values["fov_mae"] = float(np.abs(left - right).mean())
        values["fov_mean_gap"] = float(left.mean() - right.mean())
        if left.std() > 1 and right.std() > 1:
            values["fov_corr"] = float(np.corrcoef(left, right)[0, 1])
    if int(vessel_fov.sum()) > 10 and int(background.sum()) > 10:
        values["vessel_contrast"] = float(synthetic[vessel_fov].mean() - synthetic[background].mean())
    gradient_x = cv2.Sobel(synthetic, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(synthetic, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    edge = cv2.Canny((vessel.astype(np.uint8) * 255), 50, 150)
    band = cv2.dilate(edge, np.ones((5, 5), np.uint8)) > 0
    outside = np.logical_and(fov, np.logical_not(band))
    if int(band.sum()) > 10 and int(outside.sum()) > 10:
        values["edge_grad_ratio"] = float(magnitude[band].mean() / (magnitude[outside].mean() + 1e-6))
    circularity, bulge = disc_shape(synthetic)
    values["disc_circularity"] = circularity
    values["disc_bulge"] = bulge
    return values


def prefixed(metrics, prefix):
    return {prefix + key: value for key, value in metrics.items()}


def summarize(report):
    summary = []
    timesteps = sorted({row["timestep"] for row in report})
    numeric = [key for key, value in report[0].items() if isinstance(value, float)]
    for timestep in timesteps:
        rows = [row for row in report if row["timestep"] == timestep]
        means = {}
        for key in numeric:
            values = [row[key] for row in rows if row.get(key) is not None]
            if values:
                means[key] = float(sum(values) / len(values))
        summary.append({"timestep": timestep, "n": len(rows), "mean": means})
    return summary


def print_summary(summary):
    keys = [
        "raw_mean", "raw_fov_mae", "raw_fov_mean_gap", "raw_vessel_contrast",
        "raw_edge_grad_ratio", "raw_disc_circularity", "raw_disc_bulge",
        "matched_mean", "matched_fov_mae", "matched_fov_mean_gap",
        "matched_vessel_contrast", "matched_edge_grad_ratio",
        "matched_disc_circularity", "matched_disc_bulge",
        "condition_change_mae",
    ]
    for block in summary:
        print("t=%d n=%d" % (block["timestep"], block["n"]))
        for key in keys:
            if key in block["mean"]:
                print("  %s=%.4f" % (key, block["mean"][key]))


def validate(net, input_dir, output_dir, timesteps, seed=2026, match=False):
    if os.path.basename(os.path.abspath(output_dir)) in FORBIDDEN_OUTPUTS:
        raise RuntimeError("refusing to write into an existing result folder: " + output_dir)
    os.makedirs(output_dir, exist_ok=True)
    pairs, missing = find_pairs(input_dir)
    if not pairs or missing:
        raise RuntimeError("Missing synthesis image/mask pairs: " + str(missing))
    device = next(net.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    training = net.training
    net.eval()
    report = []
    try:
        with torch.no_grad():
            for stem, ip, mp in pairs:
                original, mask = read_gray(ip), read_gray(mp)
                if original.shape != mask.shape:
                    raise ValueError("Image/mask shape mismatch: " + stem)
                image = to_model_tensor(original).to(device)
                classes = torch.from_numpy(mask_classes(mask)).long().unsqueeze(0).to(device)
                condition = torch.nn.functional.one_hot(classes, 2).permute(0, 3, 1, 2).float()
                noise = torch.randn(image.shape, generator=generator, device=device)
                original_256 = cv2.resize(original, (256, 256), interpolation=cv2.INTER_AREA)
                mask_256 = mask_classes(mask)
                reference_circularity, _ = disc_shape(original_256)
                for t in timesteps:
                    if not 0 <= t < net.num_timesteps:
                        raise ValueError("Invalid timestep")
                    timestep = torch.full((1,), t, device=device, dtype=torch.long)
                    noisy = net.q_sample(image, timestep, noise=noise)
                    latent = net.denoise_fn(noisy, timestep)
                    features = torch.cat([noisy, latent], 1)
                    synthetic = net.segment_fn(features, condition)
                    # Identical reference/noise, changed condition: measures whether mask is used.
                    changed = net.segment_fn(features, torch.flip(condition, [3]))
                    reconstructed = net.segment_fn(torch.cat([
                        synthetic, net.denoise_fn(synthetic, torch.zeros_like(timestep))
                    ], 1))
                    pred = reconstructed[:, 0] > 0
                    target = classes.bool()
                    dice = (2 * (pred & target).sum().float() + 1e-6) / (pred.sum() + target.sum() + 1e-6)
                    u8 = tensor_to_u8(synthetic)
                    raw_metrics = appearance(u8, original_256, mask_256)
                    matched = histogram_match(u8, original_256, fov_mask(original_256)) if match else None
                    matched_metrics = appearance(matched, original_256, mask_256) if matched is not None else {}
                    prefix = os.path.join(output_dir, stem + "_t" + str(t))
                    save_gray(prefix + "_synthetic.png", u8)
                    columns = [original, mask_256 * 255, u8]
                    if matched is not None:
                        save_gray(prefix + "_matched.png", matched)
                        columns.append(matched)
                    save_gray(prefix + "_panel.png", panel(columns))
                    row = dict(
                        id=stem,
                        timestep=t,
                        seed=seed,
                        mask_fraction=float(target.float().mean()),
                        synthetic_mean=float(u8.mean()),
                        condition_change_mae=float((synthetic - changed).abs().mean()),
                        cycle_dice=float(dice),
                        orig_disc_circularity=reference_circularity,
                    )
                    row.update(prefixed(raw_metrics, "raw_"))
                    row.update(prefixed(matched_metrics, "matched_"))
                    report.append(row)
    finally:
        net.train(training)
    summary = summarize(report) if report else []
    note = (
        "cycle_dice uses the same trained segmenter and is not an accuracy measure. "
        "raw_* is the model output. matched_* is histogram matching inside the original FOV; "
        "it does not change weights. disc_circularity closer to orig_disc_circularity is rounder; "
        "disc_bulge is the fraction of the disc outside its fitted ellipse."
    )
    with open(os.path.join(output_dir, "synthesis_report.json"), "w") as handle:
        json.dump({"note": note, "match": match, "summary": summary, "cases": report}, handle, indent=2)
    print_summary(summary)
    return report
