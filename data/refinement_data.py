"""Full-FOV transforms and a deterministic, subject-grouped unlabeled holdout."""
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch

from data.geometry import geometry_from_mask


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def model_state_sha256(net):
    """Fingerprint current tensors even when step 1 has no checkpoint file yet."""
    digest = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str((value.dtype, tuple(value.shape))).encode('ascii'))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def read_gray(path):
    # imdecode supports non-ASCII Windows paths, unlike some cv2.imread builds.
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError('cannot read image: ' + str(path))
    return image


def reference_transform(image, size):
    ys, xs = np.where(image > 10)
    h0, w0 = image.shape
    x, y, w, h = (int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1),
                   int(ys.max()-ys.min()+1)) if len(xs) else (0, 0, w0, h0)
    side = max(w, h)
    left, top = (side-w)//2, (side-h)//2
    square = np.zeros((side, side), image.dtype)
    square[top:top+h, left:left+w] = image[y:y+h, x:x+w]
    scale = size / side
    # Matrix maps pixel centers, matching OpenCV's resize coordinate convention.
    matrix = [[scale, 0, (left-x+0.5)*scale-0.5],
              [0, scale, (top-y+0.5)*scale-0.5], [0, 0, 1]]
    meta = dict(original_shape=[h0, w0], crop=[x, y, w, h],
                padding=[left, top, side-w-left, side-h-top], size=size, matrix=matrix)
    return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA), meta


def target_transform(mask, size, reference_meta=None):
    """Never crop an independent target to its vessel bounding box."""
    if reference_meta is None:
        h, w = mask.shape
        sx, sy = size/w, size/h
        return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST), {
            'mode': 'full_domain', 'original_shape': [h, w],
            'matrix': [[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]]}
    if list(mask.shape) != reference_meta['original_shape']:
        raise ValueError('corresponding mask and reference must have identical source dimensions')
    x, y, w, h = reference_meta['crop']
    left, top, right, bottom = reference_meta['padding']
    cropped = cv2.copyMakeBorder(mask[y:y+h, x:x+w], top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_NEAREST), dict(
        mode='corresponding', **reference_meta)


def subject_key(path):
    return re.sub(r'[_-](left|right|os|od|l|r)$', '', Path(path).stem, flags=re.I)


def holdout_indices(paths, count=16, seed=2026):
    groups = {}
    for i, path in enumerate(paths):
        groups.setdefault(subject_key(path), []).append(i)
    if len(groups) < 2:
        raise ValueError('at least two subject groups required for an unlabeled holdout')
    keys = sorted(groups)
    np.random.default_rng(seed).shuffle(keys)
    held = []
    for key in keys[:-1]:
        held.extend(groups[key])
        if len(held) >= count:
            break
    return sorted(held)


def training_sample(image_path, mask_path, size, augment=True):
    image, transform = reference_transform(read_gray(image_path), size)
    raw = read_gray(mask_path)
    mask, mask_transform = target_transform(raw, size)
    mask = mask > (0 if raw.max() <= 1 else 127)
    valid = image > 10
    original_pixels = int(mask.sum())
    mask &= valid
    augmentation = np.eye(3)
    if augment:
        # Derive geometry only after the exact same augmentation of A/F/FOV.
        for axis in (0, 1):
            if np.random.random() < 0.5:
                image, mask, valid = [np.flip(a, axis=axis) for a in (image, mask, valid)]
                flip = np.eye(3)
                flip[axis ^ 1, axis ^ 1] = -1
                flip[axis ^ 1, 2] = size-1
                augmentation = flip @ augmentation
        if np.random.random() < 0.5:
            image, mask, valid = [a.T for a in (image, mask, valid)]
            augmentation = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]) @ augmentation
    tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().unsqueeze(0)
    return {'A': tensor(image / 127.5 - 1), 'F': tensor(mask * 2.0 - 1),
            'Cf': torch.from_numpy(geometry_from_mask(mask)), 'V': tensor(valid),
            'P': str(image_path), 'mask_path': str(mask_path),
            'transform': json.dumps(dict(reference=transform, target=mask_transform,
                augmentation=augmentation.tolist(), target_pixels_before_fov=original_pixels,
                target_pixels_after_fov=int(mask.sum())))}


def write_monitor_manifest(dataset, output):
    cases = []
    for i, image in enumerate(dataset.monitor_paths):
        mask = dataset.F_paths[i % len(dataset.F_paths)]
        cases.append(dict(id='monitor_%03d' % i, reference=str(Path(image).resolve()),
                          mask=str(Path(mask).resolve()), mode='independent', training_fov=True,
                          reference_sha256=sha256(image), mask_sha256=sha256(mask)))
    document = dict(seed=dataset.monitor_seed, cases=cases,
                    train_subjects=sorted({subject_key(p) for p in dataset.A_paths}),
                    monitor_subjects=sorted({subject_key(p) for p in dataset.monitor_paths}))
    Path(output).write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding='utf-8')
    return cases
