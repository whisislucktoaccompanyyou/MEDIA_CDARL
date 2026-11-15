"""Raw step exports with explicit reference/target pairing and reproducible noise."""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from data.geometry import batch_geometry
from data.refinement_data import read_gray, reference_transform, target_transform, sha256


def save_gray(path, image):
    image = np.asarray(image, dtype=np.uint8)
    cv2.imencode('.png', image)[1].tofile(str(path))


def image_u8(tensor):
    return ((tensor.detach().cpu().squeeze().numpy() + 1) * 127.5).clip(0, 255).round().astype(np.uint8)


def validate_refinement(net, cases, output_dir, timesteps, seed=2026, steps=None,
                        threshold=0.5, metadata=None, baseline_net=None):
    from model.model import CDARL
    if not 0 <= threshold <= 1:
        raise ValueError('mask threshold must be in [0,1], foreground must be positive')
    output = Path(output_dir)
    # No overwrite of any prior result, including another checkpoint's outputs.
    output.mkdir(parents=True, exist_ok=False)
    rng = CDARL.capture_training_rng()
    training = net.training
    baseline_training = baseline_net.training if baseline_net is not None else None
    device = next(net.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    size = net.image_size
    rows = []
    net.eval()
    if baseline_net is not None:
        baseline_net.eval()
    try:
        with torch.no_grad():
            for index, case in enumerate(cases):
                case_id = str(case.get('id', 'case_%03d' % index))
                if not case_id or Path(case_id).name != case_id or case_id in ('.', '..'):
                    raise ValueError('case id must be a simple directory name')
                destination = output / case_id
                destination.mkdir()
                source, raw_mask = read_gray(case['reference']), read_gray(case['mask'])
                reference, reference_meta = reference_transform(source, size)
                mode = case.get('mode', 'independent')
                if mode not in ('independent', 'corresponding'):
                    raise ValueError('mode must be independent or corresponding')
                mask, mask_meta = target_transform(raw_mask, size, reference_meta if mode == 'corresponding' else None)
                scale = 1 if raw_mask.max() <= 1 else 255
                binary = mask.astype(np.float32) / scale >= threshold
                before_fov = int(binary.sum())
                if case.get('training_fov', False):
                    binary &= reference > 10
                target = torch.from_numpy(binary.astype(np.float32) * 2 - 1)[None, None].to(device)
                image = torch.from_numpy(reference.astype(np.float32)/127.5-1)[None, None].to(device)
                geometry = batch_geometry(target)
                noise = torch.randn(image.shape, device=device, generator=generator)
                save_gray(destination / 'reference_original.png', source)
                save_gray(destination / 'reference.png', reference)
                save_gray(destination / 'target_original.png', raw_mask)
                save_gray(destination / 'target_effective.png', binary * 255)
                for c, name in enumerate(('mask', 'boundary', 'skeleton', 'signed_distance')):
                    channel = geometry[0, c].cpu().numpy()
                    save_gray(destination / ('geometry_' + name + '.png'),
                              (channel+1)*127.5 if c == 3 else channel*255)
                for timestep in timesteps:
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                    start = time.perf_counter()
                    result = net.synthesize(image, target, timestep, noise=noise, geometry=geometry, refine_steps=steps)
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - start
                    changed_target = torch.flip(target, [3])
                    changed = net.synthesize(image, changed_target, timestep, noise=noise, refine_steps=steps)['final']
                    changed_reference = torch.flip(image, [2])
                    reference_changed = net.synthesize(changed_reference, target, timestep, noise=noise,
                                                       geometry=geometry, refine_steps=steps)['final']
                    zero = torch.zeros(1, device=device, dtype=torch.long)
                    prediction = net.cycle(result['final'], zero, noise=torch.zeros_like(image))
                    pred, truth = prediction > 0, target > 0
                    dice = float((2*(pred & truth).sum()+1e-6)/(pred.sum()+truth.sum()+1e-6))
                    images = [image_u8(img) for img in result['images']]
                    prefix = 't%d_' % timestep
                    for k, img in enumerate(images):
                        save_gray(destination / (prefix+'I%d.png' % k), img)
                    save_gray(destination / (prefix+'condition_changed.png'), image_u8(changed))
                    save_gray(destination / (prefix+'reference_changed.png'), image_u8(reference_changed))
                    save_gray(destination / (prefix+'cycle.png'), image_u8(prediction))
                    save_gray(destination / (prefix+'steps.png'), np.concatenate(images, axis=1))
                    mid, radius = size//2, size//6
                    crops = [cv2.resize(img[mid-radius:mid+radius, mid-radius:mid+radius],
                                       (size, size), interpolation=cv2.INTER_NEAREST) for img in images]
                    save_gray(destination / (prefix+'detail.png'), np.concatenate(crops, axis=1))
                    baseline = 'not_provided; I0 uses jointly updated weights and is not an independent baseline'
                    if baseline_net is not None:
                        baseline_img = baseline_net.synthesize(image, target, timestep, noise=noise, refine_steps=0)['final']
                        save_gray(destination / (prefix+'original_B.png'), image_u8(baseline_img))
                        baseline = 'separate original G checkpoint; see baseline_weight metadata'
                    rows.append(dict(id=case_id, reference=str(Path(case['reference']).resolve()),
                        mask=str(Path(case['mask']).resolve()), mode=mode,
                        reference_sha256=sha256(case['reference']), mask_sha256=sha256(case['mask']),
                        reference_transform=reference_meta, target_transform=mask_meta,
                        target_pixels_before_fov=before_fov, target_pixels_after_fov=int(binary.sum()),
                        training_fov=bool(case.get('training_fov')), threshold=threshold, seed=seed,
                        timestep=timestep, refine_steps=len(images)-1, resolution=size, device=str(device),
                        elapsed_seconds=elapsed, peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                        internal_cycle_dice=dice, condition_change_mae=float((changed-result['final']).abs().mean()),
                        reference_change_mae=float((reference_changed-result['final']).abs().mean()), baseline=baseline))
    finally:
        net.train(training)
        if baseline_net is not None:
            baseline_net.train(baseline_training)
        CDARL.restore_training_rng(rng)
    report = dict(note='Raw grayscale outputs. Cycle Dice is internal consistency, not independent accuracy. '
                       'No histogram matching or low-frequency replacement postprocessing.',
                  metadata=metadata or {}, cases=rows)
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report
