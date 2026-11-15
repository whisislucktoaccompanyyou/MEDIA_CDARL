"""Single-device staged training with reproducible, complete mid-epoch resume."""
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch

import data as Data
import model as Model
from data.refinement_data import write_monitor_manifest
from refinement_validation import validate_refinement


def run(opt):
    logger = logging.getLogger('base')
    ds_opt = opt['datasets']['train']
    if ds_opt.get('num_workers', 0) != 0:
        raise ValueError('refinement v1 requires num_workers=0 to restore augmentation/sampler RNG exactly')
    if ds_opt.get('condition_dir'):
        raise ValueError('legacy conditions are forbidden; use this batch fractal F only')
    size = opt['model']['diffusion']['image_size']
    if ds_opt.get('image_size') != size or ds_opt['batch_size'] != 1 or not ds_opt.get('refinement'):
        raise ValueError('refinement v1 requires batch=1 and matching explicit model/dataset image_size')
    seed = opt['train'].get('seed', 2026)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dataset = Data.create_dataset_xcad(ds_opt, 'train')
    loader = Data.create_dataloader(dataset, ds_opt, 'train')
    if not len(loader):
        raise ValueError('empty refinement training set')
    root = Path(opt['path']['experiments_root'])
    (root / 'config_snapshot.json').write_text(json.dumps(opt, ensure_ascii=False, indent=2), encoding='utf-8')
    cases = write_monitor_manifest(dataset, root / 'monitor_manifest.json')
    diffusion = Model.create_model(opt)
    if hasattr(diffusion, 'migration_report'):
        (root / 'migration_report.json').write_text(json.dumps(diffusion.migration_report, indent=2), encoding='utf-8')
    parameters = sum(p.numel() for p in diffusion.netG.refiner.parameters())
    logger.info('Refinement parameters=%d, device=%s, train=%d, holdout=%d; NCE masks resized to 256',
                parameters, diffusion.device, len(dataset), len(cases))
    start_epoch = diffusion.begin_epoch or 1
    step = diffusion.begin_step
    limit = int(opt['train']['max_steps'])
    epoch = start_epoch
    resume_position = getattr(diffusion, 'epoch_position', 0)
    checkpoint_rng = diffusion.capture_training_rng()
    log_path = root / 'refinement_metrics.jsonl'
    safe_to_save = True
    try:
        while step < limit and epoch <= int(opt['train']['n_epoch']):
            if resume_position and diffusion.epoch_rng is not None:
                diffusion.restore_training_rng(diffusion.epoch_rng)
            else:
                diffusion.epoch_rng = diffusion.capture_training_rng()
            iterator = iter(loader)
            # Recreate the same sampler permutation and skip already-consumed batches,
            # then restore RNG from the checkpoint (including model stochastic draws).
            for _ in range(resume_position):
                next(iterator)
            if resume_position:
                diffusion.restore_training_rng(checkpoint_rng)
            position = resume_position
            resume_position = 0
            for sample in iterator:
                if diffusion.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(diffusion.device)
                    torch.cuda.synchronize(diffusion.device)
                start = time.perf_counter()
                safe_to_save = False
                diffusion.feed_data(sample)
                diffusion.optimize_parameters()
                safe_to_save = True
                step += 1
                position += 1
                diffusion.epoch_position = position
                if diffusion.device.type == 'cuda':
                    torch.cuda.synchronize(diffusion.device)
                logs = dict(diffusion.get_current_log())
                logs.update(step=step, epoch=epoch, stage=diffusion.stage,
                            seconds=time.perf_counter()-start,
                            peak_cuda_bytes=torch.cuda.max_memory_allocated(diffusion.device) if diffusion.device.type == 'cuda' else None)
                with log_path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(logs) + '\n')
                if step == 1 or step % int(opt['train']['print_freq']) == 0:
                    logger.info('Refinement: %s', logs)
                boundary = int(opt['model']['refinement']['warmup_steps'])
                if step % int(opt['train']['save_checkpoint_freq']) == 0 or step == boundary:
                    diffusion.save_network(epoch, step)
                synth = opt['synthesis']
                if step == 1 or step % int(synth['frequency']) == 0:
                    validate_refinement(diffusion.netG, cases,
                        str(Path(opt['path']['results']) / ('step_%07d' % step)),
                        synth['timesteps'], synth['seed'], metadata=dict(step=step, stage=diffusion.stage, config=opt))
                if step >= limit:
                    break
            if position == len(loader) and step < limit:
                epoch += 1
                diffusion.epoch_position = 0
            elif step < limit:
                raise RuntimeError('unexpected dataloader exhaustion')
    finally:
        # Save the last completed optimization step, including stage/R/D_local/RNG.
        if safe_to_save:
            diffusion.save_network(epoch, step)
    logger.info('Refinement training finished at step=%d epoch=%d; quality requires independent evaluation', step, epoch)
