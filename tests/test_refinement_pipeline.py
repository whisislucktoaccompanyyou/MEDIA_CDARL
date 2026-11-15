import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from core.logger import dict_to_nonedict
from data.geometry import geometry_from_mask, batch_geometry
from data.refinement_data import (holdout_indices, reference_transform,
                                  target_transform, training_sample, read_gray)
from model.ddpm_modules.diffusion_seg import GaussianDiffusion
from model.ddpm_modules.loss import geometry_losses
from model.discriminators import sample_valid_patches
from model.refinement import StructureRefiner

torch.set_num_threads(1)


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x, t):
        return self.conv(x)


class Segmenter(nn.Module):
    label_nc = 2

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 3, padding=1)

    def forward(self, x, condition=None):
        value = self.conv(x)
        return torch.tanh(value if condition is None else value + condition[:, 1:2] * 0.1)


def toy(refinement=True):
    net = GaussianDiffusion(Denoiser(), Segmenter(), image_size=64, channels=1,
        loss_type='l2', refinement={'enabled': refinement, 'channels': 8, 'steps': 3})
    net.set_loss(torch.device('cpu'))
    net.set_new_noise_schedule(dict(schedule='linear', n_timestep=8,
        linear_start=1e-6, linear_end=0.01), torch.device('cpu'))
    return net


def sample():
    target = torch.full((1, 1, 64, 64), -1.)
    target[..., 15:50, 30:33] = 1
    return dict(A=torch.randn_like(target).tanh(), F=target, Cf=batch_geometry(target),
                V=torch.ones_like(target), P=['fake-reference'])


@pytest.mark.parametrize('kind', ['empty', 'full', 'thin'])
def test_geometry_extremes_and_rotation(kind):
    mask = np.zeros((32, 32), bool)
    if kind == 'full':
        mask[:] = True
    if kind == 'thin':
        mask[4:26, 9] = True
    geometry = geometry_from_mask(mask)
    assert geometry.shape == (4, 32, 32) and np.isfinite(geometry).all()
    assert geometry.min() >= -1 and geometry.max() <= 1
    # Thinning can break ties directionally; all channels are recomputed after augmentation.
    np.testing.assert_array_equal(geometry_from_mask(mask.T)[[0, 1, 3]], geometry.transpose(0, 2, 1)[[0, 1, 3]])
    if kind in ('empty', 'full'):
        assert (geometry[3] == (-1 if kind == 'empty' else 1)).all()
    prediction = torch.zeros(1, 1, 32, 32, requires_grad=True)
    losses = geometry_losses(prediction, torch.from_numpy(mask * 2. - 1)[None, None])
    sum(losses).backward()
    assert torch.isfinite(prediction.grad).all()


def test_legacy_path_and_zero_initialization():
    torch.manual_seed(12)
    net = toy(False)
    data = sample()
    noise = torch.randn_like(data['A'])
    timestep = torch.tensor([3])
    noisy = net.q_sample(data['A'], timestep, noise)
    latent = net.denoise_fn(noisy, timestep)
    classes = torch.clamp_min(data['F'][:, 0], 0).long()
    old_condition = torch.eye(2)[:, classes].transpose(0, 1)
    old_output = net.segment_fn(torch.cat([noisy, latent], 1), old_condition)
    result = net.synthesize(data['A'], data['F'], timestep, noise=noise)
    assert torch.equal(result['final'], old_output)
    net.refiner = StructureRefiner(channels=8)
    new = net.synthesize(data['A'], data['F'], timestep, noise=noise)
    assert len(new['images']) == 4
    for image in new['images']:
        torch.testing.assert_close(image, old_output, atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match='legacy C'):
        net.synthesize(data['A'], data['F'], timestep, condition=old_condition)


def test_geometry_gradients_reach_all_refinement_steps():
    net, data = toy(), sample()
    for name, parameter in net.named_parameters():
        parameter.requires_grad_(name.startswith('refiner.'))
    # Nonzero head exercises gradients through early steps as well as the final one.
    nn.init.normal_(net.refiner.head.weight, std=0.01)
    net.path_geometry_weight = 0.25
    _, _, extra = net(data)
    for image in extra['synthesis']['images'][1:]:
        image.retain_grad()
    loss = extra['dice'] + extra['boundary'] + extra['path_dice'] + extra['path_boundary']
    loss.backward()
    assert sum(p.grad.abs().sum() for p in net.refiner.parameters() if p.grad is not None) > 0
    assert all(img.grad is not None and img.grad.abs().sum() > 0 for img in extra['synthesis']['images'][1:])
    assert all(p.grad is None for p in net.denoise_fn.parameters())
    assert extra['path_index'] in (1, 2)


def test_local_sampling_is_identical_and_differentiable():
    real = torch.randn(2, 1, 64, 64)
    fake = real.clone().requires_grad_()
    valid = torch.zeros_like(real)
    valid[..., 8:56, 8:56] = 1
    a, b = sample_valid_patches(real, fake, valid, 32, 2)
    assert a.shape == (4, 1, 32, 32) and torch.equal(a, b)
    b.sum().backward()
    assert fake.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='no local patch'):
        sample_valid_patches(real, fake, valid * 0, 32, 2)


def write_image(path, image):
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode('.png', image)[1].tofile(str(path))


def test_coordinates_augmentation_and_subject_holdout(tmp_path):
    image = np.zeros((50, 90), np.uint8)
    image[5:45, 15:75] = 160
    mask = np.zeros_like(image)
    mask[18:23, 35:37] = 255
    ref, meta = reference_transform(image, 64)
    corresponding, mask_meta = target_transform(mask, 64, meta)
    assert corresponding.shape == ref.shape and mask_meta['matrix'] == meta['matrix']
    independent, transform = target_transform(mask, 64)
    import cv2
    assert transform['mode'] == 'full_domain' and transform['matrix'][0][0] == 64/90
    np.testing.assert_array_equal(independent, cv2.resize(mask, (64, 64), interpolation=cv2.INTER_NEAREST))
    ip, mp = tmp_path/'参考图.png', tmp_path/'分形.png'
    write_image(ip, image)
    write_image(mp, mask)
    for seed in range(8):
        np.random.seed(seed)
        result = training_sample(ip, mp, 64)
        np.testing.assert_array_equal(result['Cf'].numpy(), geometry_from_mask(result['F'][0].numpy() > 0))
        assert not ((result['F'] > 0) & (result['V'] == 0)).any()
    paths = ['1_left.png', '1_right.png', '2_left.png', '2_right.png', '3_left.png']
    indices = holdout_indices(paths, 2)
    assert (0 in indices) == (1 in indices) and (2 in indices) == (3 in indices)


def tiny_options(tmp_path):
    config = json.loads(Path('config/train_kaggle_refine_v1.json').read_text())
    config['synthesis']['timesteps'] = [3]
    config['gpu_ids'], config['distributed'] = [], False
    config['path'].update(checkpoint=str(tmp_path), pretrained=None, resume_state=None)
    config['model']['diffusion']['image_size'] = 64
    config['model']['beta_schedule']['train']['n_timestep'] = 8
    config['model']['refinement'].update(channels=8, patch_size=32, warmup_steps=1, allow_from_scratch=True)
    return dict_to_nonedict(config)


def test_staged_updates_and_exact_checkpoint_resume(tmp_path, monkeypatch):
    import model.networks as networks
    from model.model import CDARL
    monkeypatch.setattr(networks, 'define_G', lambda opt: toy())
    monkeypatch.setattr(networks, 'define_D', lambda opt: nn.Conv2d(1, 1, 3, padding=1))
    config = tiny_options(tmp_path)
    model = CDARL(config)
    data = sample()
    before = {k: v.clone() for k, v in model.netG.denoise_fn.state_dict().items()}
    model.feed_data(copy.deepcopy(data))
    model.optimize_parameters()
    assert model.stage == 'warmup' and model.log_dict['R_grad_norm'] > 0
    assert all(torch.equal(v, before[k]) for k, v in model.netG.denoise_fn.state_dict().items())
    assert all(p.grad is None for p in model.netD.parameters())
    model.save_network(1, 1)
    model.feed_data(copy.deepcopy(data))
    model.optimize_parameters()
    expected = {k: v.clone() for k, v in model.netG.state_dict().items()}
    assert model.stage == 'joint'
    assert any(not torch.equal(v, before[k]) for k, v in model.netG.denoise_fn.state_dict().items())
    resume_config = copy.deepcopy(config)
    resume_config['path']['resume_state'] = str(tmp_path/'I1_E1')
    restored = CDARL(resume_config)
    assert restored.begin_step == 1
    restored.feed_data(copy.deepcopy(data))
    restored.optimize_parameters()
    for key, value in restored.netG.state_dict().items():
        torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
    assert restored.log_dict == model.log_dict


def test_monitor_reproducibility_rng_and_no_overwrite(tmp_path):
    from model.model import CDARL
    from refinement_validation import validate_refinement
    image = np.full((64, 64), 150, np.uint8)
    mask = np.zeros_like(image)
    mask[15:50, 27:30] = 255
    ip, mp = tmp_path/'reference.png', tmp_path/'mask.png'
    write_image(ip, image)
    write_image(mp, mask)
    cases = [dict(id='example', reference=str(ip), mask=str(mp), mode='independent')]
    net = toy()
    net.denoise_fn.eval()
    state = CDARL.capture_training_rng()
    first = validate_refinement(net, cases, str(tmp_path/'one'), [3], seed=42)
    draws = (random.random(), np.random.random(), torch.rand(()).item())
    CDARL.restore_training_rng(state)
    assert draws == (random.random(), np.random.random(), torch.rand(()).item())
    second = validate_refinement(net, cases, str(tmp_path/'two'), [3], seed=42)
    assert first['cases'][0]['internal_cycle_dice'] == second['cases'][0]['internal_cycle_dice']
    assert first['metadata']['model_state_sha256'] == second['metadata']['model_state_sha256']
    assert len(first['metadata']['model_state_sha256']) == 64
    assert (tmp_path/'one/example/t3_I3.png').read_bytes() == (tmp_path/'two/example/t3_I3.png').read_bytes()
    assert net.training
    assert not net.denoise_fn.training
    with pytest.raises(FileExistsError):
        validate_refinement(net, cases, str(tmp_path/'one'), [3])


def test_sensitivity_uses_another_reference_and_target(tmp_path):
    from refinement_validation import validate_refinement
    first, second = tmp_path/'first.png', tmp_path/'second.png'
    mask_a, mask_b = tmp_path/'mask_a.png', tmp_path/'mask_b.png'
    write_image(first, np.full((64, 64), 120, np.uint8))
    write_image(second, np.full((64, 64), 190, np.uint8))
    mask = np.zeros((64, 64), np.uint8)
    mask[12:45, 19:21] = 255
    write_image(mask_a, mask)
    write_image(mask_b, mask.T)
    cases = [dict(id='a', reference=str(first), mask=str(mask_a)),
             dict(id='b', reference=str(second), mask=str(mask_b))]
    report = validate_refinement(toy(), cases, str(tmp_path/'results'), [3])
    row = report['cases'][0]
    assert row['reference_change']['kind'] == 'another_reference'
    assert row['condition_change']['kind'] == 'another_target'
    assert row['reference_change']['path'] == str(second.resolve())
    np.testing.assert_array_equal(
        read_gray(tmp_path/'results/a/t3_reference_changed_input.png'),
        np.full((64, 64), 190, np.uint8))


def test_real_spade_and_unet_use_shared_synthesis(tmp_path):
    from model.networks import define_G
    config = tiny_options(Path('.'))
    config['phase'] = 'test'
    config['model']['unet'].update(inner_channel=32, channel_multiplier=[1, 2], attn_res=[16], res_blocks=1)
    net = define_G(config)
    net.set_loss(torch.device('cpu'))
    net.set_new_noise_schedule(config['model']['beta_schedule']['train'], torch.device('cpu'))
    net.eval()
    for name, parameter in net.named_parameters():
        parameter.requires_grad_(name.startswith('refiner.'))
    output, losses, extra = net(sample())
    assert output[4].shape == (1, 1, 64, 64)
    assert all(torch.isfinite(loss) for loss in losses)
    assert len(extra['synthesis']['images']) == 4
    assert output[4].min() >= -1 and output[4].max() <= 1
    (extra['dice'] + extra['boundary']).backward()
    assert net.refiner.head.weight.grad.abs().sum() > 0
    from test_refinement import load_generator
    torch.save(net.state_dict(), tmp_path/'actual_G.pth')
    loaded = load_generator(config, str(tmp_path/'actual'), torch.device('cpu'))
    data = sample()
    with torch.no_grad():
        first = net.synthesize(data['A'], data['F'], 3, noise=torch.zeros_like(data['A']))['final']
        restored = loaded.synthesize(data['A'], data['F'], 3, noise=torch.zeros_like(data['A']))['final']
    torch.testing.assert_close(first, restored, atol=0, rtol=0)


def test_strict_original_weight_migration(tmp_path, monkeypatch):
    import model.networks as networks
    from model.model import CDARL
    monkeypatch.setattr(networks, 'define_G', lambda opt: toy())
    monkeypatch.setattr(networks, 'define_D', lambda opt: nn.Conv2d(1, 1, 3, padding=1))
    config = tiny_options(tmp_path)
    original = CDARL(config)
    state = {k: v for k, v in original.netG.state_dict().items() if not k.startswith('refiner.')}
    torch.save(state, tmp_path/'original_G.pth')
    torch.save(original.netH.state_dict(), tmp_path/'original_H.pth')
    config['model']['refinement']['allow_from_scratch'] = False
    config['path']['pretrained'] = str(tmp_path/'original')
    migrated = CDARL(config)
    assert migrated.begin_step == 0
    assert all(k.startswith('refiner.') for k in migrated.migration_report['G']['missing'])
    assert migrated.migration_report['H']['missing'] == []
    assert torch.count_nonzero(migrated.netG.refiner.head.weight) == 0
    state['denoise_fn.conv.weight'] = torch.zeros(9)
    torch.save(state, tmp_path/'original_G.pth')
    with pytest.raises(ValueError, match='incompatible migration'):
        CDARL(config)


def test_attention_mapping_preserves_original_parameter_graph():
    from model.ddpm_modules.unet import UNet
    original = json.loads(Path('config/train_kaggle.json').read_text())
    refined = json.loads(Path('config/train_kaggle_refine_v1.json').read_text())
    def shapes(config):
        opt = config['model']['unet']
        with torch.device('meta'):
            net = UNet(in_channel=opt['in_channel'], out_channel=opt['out_channel'],
                       inner_channel=opt['inner_channel'], channel_mults=opt['channel_multiplier'],
                       attn_res=opt['attn_res'], res_blocks=opt['res_blocks'], dropout=opt['dropout'],
                       image_size=config['model']['diffusion']['image_size'])
        return {key: tuple(value.shape) for key, value in net.state_dict().items()}
    assert shapes(original) == shapes(refined)


def test_training_runner_mid_epoch_resume(tmp_path, monkeypatch):
    import model.networks as networks
    import refinement_training as training
    monkeypatch.setattr(networks, 'define_G', lambda opt: toy())
    monkeypatch.setattr(networks, 'define_D', lambda opt: nn.Conv2d(1, 1, 3, padding=1))
    config = tiny_options(tmp_path)
    source = tmp_path/'images'
    dataset_root = tmp_path/'dataset'
    for i in range(4):
        write_image(source/('%d_left.png' % i), np.full((64, 64), 130+i*10, np.uint8))
    mask = np.zeros((64, 64), np.uint8)
    mask[15:40, 30:32] = 255
    write_image(dataset_root/'train/trainA/fractal.png', mask)
    (dataset_root/'split_manifest.json').write_text(json.dumps({'train': ['%d_left.png' % i for i in range(4)]}))
    config['datasets']['train'].update(dataroot=str(dataset_root), source_dir=str(source),
        quality_csv=None, quality_weights=None, image_size=64, monitor_count=1)
    quality = tmp_path/'quality.csv'
    quality.write_text(',image,quality\n' + '\n'.join('%d,%d_left.png,%d' % (i, i, i % 2) for i in range(4)))
    config['datasets']['train'].update(quality_csv=str(quality), quality_weights={'0': 1.0, 'default': 0.3})
    config['train'].update(max_steps=3, print_freq=1)
    def set_output(root):
        root.mkdir()
        config['path'].update(experiments_root=str(root), checkpoint=str(root/'checkpoint'), results=str(root/'results'))
    set_output(tmp_path/'full')
    training.run(copy.deepcopy(config))
    complete = torch.load(tmp_path/'full/checkpoint/I3_E1_refine.pth', weights_only=False)
    set_output(tmp_path/'interrupted')
    original_validate = training.validate_refinement
    def interrupt(*args, **kwargs):
        raise RuntimeError('simulated interruption after first completed batch')
    monkeypatch.setattr(training, 'validate_refinement', interrupt)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        training.run(copy.deepcopy(config))
    config['path']['resume_state'] = str(tmp_path/'interrupted/checkpoint/I1_E1')
    set_output(tmp_path/'resumed')
    monkeypatch.setattr(training, 'validate_refinement', original_validate)
    training.run(copy.deepcopy(config))
    resumed = torch.load(tmp_path/'resumed/checkpoint/I3_E1_refine.pth', weights_only=False)
    assert resumed['epoch_position'] == complete['epoch_position'] == 3
    for module in ('G', 'H', 'D', 'DL'):
        for key in complete[module]:
            torch.testing.assert_close(resumed[module][key], complete[module][key], atol=0, rtol=0)
    full_transforms = [json.loads(line) for line in (tmp_path/'full/training_transforms.jsonl').read_text(encoding='utf-8').splitlines()]
    resumed_transforms = [json.loads(line) for line in (tmp_path/'resumed/training_transforms.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(full_transforms) == 3 and resumed_transforms == full_transforms[1:]
    assert 'matrix' in full_transforms[0]['reference']
    assert 'augmentation' in full_transforms[0]
