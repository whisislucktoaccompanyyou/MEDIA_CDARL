"""Train a Lab a/b head on a frozen C-DARL generator.

The generator weights are loaded and not updated. Luminance still comes from
the existing SPADE output. This script does not call the grayscale training
loop and does not overwrite that checkpoint.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import core.logger as Logger
import data as Data
import model as Model
from model.color_head import ColorHead


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config/train_kaggle.json')
    parser.add_argument(
        '--weight',
        default='experiments/CDARL_kaggle_quality_261110_223239/checkpoint/I110000_E5',
    )
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--preview_every', type=int, default=500)
    parser.add_argument('--preview_dir', default='test_input')
    return parser.parse_args()


def load_frozen_generator(opt, weight):
    opt['phase'] = 'test'
    opt['path']['resume_state'] = weight
    if not os.path.isfile(weight + '_G.pth'):
        raise RuntimeError('missing generator weight {}'.format(weight + '_G.pth'))
    diffusion = Model.create_model(opt)
    net = diffusion.netG.module if isinstance(diffusion.netG, torch.nn.DataParallel) else diffusion.netG
    net.eval()
    for param in net.parameters():
        param.requires_grad = False
    net.segment_fn.return_features = True
    return net


def bgr_to_lab_ab(image_bgr):
    """image_bgr: uint8 HxWx3. Returns a,b float32 in 0-255, OpenCV Lab scale."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    return lab[:, :, 1].astype(np.float32), lab[:, :, 2].astype(np.float32)


def fundus_mask(gray_u8):
    blur = cv2.GaussianBlur(gray_u8, (0, 0), 3)
    foreground = blur > 12
    if int(foreground.sum()) < 32:
        return np.ones(gray_u8.shape, dtype=np.float32)
    yy, xx = np.nonzero(foreground)
    cy, cx = float(np.median(yy)), float(np.median(xx))
    radius = float(np.percentile(np.hypot(xx - cx, yy - cy), 99.5)) - 1.5
    grid_y, grid_x = np.ogrid[: gray_u8.shape[0], : gray_u8.shape[1]]
    return ((grid_x - cx) ** 2 + (grid_y - cy) ** 2 <= max(radius, 1.0) ** 2).astype(np.float32)


class ColorDataset(torch.utils.data.Dataset):
    """Same crop and subsample as XCADDataset, plus Lab a/b from the color file."""

    def __init__(self, dataroot, source_dir):
        with open(os.path.join(dataroot, 'split_manifest.json')) as handle:
            manifest = json.load(handle)
        self.paths = [os.path.join(source_dir, name) for name in manifest['train']]
        self.fractal_paths = sorted(
            name for name in os.listdir(os.path.join(dataroot, 'train', 'trainA'))
            if name.endswith('.png')
        )
        self.fractal_dir = os.path.join(dataroot, 'train', 'trainA')
        if not self.paths or not self.fractal_paths:
            raise RuntimeError('empty color training list or fractal masks')

    def __len__(self):
        return len(self.paths)

    def _crop_square(self, image, gray):
        foreground = gray > 10
        ys, xs = np.where(foreground)
        if len(xs):
            top, bottom = ys.min(), ys.max() + 1
            left, right = xs.min(), xs.max() + 1
            image = image[top:bottom, left:right]
            gray = gray[top:bottom, left:right]
        height, width = gray.shape
        side = max(height, width)
        pad = ((side - height) // 2, (side - height + 1) // 2, (side - width) // 2, (side - width + 1) // 2)
        gray = cv2.copyMakeBorder(gray, *pad, cv2.BORDER_CONSTANT, value=0)
        image = cv2.copyMakeBorder(image, *pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        gray = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        return image, gray

    def __getitem__(self, index):
        path = self.paths[index]
        color = cv2.imread(path, cv2.IMREAD_COLOR)
        if color is None or color.ndim != 3:
            raise RuntimeError('color source is not a 3-channel image: {}'.format(path))
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        color, gray = self._crop_square(color, gray)
        opt1, opt2 = np.random.randint(0, 2), np.random.randint(0, 2)
        gray = gray[opt1::2, opt2::2]
        color = color[opt1::2, opt2::2]
        a, b = bgr_to_lab_ab(color)
        field = fundus_mask(gray)
        fractal_name = self.fractal_paths[np.random.randint(0, len(self.fractal_paths))]
        fractal = cv2.imread(os.path.join(self.fractal_dir, fractal_name), cv2.IMREAD_GRAYSCALE)
        fractal = cv2.resize(fractal, (512, 512), interpolation=cv2.INTER_NEAREST)
        fractal = fractal[opt1::2, opt2::2]
        luminance = torch.from_numpy(gray.astype(np.float32) / 255.0 * 2.0 - 1.0).unsqueeze(0)
        target = torch.from_numpy(np.stack([a, b], axis=0))
        field = torch.from_numpy(field).unsqueeze(0)
        fractal = torch.from_numpy((fractal > 127).astype(np.float32))
        return {'A': luminance, 'AB': target, 'field': field, 'F': fractal, 'P': path}


def lab_to_bgr(luminance_tanh, ab):
    """luminance_tanh: 1xHxW in -1..1. ab: 2xHxW in 0..255."""
    ell = ((luminance_tanh + 1.0) * 0.5 * 255.0).clamp(0, 255)
    lab = torch.cat([ell.unsqueeze(0), ab.clamp(0, 255)], dim=0)
    array = lab.detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return cv2.cvtColor(array, cv2.COLOR_LAB2BGR)


def synthesize_feature(net, image, mask, timestep, generator=None):
    """Same B path as synthesis_validation: noise the image, condition on the mask."""
    classes = torch.from_numpy((mask >= 0.5).astype(np.int64)).long().unsqueeze(0).to(image.device)
    condition = torch.nn.functional.one_hot(classes, 2).permute(0, 3, 1, 2).float()
    noise = torch.randn(image.shape, generator=generator, device=image.device)
    t = torch.full((1,), int(timestep), device=image.device, dtype=torch.long)
    noisy = net.q_sample(image, t, noise=noise)
    latent = net.denoise_fn(noisy, t)
    return net.segment_fn(torch.cat([noisy, latent], dim=1), condition)


def save_preview(net, head, preview_dir, out_dir, device, step):
    if not os.path.isdir(preview_dir):
        return
    names = sorted(
        name for name in os.listdir(preview_dir) if name.endswith('_image.png')
    )[:4]
    os.makedirs(out_dir, exist_ok=True)
    for name in names:
        stem = name[:-len('_image.png')]
        mask_path = os.path.join(preview_dir, stem + '_mask.png')
        gray = cv2.imread(os.path.join(preview_dir, name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.isfile(mask_path) else None
        if gray is None or mask is None:
            continue
        gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        tensor = torch.from_numpy(gray.astype(np.float32) / 255.0 * 2.0 - 1.0)
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            luminance, feature = synthesize_feature(net, tensor, mask > 127, 200)
            ab = head(feature)
        bgr = lab_to_bgr(luminance[0, 0], ab[0])
        cv2.imwrite(os.path.join(out_dir, '{}_step{:06d}.png'.format(stem, step)), bgr)


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cli = argparse.Namespace(config=args.config, phase='test', gpu_ids=args.gpu, debug=False)
    opt = Logger.dict_to_nonedict(Logger.parse(cli))
    opt['distributed'] = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = load_frozen_generator(opt, args.weight)
    head = ColorHead().to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr, betas=(0.5, 0.999))

    dataset_opt = opt['datasets']['train']
    dataset = ColorDataset(dataset_opt['dataroot'], dataset_opt['source_dir'])
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True
    )

    experiment = os.path.join('experiments', 'color_head_{}'.format(Logger.get_timestamp()))
    checkpoint = os.path.join(experiment, 'checkpoint')
    preview = os.path.join(experiment, 'preview')
    os.makedirs(checkpoint, exist_ok=True)
    print('experiment', experiment, 'steps', args.steps, 'images', len(dataset), flush=True)

    head.train()
    step = 0
    while step < args.steps:
        for batch in loader:
            luminance = batch['A'].to(device)
            target = batch['AB'].to(device)
            field = batch['field'].to(device)
            with torch.no_grad():
                timestep = torch.randint(0, net.num_timesteps, (1,), device=device)
                fractal = batch['F'].numpy()[0] > 0.5
                synthetic, feature = synthesize_feature(net, luminance, fractal, int(timestep))
            predicted = head(feature)
            weight = field.expand_as(predicted)
            loss = (predicted - target).abs().mul(weight).sum() / weight.sum().clamp_min(1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step == 1 or step % 100 == 0:
                print('step', step, 'l_ab', float(loss), flush=True)
            if step % args.preview_every == 0:
                save_preview(net, head, args.preview_dir, preview, device, step)
            if step % args.save_every == 0 or step == args.steps:
                path = os.path.join(checkpoint, 'color_head_{:06d}.pth'.format(step))
                torch.save({'step': step, 'head': head.state_dict(), 'opt': optimizer.state_dict()}, path)
                print('saved', path, flush=True)
            if step >= args.steps:
                break
    print('done', experiment, flush=True)


if __name__ == '__main__':
    sys.exit(main())
