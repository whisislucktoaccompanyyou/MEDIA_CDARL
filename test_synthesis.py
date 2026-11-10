"""Synthesize fundus images from uploaded image/mask pairs.

This is not official `main.py -p test`. Official test computes vessel
segmentation Dice. Synthesis here follows the training B path in
GaussianDiffusion.p_losses: noise the grayscale fundus image, predict the
noise, then run the SPADE generator conditioned on the vessel mask.
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch

import core.logger as Logger
import model as Model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config/train_fundus.json')
    parser.add_argument('--input_dir', default='test_input')
    parser.add_argument('--output_dir', default='test_output')
    parser.add_argument(
        '--weight',
        default='experiments/CDARL_fundus_UWF_FP_261102_111517/checkpoint/I151050_E150',
    )
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--timesteps', type=int, nargs='+', default=[200,800,1600])
    return parser.parse_args()


def find_pairs(input_dir):
    images = sorted(glob.glob(os.path.join(input_dir, '*_image.png')))
    pairs = []
    missing = []
    for image_path in images:
        stem = os.path.basename(image_path)[:-len('_image.png')]
        mask_path = os.path.join(input_dir, stem + '_mask.png')
        if os.path.isfile(mask_path):
            pairs.append((stem, image_path, mask_path))
        else:
            missing.append(stem)
    return pairs, missing


def read_gray(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError('cannot read {}'.format(path))
    return image


def to_model_tensor(image, size=256):
    # Training reads 512 grayscale, divides by 255, then randomly subsamples
    # by 2 to 256. Test uses one deterministic resize to that same size.
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
    tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(0).unsqueeze(0)


def save_gray(path, image):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not cv2.imwrite(path, image):
        raise RuntimeError('cannot write {}'.format(path))


def tensor_to_u8(tensor):
    array = tensor.detach().float().cpu().squeeze().numpy()
    array = (array + 1.0) * 0.5
    array = np.clip(array, 0.0, 1.0)
    return np.round(array * 255.0).astype(np.uint8)


def panel(original, mask, synthetic):
    height, width = synthetic.shape
    original = cv2.resize(original, (width, height), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    gap = np.full((height, 8), 128, dtype=np.uint8)
    return np.concatenate([original, gap, mask, gap, synthetic], axis=1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pairs, missing = find_pairs(args.input_dir)
    if not pairs:
        raise RuntimeError('no *_image.png / *_mask.png pairs in {}'.format(args.input_dir))
    if missing:
        print('missing masks:', ', '.join(missing))

    weight = args.weight
    if not os.path.isfile(weight + '_G.pth'):
        raise RuntimeError('missing weight {}'.format(weight + '_G.pth'))

    cli = argparse.Namespace(
        config=args.config,
        phase='test',
        gpu_ids=args.gpu,
        debug=False,
    )
    opt = Logger.dict_to_nonedict(Logger.parse(cli))
    opt['phase'] = 'test'
    opt['path']['resume_state'] = weight
    opt['distributed'] = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffusion = Model.create_model(opt)
    net = diffusion.netG.module if isinstance(diffusion.netG, torch.nn.DataParallel) else diffusion.netG
    net.eval()
    print('loaded', weight + '_G.pth', 'device', device)
    print('pairs', len(pairs))

    from synthesis_validation import validate
    validate(net, args.input_dir, args.output_dir, args.timesteps, args.seed)
    print('saved', args.output_dir)


if __name__ == '__main__':
    sys.exit(main())
