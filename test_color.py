"""Colorize grayscale images with a trained color head. Does not train."""

import argparse
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import torch

import core.logger as Logger
from model.color_head import ColorHead
from train_color import fundus_mask, lab_to_bgr, load_frozen_generator, synthesize_feature


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config/train_kaggle.json')
    parser.add_argument(
        '--weight',
        default='experiments/CDARL_kaggle_quality_261110_223239/checkpoint/I110000_E5',
    )
    parser.add_argument('--head', required=True, help='color_head_XXXXXX.pth')
    parser.add_argument('--input_dir', default='test_input')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--timestep', type=int, default=200)
    parser.add_argument('--seed', type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cli = argparse.Namespace(config=args.config, phase='test', gpu_ids=args.gpu, debug=False)
    opt = Logger.dict_to_nonedict(Logger.parse(cli))
    opt['distributed'] = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = load_frozen_generator(opt, args.weight)
    head = ColorHead().to(device)
    state = torch.load(args.head, map_location=device)
    head.load_state_dict(state['head'])
    head.eval()

    if not args.output_dir:
        args.output_dir = os.path.join(
            'test_output_color', datetime.now().strftime('%Y%m%d_%H%M%S')
        )
    names = sorted(name for name in os.listdir(args.input_dir) if name.endswith('_image.png'))
    if not names:
        raise RuntimeError('no *_image.png in {}'.format(args.input_dir))
    generator = torch.Generator(device=device).manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    for name in names:
        stem = name[:-len('_image.png')]
        gray = cv2.imread(os.path.join(args.input_dir, name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(args.input_dir, stem + '_mask.png'), cv2.IMREAD_GRAYSCALE)
        if gray is None or mask is None:
            raise RuntimeError('missing image/mask pair {}'.format(stem))
        gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        tensor = torch.from_numpy(gray.astype('float32') / 255.0 * 2.0 - 1.0)
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            luminance, feature = synthesize_feature(
                net, tensor, mask > 127, args.timestep, generator
            )
            ab = head(feature)
        bgr = lab_to_bgr(luminance[0, 0], ab[0])
        outside = fundus_mask(gray) < 0.5
        bgr[np.repeat(outside[:, :, None], 3, axis=2)] = 0
        out = os.path.join(args.output_dir, stem + '_color.png')
        if not cv2.imwrite(out, bgr):
            raise RuntimeError('cannot write {}'.format(out))
        print('saved', out, flush=True)


if __name__ == '__main__':
    sys.exit(main())
