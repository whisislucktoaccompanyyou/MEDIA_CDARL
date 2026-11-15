"""Reference + target inference. Directory mode pairs explicitly by identical stem."""
import argparse
import copy
import json
from pathlib import Path

import torch

from core.logger import dict_to_nonedict
from data.refinement_data import sha256
from model.networks import define_G
from refinement_validation import validate_refinement


def load_generator(opt, prefix, device, original=False):
    opt = copy.deepcopy(opt)
    opt['phase'], opt['distributed'], opt['gpu_ids'] = 'test', False, []
    if original:
        opt['model']['refinement']['enabled'] = False
    net = define_G(dict_to_nonedict(opt)).to(device)
    net.set_loss(device)
    net.set_new_noise_schedule(opt['model']['beta_schedule']['train'], device)
    net.load_state_dict(torch.load(prefix + '_G.pth', map_location=device, weights_only=True), strict=True)
    return net.eval()


def collect_cases(args):
    if args.manifest:
        path = Path(args.manifest).resolve()
        value = json.loads(path.read_text(encoding='utf-8'))
        cases = value['cases'] if isinstance(value, dict) else value
        for case in cases:
            for key in ('reference', 'mask'):
                case[key] = str((path.parent / case[key]).resolve())
        return cases
    if not args.reference_dir or not args.mask_dir:
        raise ValueError('provide --manifest or both --reference_dir and --mask_dir')
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    references = {}
    for p in Path(args.reference_dir).iterdir():
        if p.suffix.lower() in extensions:
            if p.stem in references:
                raise ValueError('duplicate reference stem: ' + p.stem)
            references[p.stem] = p
    cases = []
    for mask in sorted(Path(args.mask_dir).iterdir()):
        if mask.suffix.lower() in extensions:
            if mask.stem not in references:
                raise ValueError('no reference with same stem for ' + str(mask))
            cases.append(dict(id=mask.stem, reference=str(references[mask.stem]), mask=str(mask), mode=args.mode))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train_kaggle_refine_v1.json')
    parser.add_argument('--weight', required=True, help='new checkpoint prefix, without _G.pth')
    parser.add_argument('--baseline_weight', help='optional verified original 2-channel G prefix')
    parser.add_argument('--manifest', help='JSON cases with explicit reference/mask/mode')
    parser.add_argument('--reference_dir')
    parser.add_argument('--mask_dir')
    parser.add_argument('--mode', choices=['independent', 'corresponding'], default='independent')
    parser.add_argument('--output_dir', required=True, help='must be a new directory')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--timesteps', nargs='+', type=int, default=[200, 800, 1600])
    parser.add_argument('--refine_steps', type=int, default=3)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    cases = collect_cases(args)
    if not cases:
        raise ValueError('no explicit reference/mask cases')
    torch.manual_seed(args.seed)
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    net = load_generator(config, args.weight, torch.device(args.device))
    baseline = load_generator(config, args.baseline_weight, torch.device(args.device), original=True) if args.baseline_weight else None
    metadata = dict(config=config, weight=args.weight, weight_sha256=sha256(args.weight+'_G.pth'),
                    baseline_weight=args.baseline_weight,
                    baseline_sha256=sha256(args.baseline_weight+'_G.pth') if args.baseline_weight else None)
    validate_refinement(net, cases, args.output_dir, args.timesteps, args.seed,
                        args.refine_steps, args.threshold, metadata, baseline)
    print('Saved raw refinement outputs:', args.output_dir)


if __name__ == '__main__':
    main()
