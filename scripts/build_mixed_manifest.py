#!/usr/bin/env python3
"""Scan fundus datasets and write the mixed-training manifest.

Images are not copied. The loader reads the original files through the
paths recorded in split_manifest.json. Fractal labels stay in trainA.
"""
import argparse
import json
import os
from collections import OrderedDict

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.ppm'}
SKIP_DIR_NAMES = {
    'mask', 'masks', 'label', 'labels', 'annotation', 'annotations',
    'groundtruth', 'ground_truth', 'gt', 'seg', 'segmentation',
}


def is_image(name):
    return os.path.splitext(name)[1].lower() in IMAGE_EXTS


def looks_like_label_dir(path):
    name = os.path.basename(path).lower().replace('-', '_').replace(' ', '_')
    return name in SKIP_DIR_NAMES or name.endswith('_mask') or name.endswith('_masks')


def scan(root):
    found = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = [d for d in dirnames if not looks_like_label_dir(os.path.join(dirpath, d))]
        for name in filenames:
            if is_image(name):
                found.append(os.path.join(dirpath, name))
    return sorted(set(os.path.realpath(p) for p in found))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--datasets', nargs='+', required=True)
    args = parser.parse_args()

    by_dataset = OrderedDict()
    missing = []
    for name in args.datasets:
        root = os.path.join(args.data_root, name)
        if not os.path.isdir(root):
            missing.append(root)
            continue
        paths = scan(root)
        if not paths:
            missing.append(root + ' (no images)')
            continue
        by_dataset[name] = paths

    if missing:
        raise SystemExit('Cannot use these dataset directories:\n  ' + '\n  '.join(missing))

    train = []
    counts = OrderedDict()
    for name, paths in by_dataset.items():
        counts[name] = len(paths)
        train.extend(paths)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    payload = {
        'train': train,
        'counts': counts,
        'total': len(train),
        'note': 'Absolute paths. Loader reads originals; nothing is copied.',
    }
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)
        f.write('\n')
    print('mixed manifest: %d images -> %s' % (len(train), args.out))
    for name, count in counts.items():
        print('  %-12s %d' % (name, count))


if __name__ == '__main__':
    main()
