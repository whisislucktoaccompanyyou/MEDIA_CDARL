"""Colorize a grayscale C-DARL synthesis using a real color fundus as chroma reference.

Does not load the model and does not change training. The synthetic image supplies
luminance; a and b come from a real color photo. Vessels stay dark because they are
already dark in L, so they must not be painted solid red.

Use the brightness-matched grayscale (*_matched.png) when it exists. The raw
*_synthetic.png is much darker than a real fundus, and substituting it for L
washes the color out.
"""

import argparse
import os

import cv2
import numpy as np


def disk_mask(gray):
    """Bright optic-disc core. Keeps reference chroma instead of retinal red."""
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    _, mask = cv2.threshold(blur, 225, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask > 0


def field_circle(gray, thresh=12):
    """Circle of the illuminated fundus. Ignores the black surround."""
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    mask = blur > thresh
    if int(mask.sum()) < 32:
        height, width = gray.shape
        return width / 2.0, height / 2.0, min(width, height) / 2.0
    yy, xx = np.nonzero(mask)
    # Median center is stable when a dark lesion bites into one side.
    cy = float(np.median(yy))
    cx = float(np.median(xx))
    dist = np.hypot(xx - cx, yy - cy)
    radius = float(np.percentile(dist, 99.5))
    return cx, cy, max(radius - 1.5, 1.0)


def circle_mask(shape, cx, cy, radius):
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def warp_ref_to_gray(gray_u8, ref_bgr):
    """Scale and translate the reference so both fundus circles coincide.

    A plain resize stretches a non-square photo and puts its disc off-center,
    which is why the rim color was ragged.
    """
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    gx, gy, gr = field_circle(gray_u8)
    rx, ry, rr = field_circle(ref_gray)
    scale = gr / max(rr, 1.0)
    # Map reference pixels onto the synthetic canvas.
    matrix = np.array(
        [[scale, 0.0, gx - scale * rx], [0.0, scale, gy - scale * ry]],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        ref_bgr,
        matrix,
        (gray_u8.shape[1], gray_u8.shape[0]),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def colorize(gray_u8, ref_bgr, chroma_sigma=21):
    if gray_u8.ndim != 2:
        raise ValueError("synthetic image must be single-channel grayscale")
    aligned = warp_ref_to_gray(gray_u8, ref_bgr)
    lab = cv2.cvtColor(aligned, cv2.COLOR_BGR2LAB).astype(np.float32)

    cx, cy, radius = field_circle(gray_u8)
    field = circle_mask(gray_u8.shape, cx, cy, radius)
    a = lab[:, :, 1].copy()
    b = lab[:, :, 2].copy()
    if chroma_sigma > 0 and field.any():
        # Fill outside the circle before blurring, or the black surround
        # pulls the rim chroma toward gray.
        a_fill = a.copy()
        b_fill = b.copy()
        a_fill[~field] = np.median(a[field])
        b_fill[~field] = np.median(b[field])
        a = np.where(field, cv2.GaussianBlur(a_fill, (0, 0), chroma_sigma), a)
        b = np.where(field, cv2.GaussianBlur(b_fill, (0, 0), chroma_sigma), b)

    disk = disk_mask(gray_u8) & field
    a = np.where(disk, lab[:, :, 1], a)
    b = np.where(disk, lab[:, :, 2], b)

    out = np.stack([gray_u8.astype(np.float32), a, b], axis=-1)
    out = np.clip(out, 0, 255).astype(np.uint8)
    color = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    color[~field] = 0
    return color


def split_existing_panel(panel_bgr):
    """A saved *_panel.png is already three square tiles separated by gray gaps.

    Resizing the whole strip to a square is what squashed the left columns.
    """
    if panel_bgr.ndim != 3 or panel_bgr.shape[1] < panel_bgr.shape[0] * 2:
        raise ValueError("expected a wide three-column panel, got {}".format(panel_bgr.shape))
    gray = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY)
    # Gap columns are a flat mid-gray. Image columns are not.
    col_std = gray.std(axis=0)
    is_gap = col_std < 2.0
    tiles = []
    in_tile = False
    start = 0
    for index, gap in enumerate(is_gap):
        if not gap and not in_tile:
            start = index
            in_tile = True
        elif gap and in_tile:
            tiles.append(panel_bgr[:, start:index])
            in_tile = False
    if in_tile:
        tiles.append(panel_bgr[:, start:])
    if len(tiles) < 3:
        raise ValueError("could not split panel into 3 tiles, found {}".format(len(tiles)))
    return tiles[0], tiles[1], tiles[2]


def as_bgr(image, size):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[0] != size or image.shape[1] != size:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return image


def panel(tiles, color):
    """Each tile is one square image. Never pass an already-concatenated strip."""
    size = color.shape[0]
    color = as_bgr(color, size)
    shown = [as_bgr(tile, size) for tile in tiles]
    shown.append(color)
    gap = np.full((size, 8, 3), 128, dtype=np.uint8)
    parts = []
    for index, tile in enumerate(shown):
        if index:
            parts.append(gap)
        parts.append(tile)
    return np.concatenate(parts, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gray", required=True, help="*_matched.png or *_synthetic.png")
    parser.add_argument("--ref", required=True, help="real color fundus, BGR")
    parser.add_argument("--out", required=True)
    parser.add_argument("--panel_out", default="")
    parser.add_argument("--original", default="", help="square grayscale original, not a panel strip")
    parser.add_argument("--mask", default="", help="square vessel mask")
    parser.add_argument(
        "--source_panel",
        default="",
        help="existing *_panel.png; split into tiles instead of resizing the whole strip",
    )
    parser.add_argument("--sigma", type=float, default=21)
    args = parser.parse_args()

    gray = cv2.imread(args.gray, cv2.IMREAD_GRAYSCALE)
    ref = cv2.imread(args.ref, cv2.IMREAD_COLOR)
    if gray is None:
        raise RuntimeError("cannot read {}".format(args.gray))
    if ref is None or ref.ndim != 3:
        raise RuntimeError("reference must be a color image: {}".format(args.ref))

    color = colorize(gray, ref, chroma_sigma=args.sigma)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if not cv2.imwrite(args.out, color):
        raise RuntimeError("cannot write {}".format(args.out))
    print("saved", args.out, color.shape)

    if args.panel_out:
        tiles = []
        if args.source_panel:
            source = cv2.imread(args.source_panel, cv2.IMREAD_COLOR)
            if source is None:
                raise RuntimeError("cannot read {}".format(args.source_panel))
            left, mid, right = split_existing_panel(source)
            tiles.extend([left, mid, right])
        else:
            if args.original:
                original = cv2.imread(args.original, cv2.IMREAD_GRAYSCALE)
                if original is None:
                    raise RuntimeError("cannot read {}".format(args.original))
                tiles.append(original)
            if args.mask:
                mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise RuntimeError("cannot read {}".format(args.mask))
                tiles.append(mask)
            tiles.append(gray)
        if not cv2.imwrite(args.panel_out, panel(tiles, color)):
            raise RuntimeError("cannot write {}".format(args.panel_out))
        print("saved", args.panel_out)


if __name__ == "__main__":
    main()
