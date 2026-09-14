"""How pixel-art-like is an image?

Every function takes a uint8 RGB array of shape (H, W, 3) and returns a scalar.
No eyeballing anywhere in this project.
"""

import numpy as np


def block_rms(img, block):
    """RMS deviation of each pixel from the mean of its own block, in grey levels.

    Pixel art upscaled with nearest-neighbour is flat inside every block x block
    cell, so a real sprite scores exactly 0.0. Anything the VAE or the UNet
    smears across a cell shows up here immediately. The units are 0-255, so a
    score of 4.0 reads as "pixels sit on average four grey levels away from the
    single colour their block is supposed to be".

    This is the primary metric of the project: it needs no reference
    distribution to interpret, because 0 is perfect by definition.
    """
    h, w, _ = img.shape
    if h % block or w % block:
        raise ValueError(f"{h}x{w} image is not divisible by block size {block}")
    cells = img.astype(np.float64).reshape(h // block, block, w // block, block, 3)
    return float(np.sqrt(cells.var(axis=(1, 3)).mean()))


def _nearest_palette_distance(img, palette):
    """Euclidean RGB distance from every pixel to the closest palette colour."""
    px = img.reshape(-1, 3).astype(np.float32)
    pal = palette.reshape(-1, 3).astype(np.float32)
    return np.linalg.norm(px[:, None, :] - pal[None, :, :], axis=2).min(axis=1)


def palette_distance(img, palette):
    """Mean distance from a pixel to the nearest palette colour, in grey levels.

    The continuous companion to palette_adherence. On a VAE reconstruction the
    exact-hit fraction collapses to ~0 and tells you nothing, while this keeps
    reporting how far off-palette the colours actually drifted.
    """
    return float(_nearest_palette_distance(img, palette).mean())


def palette_adherence(img, palette, tol=0.0):
    """Fraction of pixels within `tol` of a palette colour (tol=0 is an exact hit).

    Note this is only meaningful *before* post-processing: quantising to the
    palette pins it to 1.0 by construction, so reporting it on a quantised
    image measures nothing.
    """
    return float((_nearest_palette_distance(img, palette) <= tol).mean())
