"""The VAE reconstruction floor: the ceiling no LoRA adapter can beat.

Stable Diffusion never touches pixels. It denoises in the latent space of a
frozen VAE that downsamples by 8, and every sample has to come back out through
that decoder. A LoRA adapter only modifies the UNet, so whatever the VAE
destroys on the round trip is damage the adapter cannot repair, no matter how
well it trains. Measuring that first tells us whether this project is viable
before we build anything on top of it.

It also settles the sprite resolution, because sprite size decides how many
latent cells carry a single sprite pixel once we upscale to SD's native 512:

    16x16 sprite -> nearest x32 -> 32px blocks -> 4x4 latents per sprite pixel
    32x32 sprite -> nearest x16 -> 16px blocks -> 2x2 latents per sprite pixel
    64x64 sprite -> nearest  x8 ->  8px blocks -> 1x1 latent  per sprite pixel

At 64x64 the VAE has exactly one latent cell per sprite pixel and structurally
cannot hold a hard edge. The question is what 32x32 costs us.

Inputs are synthetic on purpose. A generated sprite is guaranteed to be exactly
on-palette and exactly on-grid, so it scores 0.0 block RMS and 1.0 palette
adherence going in, which means every bit of degradation coming out is
attributable to the VAE alone. Real sprites would arrive pre-contaminated with
resampling and compression artefacts and blur the reading. We re-run this same
probe on the real dataset once we have one.

Run from the repo root:  python -m experiments.vae_floor
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import AutoencoderKL

from src.metrics import block_rms, palette_adherence, palette_distance

MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CANVAS = 512               # SD 1.5's native training resolution
SPRITE_SIZES = (16, 32, 64)
N_SPRITES = 8
SEED = 0

# Index 0 is the background and index 1 the outline; the rest are fills.
PALETTE = np.array(
    [
        [232, 232, 240],
        [26, 28, 44],
        [93, 39, 93],
        [177, 62, 83],
        [239, 125, 87],
        [255, 205, 117],
        [99, 199, 77],
        [62, 137, 72],
    ],
    dtype=np.uint8,
)


def synth_sprite(n, rng):
    """A left-right symmetric blob with a one-pixel outline, drawn only from PALETTE.

    Silhouette detail scales with n, so a 64x64 sprite genuinely carries more
    shape than a 16x16 one rather than being the same sprite blown up. Colour
    regions stay large and contiguous, which is what real sprites look like and
    what decides how much high-frequency content the VAE has to cope with.
    """
    g = n // 2                                              # logical cells, 2x2 px each
    solid = rng.random((g, g // 2)) < 0.62                  # left half of the silhouette
    solid[:, -1] = True                                     # keep the centre column joined
    colour = np.kron(
        rng.integers(2, len(PALETTE), (4, 2)),              # 4x2 coarse colour layout
        np.ones((g // 4, g // 4), int),                     # blown up to the cell grid
    )

    idx = np.where(solid, colour, 0)                        # 0 = background
    idx = np.concatenate([idx, idx[:, ::-1]], axis=1)       # mirror to full width
    idx = np.kron(idx, np.ones((2, 2), int))                # cells -> pixels, now (n, n)

    filled = idx != 0
    p = np.pad(filled, 1)
    interior = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    idx[filled & ~interior] = 1                             # outline

    return PALETTE[idx]


def upscale(sprite, size):
    """Nearest-neighbour only. Any interpolation here would destroy the thing we measure."""
    k = size // sprite.shape[0]
    return np.repeat(np.repeat(sprite, k, axis=0), k, axis=1)


@torch.no_grad()
def roundtrip(vae, img, device):
    """Encode to latents and decode straight back, with nothing in between."""
    x = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 127.5 - 1.0
    # .mean, not .sample(): the VAE encoder outputs a distribution and training
    # draws from it, but a floor measurement wants the best the VAE can do, not
    # one noisy draw. vae.config.scaling_factor is skipped deliberately - it is
    # a constant that would be divided straight back out before decoding.
    z = vae.encode(x.to(device)).latent_dist.mean
    y = vae.decode(z).sample
    y = (y[0].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5
    return np.clip(np.round(y), 0, 255).astype(np.uint8)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading VAE on {device}")
    vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae").to(device).eval()

    rows, examples = [], {}
    for size in SPRITE_SIZES:
        rng = np.random.default_rng(SEED)          # same seed per size, honest comparison
        block = CANVAS // size                     # how many screen pixels one sprite pixel owns
        scores = []
        for i in range(N_SPRITES):
            original = upscale(synth_sprite(size, rng), CANVAS)
            recon = roundtrip(vae, original, device)
            scores.append(
                (
                    block_rms(recon, block),
                    palette_distance(recon, PALETTE),
                    palette_adherence(recon, PALETTE, tol=8.0),
                )
            )
            if i == 0:
                examples[size] = (original, recon)

        mean = np.mean(scores, axis=0)
        std = np.std(scores, axis=0)
        rows.append((size, block, (block // 8) ** 2, *mean, *std))
        print(
            f"{size:>2}x{size:<2}  block {block:>3}px  {(block // 8) ** 2:>2} latents/px   "
            f"block RMS {mean[0]:6.2f} +/- {std[0]:4.2f}   "
            f"palette dist {mean[1]:6.2f}   within 8: {mean[2]:5.1%}"
        )

    # sanity check: a real sprite must score a perfect 0.0 / 1.0 or the metrics are wrong
    probe = upscale(synth_sprite(32, np.random.default_rng(SEED)), CANVAS)
    print(
        f"\ninput sanity  block RMS {block_rms(probe, 16):.4f}   "
        f"exact palette adherence {palette_adherence(probe, PALETTE):.4f}"
    )

    with open("results/vae_floor.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["sprite", "block_px", "latents_per_sprite_px", "block_rms",
             "palette_dist", "adherence_8", "block_rms_std", "palette_dist_std",
             "adherence_8_std"]
        )
        w.writerows(rows)

    fig, axes = plt.subplots(len(SPRITE_SIZES), 3, figsize=(8.4, 2.9 * len(SPRITE_SIZES)))
    for ax_row, size in zip(axes, SPRITE_SIZES):
        original, recon = examples[size]
        diff = np.abs(original.astype(int) - recon.astype(int)).mean(axis=2)
        rms = block_rms(recon, CANVAS // size)
        for ax, im, title, kw in zip(
            ax_row,
            [original, recon, diff],
            [f"{size}x{size} sprite", f"VAE round trip (RMS {rms:.1f})", "absolute error"],
            [{}, {}, dict(cmap="inferno", vmin=0, vmax=48)],
        ):
            ax.imshow(im, interpolation="nearest", **kw)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.suptitle("What SD 1.5's VAE does to pixel art before any training", fontsize=11)
    fig.tight_layout()
    fig.savefig("results/vae_floor.png", dpi=140)
    print("wrote results/vae_floor.csv and results/vae_floor.png")


if __name__ == "__main__":
    main()
