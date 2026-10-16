"""Off-manifold % of latent paths, measured on the magnification m(z) = log10 det M(z).

Far from the data the decoder is flat and the RBF std saturates, so det M drops again --
a low m(z) alone does not mean "on the manifold". The measure therefore looks for a climb
over the wall that surrounds the data: the threshold sits a fraction alpha of the way from
the floor (median m on the encoded training points) to the wall (peak of the radial profile
of m with distance from the data). One reference per VAE (beta): every method at that beta
is scored against the same grid, floor and wall.
"""
import math
import os

import numpy as np
import torch

from vae.vae_model import get_M
from vae.data.vae_dataset import load_vae_training_points
from vae.utils.config import load_dataset_config

ALPHAS = (0.25, 0.5, 0.75)
DENSE_SAMPLES = 200  # S: points per path, so short excursions between the T output points are caught
_MIN_BATCH = 128     # the vMF decoder is only exercised with batches of at least this size elsewhere


def log10_det_M(vae_model, z, chunk=2048):
    """m(z) = log10 det M(z) for latent points z [N, d] -> [N] (CPU float64)."""
    z = torch.as_tensor(z, dtype=torch.float32).reshape(-1, z.shape[-1])
    out = []
    for i in range(0, z.shape[0], chunk):
        z_c = z[i:i + chunk]
        n = z_c.shape[0]
        if n < _MIN_BATCH:
            z_c = torch.cat([z_c, z_c[-1:].expand(_MIN_BATCH - n, -1)], dim=0)
        M = get_M(vae_model, z_c)[2].detach().double().cpu()
        out.append(torch.linalg.slogdet(M)[1][:n] / math.log(10.0))
    return torch.cat(out)


def encode_training_points(vae_model, dataset, shape):
    """Z: all VAE training demonstration points, encoded (latent means)."""
    shape_arg = None if shape in [None, 'None'] else shape
    x = load_vae_training_points(load_dataset_config(dataset, shape_arg))
    device = next(vae_model.parameters()).device
    with torch.no_grad():
        z = vae_model.encode(x.to(device), train_rbf=True)[1]
    return z.detach().cpu()


def build_reference(vae_model, z_train, grid_n=120, pad=0.5):
    """Floor, wall and radial profile of m around the encoded training points z_train [N, d]."""
    z_train = torch.as_tensor(z_train, dtype=torch.float32)
    z_min, z_max = z_train.min(dim=0).values, z_train.max(dim=0).values
    span = float(torch.linalg.norm(z_max - z_min))

    # Floor: median magnification on the data
    floor = float(log10_det_M(vae_model, z_train).median())

    # Grid over the bounding box, padded by `pad` of its size on each side
    lo, hi = z_min - pad * (z_max - z_min), z_max + pad * (z_max - z_min)
    gx, gy = torch.linspace(lo[0], hi[0], grid_n), torch.linspace(lo[1], hi[1], grid_n)
    xx, yy = torch.meshgrid(gx, gy, indexing='ij')
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    # d(g): distance to the nearest training point, in % of the span
    d = torch.cat([torch.cdist(grid[i:i + 1024], z_train).min(dim=1).values
                   for i in range(0, grid.shape[0], 1024)]) * 100.0 / span
    m_grid = log10_det_M(vae_model, grid)

    # Radial profile: median m in 1% bins of d; the wall is its peak
    bins = torch.floor(d).long()
    bin_ids, profile = [], []
    for b in torch.unique(bins).tolist():
        bin_ids.append(b)
        profile.append(float(m_grid[bins == b].median()))
    wall = float(max(profile))
    if wall <= floor:
        print(f"⚠️ Off-manifold wall ({wall:.3f}) is not above the floor ({floor:.3f}): thresholds are meaningless")

    print(f"Off-manifold reference: floor f = {floor:.3f}, wall w = {wall:.3f}, span = {span:.3f}")
    return {
        'floor': floor,
        'wall': wall,
        'span': span,
        'bbox_lo': lo,
        'bbox_hi': hi,
        'grid_n': grid_n,
        'pad': pad,
        'profile_bins': torch.tensor(bin_ids, dtype=torch.long),
        'profile_median_m': torch.tensor(profile, dtype=torch.float64),
    }


def get_off_manifold_reference(vae_model, dataset, shape, cache_path):
    """build_reference on this VAE's training points, cached at cache_path (one file per beta)."""
    if os.path.exists(cache_path):
        print(f"✅ Found off-manifold reference at: {cache_path}")
        return torch.load(cache_path, weights_only=True)
    print("Building off-manifold reference (grid, floor, wall)...")
    ref = build_reference(vae_model, encode_training_points(vae_model, dataset, shape))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(ref, cache_path)
    return ref


def reference_summary(ref):
    """The JSON-friendly scalars of a reference (with the thresholds per alpha)."""
    return {
        'floor': ref['floor'],
        'wall': ref['wall'],
        'span': ref['span'],
        'grid_n': ref['grid_n'],
        'pad': ref['pad'],
        'thresholds': {f"alpha_{a}": ref['floor'] + a * (ref['wall'] - ref['floor']) for a in ALPHAS},
        'profile': {'bin_pct': ref['profile_bins'].tolist(), 'median_m': ref['profile_median_m'].tolist()},
    }


def off_manifold_rates(vae_model, paths, ref, alphas=ALPHAS):
    """Off-manifold paths % and points % of latent paths [B, S, d] for each threshold alpha."""
    paths = torch.as_tensor(paths, dtype=torch.float32).detach().cpu()
    B, S, _ = paths.shape
    m = log10_det_M(vae_model, paths.reshape(-1, paths.shape[-1])).reshape(B, S)

    report = {'samples_per_path': S, 'num_paths': B}
    for a in alphas:
        tau = ref['floor'] + a * (ref['wall'] - ref['floor'])
        above = m > tau
        report[f"alpha_{a}"] = {
            'tau': float(tau),
            'paths_pct': float(100.0 * above.any(dim=1).float().mean()),
            'points_pct': float(100.0 * above.float().mean()),
        }
    print("Off-manifold % (paths / points): " + " | ".join(
        f"α={a}: {report[f'alpha_{a}']['paths_pct']:.1f} / {report[f'alpha_{a}']['points_pct']:.1f}" for a in alphas))
    return report
