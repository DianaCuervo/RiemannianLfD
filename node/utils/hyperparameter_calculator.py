import torch
from vae.vae_model import get_M

def calculate_dataset_baselines(vae, latent_paths, device='cpu'):
    """
    Calculates the Riemannian Ground Density and Path Energy
    from a list of RAW, unsegmented encoded human demonstrations.
    """
    vae.eval()
    vae.to(device)

    total_density, total_energy = 0.0, 0.0
    total_points, total_steps = 0, 0

    print("\n🔍 Calculating Riemannian Baselines from RAW Demonstrations...")

    with torch.no_grad():
        for path in latent_paths:
            if not isinstance(path, torch.Tensor):
                path = torch.tensor(path, dtype=torch.float32)
            z_raw = path.to(device)
            T, D = z_raw.shape

            G_raw = get_M(vae, z_raw.clone())[2]
            if not isinstance(G_raw, torch.Tensor):
                G_raw = torch.tensor(G_raw, dtype=torch.float32)

            G = torch.clamp(G_raw.clone().detach(), min=1e-3, max=1e6).to(device)

            metric_trace = torch.diagonal(G, dim1=-2, dim2=-1).sum(-1)
            total_density += metric_trace.sum().item() / 2.0
            total_points += T

            dt = 1.0 / (T - 1) if T > 1 else 1.0
            v_raw = (z_raw[1:, :] - z_raw[:-1, :]) / dt
            G_mid = G[:-1, :, :]

            energy_step = torch.einsum('ti, tij, tj -> t', v_raw, G_mid, v_raw)
            total_energy += energy_step.sum().item()
            total_steps += (T - 1)

    safe_baseline = total_density / total_points if total_points > 0 else 1.0
    avg_path_energy = total_energy / total_steps if total_steps > 0 else 1.0
    recommended_scale = 1.0 / avg_path_energy if avg_path_energy > 0 else 1.0

    print(f"{'=' * 60}")
    print(f"📊 RAW DATASET BASELINES CALCULATED")
    print(f"{'=' * 60}")
    print(f"1. Ground Density (Trace/2): {safe_baseline:,.4f}")
    print(f"2. Avg Demo Path Energy:     {avg_path_energy:,.4f}")
    print(f"👉 auto-calculated safe_baseline: {safe_baseline:.2f}")
    print(f"👉 auto-calculated metric_scale_energy: {recommended_scale:.2e}")
    print(f"{'=' * 60}\n")

    return safe_baseline, recommended_scale

