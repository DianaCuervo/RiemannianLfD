import torch
from vae.vae_model import get_M

def calculate_dataset_baselines(vae, latent_paths, device='cpu'):
    """
    Calculates the Riemannian Ground Density and Path Energy
    from a list of RAW, unsegmented encoded human demonstrations using ||Jv||^2.
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

            # 1. Get raw Jacobian J (index 1) instead of G
            _, J_raw, _ = get_M(vae, z_raw.clone())
            J_raw = torch.as_tensor(J_raw, dtype=torch.float32, device=device)
            K = J_raw.shape[1]
            J = J_raw.reshape(T, K, D)

            # Density from J: Trace of G (sum of squared elements of J) / 2
            point_density = torch.sum(J ** 2, dim=(-2, -1)) / 2.0
            total_density += point_density.sum().item()
            total_points += T

            # 2. Velocity
            dt = 1.0 / (T - 1) if T > 1 else 1.0
            v_raw = (z_raw[1:, :] - z_raw[:-1, :]) / dt
            J_mid = J[:-1, :, :]  # Midpoints for velocity steps

            # 3. Energy using ||Jv||^2
            Jv = torch.einsum('tkd,td->tk', J_mid, v_raw)
            energy_step = torch.sum(Jv ** 2, dim=-1)

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