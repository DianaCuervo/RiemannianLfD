import torch
import os
import json

import torch.nn as nn
import torch.nn.functional as F
from node.utils.hyperparameter_calculator import calculate_dataset_baselines
from node.utils.plots import plot_sphere_trajectories, plot_sphere_trajectories_from_dataloader


class SphereManifold:
    """
    Analytic 2-Sphere Manifold embedded in 3D Euclidean space (S^2 subset of R^3).
    Provides geometry operations, SLERP data generation, tangent projection,
    and exact Riemannian loss functions.
    """

    def __init__(self, radius: float = 1.0, eps: float = 1e-7):
        self.radius = radius
        self.eps = eps

    # -------------------------------------------------------------------------
    # 1. Manifold Constraints & Tangent Space Projection
    # -------------------------------------------------------------------------
    def project_tangent(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Projects an unconstrained 3D vector u onto the tangent space T_x(S^2).

        Args:
            x: Points on sphere, shape [..., 3]
            u: Unconstrained velocity output from Neural Network, shape [..., 3]
        Returns:
            v: Tangent velocity vector, orthogonal to x, shape [..., 3]
        """
        x_unit = x / (torch.norm(x, dim=-1, keepdim=True) + self.eps)
        dot = torch.sum(u * x_unit, dim=-1, keepdim=True)
        return u - dot * x_unit

    def project_manifold(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalizes points back onto the unit sphere surface (corrects ODE integration drift).
        """
        return self.radius * (x / (torch.norm(x, dim=-1, keepdim=True) + self.eps))

    # -------------------------------------------------------------------------
    # 2. Riemannian Distance & Metric Operations
    # -------------------------------------------------------------------------
    def distance(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Computes the exact Riemannian geodesic distance on S^2 using the inner product:
        d(p, q) = R * arccos(p . q / (|p| * |q|))

        Args:
            p, q: Tensors of shape [..., 3]
        Returns:
            Geodesic distance tensor of shape [..., 1]
        """
        p_unit = p / (torch.norm(p, dim=-1, keepdim=True) + self.eps)
        q_unit = q / (torch.norm(q, dim=-1, keepdim=True) + self.eps)

        dot = torch.sum(p_unit * q_unit, dim=-1, keepdim=True)
        # Clamp to prevent NaN gradients near arccos(-1) and arccos(1)
        dot_clamped = torch.clamp(dot, -1.0 + self.eps, 1.0 - self.eps)
        return self.radius * torch.acos(dot_clamped)

    # -------------------------------------------------------------------------
    # 3. Loss Functions
    # -------------------------------------------------------------------------
    def imitation_loss(self, pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
        """
        Term 1: Pointwise Riemannian Imitation Loss across time steps.

        Args:
            pred_traj: Predicted trajectory [Batch, Time, 3]
            gt_traj:   Ground truth demonstration [Batch, Time, 3]
        """
        pointwise_dist = self.distance(pred_traj, gt_traj)  # [Batch, Time, 1]
        return torch.mean(pointwise_dist)

    def energy_loss(self, pred_traj: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """
        Term 2: Riemannian Energy Functional E = 0.5 * integral(||v_t||^2 dt).
        Uses discrete finite-difference velocities on the predicted trajectory.

        Args:
            pred_traj: Predicted trajectory [Batch, Time, 3]
            times:     Time discretization grid [Time]
        """
        dt = (times[1:] - times[:-1]).view(1, -1, 1)  # [1, Time-1, 1]
        vel = (pred_traj[:, 1:, :] - pred_traj[:, :-1, :]) / (dt + self.eps)

        # Ensure velocity vectors lie in the tangent space at pred_traj[:, :-1, :]
        pts = pred_traj[:, :-1, :]
        tangent_vel = self.project_tangent(pts, vel)

        speed_sq = torch.sum(tangent_vel ** 2, dim=-1)  # [Batch, Time-1]
        dt_flat = (times[1:] - times[:-1]).view(1, -1)  # [1, Time-1]

        energy = 0.5 * torch.sum(speed_sq * dt_flat, dim=-1)  # [Batch]
        return torch.mean(energy)

    def goal_loss(self, pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
        """
        Term 3: Goal Convergence Loss (Riemannian distance squared at final time step).

        Args:
            pred_traj: [Batch, Time, 3]
            gt_traj:   [Batch, Time, 3]
        """
        pred_final = pred_traj[:, -1, :]
        gt_final = gt_traj[:, -1, :]
        goal_dist = self.distance(pred_final, gt_final)
        return torch.mean(goal_dist ** 2)

    def compute_composite_loss(self, pred_traj: torch.Tensor, gt_traj: torch.Tensor,
                               times: torch.Tensor, w_imit: float = 1.0,
                               w_energy: float = 0.01, w_goal: float = 1.0):
        """
        Combines all three terms into a single scalar loss.
        """
        l_imit = self.imitation_loss(pred_traj, gt_traj)
        l_energy = self.energy_loss(pred_traj, times)
        l_goal = self.goal_loss(pred_traj, gt_traj)

        total_loss = w_imit * l_imit + w_energy * l_energy + w_goal * l_goal
        return total_loss, {"imit": l_imit.item(), "energy": l_energy.item(), "goal": l_goal.item()}

    # -------------------------------------------------------------------------
    # 4. Dataset Generation (SLERP Ground Truth)
    # -------------------------------------------------------------------------
    def slerp(self, p0: torch.Tensor, p1: torch.Tensor, num_points: int = 50) -> torch.Tensor:
        """
        Generates ground-truth geodesic arcs using Spherical Linear Interpolation.

        Args:
            p0: Start points on unit sphere [Batch, 3]
            p1: Goal points on unit sphere [Batch, 3]
            num_points: Number of trajectory points (e.g., 50)
        Returns:
            Geodesic arc trajectories of shape [Batch, num_points, 3]
        """
        p0 = self.project_manifold(p0)
        p1 = self.project_manifold(p1)

        dot = torch.sum(p0 * p1, dim=-1, keepdim=True)
        dot = torch.clamp(dot, -1.0 + self.eps, 1.0 - self.eps)

        omega = torch.acos(dot)  # [Batch, 1]
        sin_omega = torch.sin(omega)  # [Batch, 1]

        t_grid = torch.linspace(0.0, 1.0, num_points, device=p0.device).view(1, num_points, 1)

        p0_exp = p0.unsqueeze(1)  # [Batch, 1, 3]
        p1_exp = p1.unsqueeze(1)  # [Batch, 1, 3]

        # Standard SLERP weights
        w0 = torch.sin((1.0 - t_grid) * omega.unsqueeze(1)) / (sin_omega.unsqueeze(1) + self.eps)
        w1 = torch.sin(t_grid * omega.unsqueeze(1)) / (sin_omega.unsqueeze(1) + self.eps)

        traj = w0 * p0_exp + w1 * p1_exp

        # Fallback to linear interpolation for nearly parallel/identical start-goal pairs
        mask = (sin_omega < 1e-5).unsqueeze(1)
        lerp_traj = (1.0 - t_grid) * p0_exp + t_grid * p1_exp
        lerp_traj = self.project_manifold(lerp_traj)

        return torch.where(mask, lerp_traj, traj)

    def generate_random_dataset(self, num_samples: int = 1000, num_points: int = 50):
        """
        Generates a synthetic dataset of geodesic arcs flowing consistently from South to North.
        The paths follow lines of longitude (meridians) to ensure they form a valid,
        non-intersecting vector field for the Neural ODE to learn.
        """
        # 1. Random longitudes (phi) from 0 to 2*pi
        phi = torch.rand(num_samples) * 2 * torch.pi

        # 2. Random start colatitudes (theta_0).
        # theta = 0 is the North Pole, theta = pi is the South Pole.
        # We sample starting points anywhere on the sphere (both hemispheres).
        # Adding a tiny epsilon to prevent starting exactly on the geometric singularity of the pole.
        theta_0 = torch.rand(num_samples) * (torch.pi - 1e-4) + 1e-4

        # 3. Random end colatitudes (theta_1).
        # To guarantee the path strictly flows South -> North, the end angle MUST be closer to 0.
        # Multiplying by a random fraction ensures varying lengths while always moving North.
        theta_1 = theta_0 * torch.rand(num_samples)

        # 4. Convert spherical coordinates to 3D Euclidean coordinates (x, y, z)
        p0_x = torch.sin(theta_0) * torch.cos(phi)
        p0_y = torch.sin(theta_0) * torch.sin(phi)
        p0_z = torch.cos(theta_0)
        p0 = torch.stack([p0_x, p0_y, p0_z], dim=-1)

        p1_x = torch.sin(theta_1) * torch.cos(phi)
        p1_y = torch.sin(theta_1) * torch.sin(phi)
        p1_z = torch.cos(theta_1)
        p1 = torch.stack([p1_x, p1_y, p1_z], dim=-1)

        # 5. Project to manifold (to correct any floating point rounding errors)
        p0 = self.project_manifold(p0)
        p1 = self.project_manifold(p1)

        # 6. Generate the smooth trajectories using SLERP
        gt_trajectories = self.slerp(p0, p1, num_points=num_points)

        return gt_trajectories

    # Dataset creation
    def create_dataset(self, demo_paths, save_dir):
        """
        Args:
            encoded_paths (list of tensors): List of encoded trajectories, each shape [Steps, Latent_Dim]
            save_dir (str): Folder to save .pt files.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_count = 0
        for path_idx, path in enumerate(demo_paths):
            sample = {
                'p1': path[0, :].clone(),
                'p2': path[-1, :].clone(),
                'z': path.clone(),
                'path_id': path_idx
            }
            torch.save(sample, f"{save_dir}/path_{file_count}.pt")
            file_count += 1
        print(f"✅ Successfully created {file_count} segments in {save_dir}")

    def calculate_dataset_baselines(self, raw_paths, device='cpu'):
        """
        Calculates the Riemannian Ground Density and Path Energy
        from a list of RAW encoded demonstrations using ||Jv||^2.
        Adapted for the Analytic Sphere Manifold.
        """
        total_density, total_energy = 0.0, 0.0
        total_points, total_steps = 0, 0

        print("\n🔍 Calculating Riemannian Baselines from RAW Sphere Demonstrations...")

        with torch.no_grad():
            for path in raw_paths:
                if not isinstance(path, torch.Tensor):
                    path = torch.tensor(path, dtype=torch.float32)

                # Move path coordinates to device
                z_raw = path.to(device)
                T, D = z_raw.shape

                # FIX: Pass 'self' instead of 'manifold' to get_M
                _, J_raw, _ = get_M(self, z_raw.clone())
                J_raw = torch.as_tensor(J_raw, dtype=torch.float32, device=device)

                # Dynamically handle shape: J_raw is (T, 3, 3) for the sphere
                K = J_raw.shape[-2]
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
        print(f"📊 RAW SPHERE DATASET BASELINES CALCULATED")
        print(f"{'=' * 60}")
        print(f"1. Ground Density (Trace/2): {safe_baseline:,.4f}")
        print(f"2. Avg Demo Path Energy:     {avg_path_energy:,.4f}")
        print(f"👉 auto-calculated safe_baseline: {safe_baseline:.2f}")
        print(f"👉 auto-calculated metric_scale_energy: {recommended_scale:.2e}")
        print(f"{'=' * 60}\n")

        return safe_baseline, recommended_scale

    def build_sphere_dataset_offline(self, config):
        """
        Reads the config, looks at the requested dataset, and processes it
        if the .pt files don't already exist.
        """
        dataset_type = config['dataset']['type']  # e.g., 'lasa' or 'toy'
        save_dir = config['dataset']['save_dir']  # e.g., './data/NShape_50seg'

        if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
            print(f"Dataset already exists at {save_dir}. Skipping preprocessing.")
            return save_dir

        print(f"Building dataset for {dataset_type.upper()}...")
        os.makedirs(save_dir, exist_ok=True)

        num_samples = config['dataset']['number_demos']
        num_points = config['dataset']['time_steps']
        path_demos = self.generate_random_dataset(num_samples=num_samples, num_points=num_points)
        device = path_demos.device

        plot_sphere_trajectories(path_demos)

        # --- RUN THE BASELINE CALCULATOR ---
        calc_baseline, calc_scale = self.calculate_dataset_baselines(path_demos, device=device)
        metadata_path = os.path.join(save_dir, "dataset_baselines.json")
        with open(metadata_path, 'w') as f:
            json.dump({"safe_baseline": calc_baseline, "metric_scale_energy": calc_scale}, f, indent=4)
        print(f"💾 Saved baselines permanently to: {metadata_path}")

        plot_sphere_trajectories(path_demos)
        self.create_dataset(path_demos, save_dir)
        print(f"Generated {num_samples} trajectories, each with {num_points} points.")

        return save_dir

    # -------------------------------------------------------------------------
    # 5. Rewritten embed fn
    # -------------------------------------------------------------------------
    def embed(self, points, jacobian=False):
        """
        Rigorously maps ambient 3D points (on or off the sphere) onto the unit
        sphere and calculates the true analytic Jacobian and Riemannian metric
        accounting for off-manifold scaling.
        """
        is_batched = points.dim() > 2
        if not is_batched:
            points = points.unsqueeze(0)  # Ensure BxNx3 format

        B, N, D = points.shape

        if jacobian:
            # 1. Compute norms and true unit direction vectors (embedded surface points)
            norms = torch.norm(points, dim=-1, keepdim=True).clamp(min=1e-8)  # [B, N, 1]
            embedded = points / norms  # [B, N, 3] (Strictly on S^2)

            # 2. Create Identity matrix [B, N, 3, 3]
            eye = torch.eye(D, device=points.device, dtype=points.dtype).view(1, 1, D, D).expand(B, N, -1, -1)

            # 3. Calculate outer product of the normalized points: x_hat * x_hat^T
            xxT = torch.matmul(embedded.unsqueeze(-1), embedded.unsqueeze(-2))  # [B, N, 3, 3]

            # 4. True Generalized Jacobian: J = (1 / r) * (I - x_hat * x_hat^T)
            # This correctly penalizes/scales points that have drifted off-manifold!
            base_projection = eye - xxT
            j = base_projection / norms.unsqueeze(-1)  # Scale by 1 / ||points||

            # 5. Metric Tensor M = J^T * J
            # Since base_projection is a symmetric projection matrix (P^2 = P),
            # J^T * J simplifies to (1 / r^2) * base_projection
            metric = base_projection / (norms ** 2).unsqueeze(-1)

            if not is_batched:
                embedded = embedded.squeeze(0)
                j = j.squeeze(0)
                metric = metric.squeeze(0)

            return embedded, j, metric

        else:
            # Just return the projected points if Jacobian isn't requested
            norms = torch.norm(points, dim=-1, keepdim=True).clamp(min=1e-8)
            embedded = points / norms
            if not is_batched:
                embedded = embedded.squeeze(0)
            return embedded

    # def embed(self, points, jacobian=False):
    #     """
    #     Analytic equivalent of VAE embed for the 3D Sphere.
    #     Maps ambient 3D points exactly onto the unit sphere and calculates
    #     the analytic Jacobian and Riemannian metric tensor.
    #     """
    #     is_batched = points.dim() > 2
    #     if not is_batched:
    #         points = points.unsqueeze(0)  # BxNx3
    #
    #     # 1. Embed points onto the sphere surface
    #     embedded = self.project_manifold(points)
    #
    #     if jacobian:
    #         # 2. Compute the analytic Jacobian of the projection
    #         # For f(x) = x / ||x||, on the unit sphere, J = I - x * x^T
    #         B, N, D = embedded.shape
    #
    #         # Create an identity matrix [B, N, 3, 3]
    #         eye = torch.eye(D, device=points.device, dtype=points.dtype).view(1, 1, D, D).expand(B, N, -1, -1)
    #
    #         # Calculate outer product: x * x^T
    #         # embedded is [B, N, 3]. unsqueeze(-1) -> [B, N, 3, 1]. unsqueeze(-2) -> [B, N, 1, 3]
    #         xxT = torch.matmul(embedded.unsqueeze(-1), embedded.unsqueeze(-2))  # Result: [B, N, 3, 3]
    #
    #         j = eye - xxT
    #
    #         # 3. Metric M = J^T * J. Since J is an orthogonal projection matrix, M = J.
    #         metric = j.clone()
    #
    #         if not is_batched:
    #             embedded = embedded.squeeze(0)
    #             j = j.squeeze(0)
    #             metric = metric.squeeze(0)
    #
    #         return embedded, j, metric.detach().cpu().numpy()
    #
    #     else:
    #         if not is_batched:
    #             embedded = embedded.squeeze(0)
    #         return embedded

# -------------------------------------------------------------------------
# 6. Rewritten get_M fn for NODE training
# -------------------------------------------------------------------------
def get_M(manifold, data):
    """
    Getting metric for all points in the tensor data (Nx3).
    Analytic replacement for the VAE's get_M.
    """
    # Manage device mapping based on the incoming data
    device = data.device if isinstance(data, torch.Tensor) else 'cpu'

    if not isinstance(data, torch.Tensor):
        data = torch.as_tensor(data, dtype=torch.float32, device=device)
    else:
        data = data.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        # Call our new analytic embed function
        e, J, M = manifold.embed(data.unsqueeze(0), jacobian=True)

    if not isinstance(M, torch.Tensor):
        M = torch.as_tensor(M, dtype=torch.float32, device=device)

    # Clean up dimensions to match original pipeline expectations
    if M.dim() == 4 and M.shape[0] == 1:
        M = M.squeeze(0)

    if e.dim() == 3 and e.shape[0] == 1:
        e = e.squeeze(0)

    if J.dim() == 4 and J.shape[0] == 1:
        J = J.squeeze(0)

    return e, J, M

