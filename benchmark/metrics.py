import torch
import numpy as np


# ==========================================
# EUCLIDEAN DECODING UTILS
# ==========================================

#Function to decode latent proxy-geodesic to robot Cartesian space trajectories
def decode_latente_paths(vae_model, latent_paths):
    # If the paths were returned as a Python list, stack them into a single batched tensor
    if isinstance(latent_paths, list):
        if len(latent_paths) == 0:
            raise ValueError("The latent_paths list is completely empty!")

        # If the elements are PyTorch tensors, stack them safely
        if torch.is_tensor(latent_paths[0]):
            latent_paths = torch.stack(latent_paths)
        else:
            # Fallback if they are raw NumPy arrays
            latent_paths = torch.as_tensor(np.stack(latent_paths), dtype=torch.float32)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = next(vae_model.parameters()).device  # Use the same device as the VAE model
    batch_size, time_steps, latent_dim = latent_paths.shape
    # Flatten everything: [2500, 2]
    z_flat = torch.as_tensor(latent_paths, dtype=torch.float32).to(device).reshape(-1, latent_dim)

    decoded_full = []
    FIXED_BATCH = 128  # The hard-coded requirement of your vMF distribution

    vae_model.eval()
    with torch.no_grad():
        for i in range(0, z_flat.shape[0], FIXED_BATCH):
            z_chunk = z_flat[i: i + FIXED_BATCH]
            actual_len = z_chunk.shape[0]

            # --- PADDING LOGIC ---
            # If the chunk is not 128, pad it with zeros to reach 128
            if actual_len < FIXED_BATCH:
                padding = torch.zeros((FIXED_BATCH - actual_len, latent_dim)).to(device)
                z_chunk = torch.cat([z_chunk, padding], dim=0)

            # Now z_chunk is exactly [128, 2]
            pos_dist, qua_dist = vae_model.decode(z_chunk)

            # Extract Means
            # print(pos_dist.mean.shape, qua_dist.loc.shape)

            p_m = pos_dist.mean  # [128, 3]
            # q_m = qua_dist.mean  # [128, 2/4] #--> Original line: vMF mean is loc shrunk by a Bessel ratio < 1
            q_m = qua_dist.loc  # [128, 2/4] unit-norm mode, as decode_latent_goals
            # print(f"p_m shape: {p_m.shape}, q_m shape: {q_m.shape}, q_d shape: {q_d.shape}")

            combined = torch.cat([p_m, q_m], dim=-1)

            # --- UN-PADDING ---
            # Remove the dummy data we added before saving
            decoded_full.append(combined[:actual_len])

    # Combine everything back together
    full_state_flat = torch.cat(decoded_full, dim=0).cpu().numpy()

    # Reshape to [5, 500, 5] (or 7, depending on your robot's DoF)
    real_space_paths = full_state_flat.reshape(batch_size, time_steps, -1)
    return real_space_paths

def decode_latent_goals(vae_model, latent_goals):
    """
    Decodes independent goal points from the latent space to the real space.
    Handles vMF 128-batch requirements seamlessly for standalone 2D inputs.

    Args:
        vae_model: The trained VAE model.
        latent_goals: List of tensors/arrays OR a single tensor of shape [N, 2]

    Returns:
        real_space_goals: NumPy array of shape [N, RealSpaceDim] (e.g., [N, 5] or [N, 7])
    """
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = next(vae_model.parameters()).device  # Use the same device as the VAE model
    # 1. Safely handle input types (list vs. raw tensor) and push to device
    if isinstance(latent_goals, list):
        z_flat = torch.stack([torch.as_tensor(g, dtype=torch.float32) for g in latent_goals]).to(device)
    else:
        z_flat = torch.as_tensor(latent_goals, dtype=torch.float32).to(device)

    num_goals, latent_dim = z_flat.shape
    decoded_full = []
    FIXED_BATCH = 128  # The hard-coded requirement of your vMF distribution

    vae_model.eval()
    with torch.no_grad():
        for i in range(0, num_goals, FIXED_BATCH):
            z_chunk = z_flat[i: i + FIXED_BATCH]
            actual_len = z_chunk.shape[0]

            # --- PADDING LOGIC ---
            # If the final chunk is less than 128, pad it out to 128
            if actual_len < FIXED_BATCH:
                padding = torch.zeros((FIXED_BATCH - actual_len, latent_dim)).to(device)
                z_chunk = torch.cat([z_chunk, padding], dim=0)

            # Now z_chunk is exactly [128, 2]
            pos_dist, qua_dist = vae_model.decode(z_chunk)

            # Extract Means
            p_m = pos_dist.mean  # Shape: [128, 3]
            #q_m = qua_dist.mean  # Shape: [128, 2] or [128, 4] ###---> Original line
            q_m = qua_dist.loc  # Shape: [128, 2] or [128, 4]

            combined = torch.cat([p_m, q_m], dim=-1)

            # --- UN-PADDING ---
            # Slice away the dummy padding variables
            decoded_full.append(combined[:actual_len])

    # 2. Combine everything directly into a 2D structure [N, RealSpaceDim]
    real_space_goals = torch.cat(decoded_full, dim=0).cpu().numpy()

    return real_space_goals


# ==========================================
# METRICS CALCULATORS
# ==========================================

# Especial evaluation metrics for > 30 paths
def calculate_ground_metrics_chuncked_or(z_pred, z_real, p2_goal, vae, density_threshold=1.5, chunk_size=25):
    """
    Evaluates N trajectories in chunks to avoid GPU Out-of-Memory (OOM) errors.

    z_pred:   [Batch, Time, 2]
    z_real:   [Batch, Time, 2]
    p2_goal:  [2] or [Batch, 2] (Consensus goal or individual goals)
    vae:      The VAE model
    """

    print(f"Calculating Riemannian metrics in chunks of {chunk_size}...")

    num_batches = z_pred.shape[0]
    num_time_steps = z_pred.shape[1]

    # Storage for chunk-wise results
    all_mse = []
    all_radial = []
    all_violation_rates = []
    all_energies_pred = []
    all_energies_gt = []

    ### To Correct MSE
    # Storage
    total_mse_sum = 0.0  # Changed from list to a running sum
    total_elements = 0  # Keep track of how many elements we've seen
    all_radial = []

    vae.eval()

    # Iterate through paths in chunks
    for i in range(0, num_batches, chunk_size):
        # 1. Slice Chunks
        end_idx = min(i + chunk_size, num_batches)
        zp_c = z_pred[i:end_idx]  # [Chunk, Time, 2]
        zr_c = z_real[i:end_idx]

        #print(f"end_idx leth: {end_idx:.4f}")
        #print(f"zp_c leth: {zp_c.shape}")
        #print(f"zr_c leth: {zr_c.shape}")

        # Handle p2_goal if it's a single point or a list
        pg_c = p2_goal if p2_goal.dim() == 1 else p2_goal[i:end_idx]

        # 2. Basic Errors (MSE & Radial)
        # MSE for this chunk
        #print(f"MSE manual")
        #mse_c = torch.mean((zp_c - zr_c) ** 2)
        #all_mse.append(mse_c)
        #print(f"MSE total: {mse_c:.4f}")
        chunk_sq_error_sum = torch.sum((zp_c - zr_c) ** 2).item()
        total_mse_sum += chunk_sq_error_sum
        total_elements += zp_c.numel()  # Counts all numbers in [Chunk, Time, 2]
        #print(f"MSE per path {i}: {chunk_sq_error_sum:.4f}")
        #print(f"MSE total until path {i}: {total_mse_sum:.4f}")
        #print(f"Total elements: {total_elements:.4f}")
        #print(f"True MSE per path {i}: {total_mse_sum / total_elements:.4f}")

        # Radial Goal Error
        z_final = zp_c[:, -1, :]
        radial_c = torch.norm(z_final - pg_c, dim=1)
        all_radial.append(radial_c)

        # 3. Energy & Violation Metrics (The Memory-Heavy Part)
        with torch.no_grad():
            # # Flatten Chunk for Jacobian [Chunk*Time, 2]
            # flat_p = zp_c.reshape(-1, 2)
            # flat_r = zr_c.reshape(-1, 2)
            #
            # # Get Metrics for Chunk
            # _, _, G_p = get_M(vae, flat_p.clone())
            # _, _, G_r = get_M(vae, flat_r.clone())
            #
            # G_p = torch.as_tensor(G_p)
            # G_r = torch.as_tensor(G_r)
            #
            # # Trace Density
            # dens_p = (G_p[:, 0, 0] + G_p[:, 1, 1]) / 2.0
            # dens_r = (G_r[:, 0, 0] + G_r[:, 1, 1]) / 2.0
            #
            # all_energies_pred.append(dens_p.mean().item())
            # all_energies_gt.append(dens_r.mean().item())
            #
            # # Violation Rate for this chunk
            # dens_p_bt = dens_p.reshape(-1, num_time_steps)  # [Chunk, Time]
            # violation_mask = dens_p_bt > (SAFE_BASELINE * density_threshold)
            # v_rate_c = (violation_mask.sum(dim=1).float() / num_time_steps)
            v_rate_c = torch.zeros(z_pred.shape)
            all_violation_rates.append(v_rate_c)

        # Clear GPU Cache after each chunk
        torch.cuda.empty_cache()

    # 4. Aggregate All Results
    radial_full = torch.cat(all_radial).detach().cpu().numpy()
    violation_full = torch.cat(all_violation_rates).detach().cpu().numpy()
    #mse_array = np.array(all_mse)
    true_mse = total_mse_sum / total_elements

    report = {
        #"MSE_Error": np.mean(all_mse),
        "MSE_Error": true_mse,
        "Radial_Error": radial_full.mean().item(),
        "Max_Radial": radial_full.max().item(),
        "Violation_Rate": violation_full.mean().item(),
        "Avg_Energy_Pred": np.mean(all_energies_pred),
        "Avg_Energy_GT": np.mean(all_energies_gt),
        "MSE_Mean": np.mean(true_mse),#(mse_array),
        "MSE_Std": np.std(true_mse),#(mse_array),
        "Radial_Mean": np.mean(radial_full),
        "Radial_Std": np.std(radial_full),
        "Violation_Mean": np.mean(violation_full),
        "Violation_Std": np.std(violation_full)
    }

    print(f"--- Final Metric Report ({num_batches} paths) ---")
    print(f"MSE = {report['MSE_Mean']:.4f} ± {report['MSE_Std']:.4f}")
    print(f"Radial = {report['Radial_Mean']:.4f} ± {report['Radial_Std']:.4f}")
    print(f"Radial Max = {report['Max_Radial']:.4f}")
    print(f"Violation = {report['Violation_Mean']:.4f} ± {report['Violation_Std']:.4f}")
    print(f"MSE: {report['MSE_Error']:.6f}")
    print(f"Radial Error: {report['Radial_Error']:.4f}")
    print(f"Violation Rate: {report['Violation_Rate'] * 100:.2f}%")
    print(f"Pred Energy: {report['Avg_Energy_Pred']:.4f} vs GT: {report['Avg_Energy_GT']:.4f}")

    return report
def calculate_euclidean_metrics_or(predicted_paths, ground_paths, predicted_goals, ground_goals):
    """
    Calculates standard physical metrics (MSE, radial error, final position error).
    """
    print("Calculating Euclidean metrics...")

    print(f"--- Final Euclidean Metric Report ({predicted_goals.shape[0]} paths) --- ")

    # ==========================================
    # 1. SPLIT DATA INTO POSITION & ORIENTATION
    # ==========================================
    # Trajectories: [N, T, 5]
    pred_pos_traj = predicted_paths[:, :, :2]  # First 2 columns (X, Y)
    true_pos_traj = ground_paths[:, :, :2]

    pred_ori_traj = predicted_paths[:, :, 2:]  # Last 3 columns (Orientation)
    true_ori_traj = ground_paths[:, :, 2:]

    # Goals: [N, 5]
    pred_pos_goal = predicted_goals[:, :2]
    true_pos_goal = ground_goals[:, :2]

    pred_ori_goal = predicted_goals[:, 2:]
    true_ori_goal = ground_goals[:, 2:]

    # ==========================================
    # 2. IMITATION LOSS (Full Trajectory Tracking)
    # ==========================================
    print(f"*** Imitation MSE in Euclidean Space ***")
    # Position Tracking MSE (Perfectly valid in flat Euclidean space)
    loss_imitation_pos = np.mean((pred_pos_traj - true_pos_traj) ** 2)
    print(f"Imitation Position MSE (Spatial): {loss_imitation_pos:.6f}")

    # Orientation Tracking Metric
    # Note: While still an approximation of rotation distance, separating it
    # isolates your angular tracking error from your spatial coordinates.
    loss_imitation_ori = np.mean((pred_ori_traj - true_ori_traj) ** 2)
    print(f"Imitation Orientation Metric Value:  {loss_imitation_ori:.6f}")

    # ==========================================
    # 3. GOAL LOSS (The Final Destination Precision)
    # ==========================================
    print(f"*** Goal MSE in Euclidean Space ***")
    # Spatial Arrival Precision
    loss_goal_pos = np.mean((pred_pos_goal - true_pos_goal) ** 2)
    print(f"Goal Position MSE:         {loss_goal_pos:.6f}")

    # Angular Arrival Precision
    loss_goal_ori = np.mean((pred_ori_goal - true_ori_goal) ** 2)
    print(f"Goal Orientation Metric:    {loss_goal_ori:.6f}")


def calculate_ground_metrics_chuncked(z_pred, z_real, p2_goal, vae, density_threshold=1.5, chunk_size=25):
    """
    Evaluates N trajectories in chunks to avoid GPU Out-of-Memory (OOM) errors.

    z_pred:   [Batch, Time, 2]
    z_real:   [Batch, Time, 2]
    p2_goal:  [2] or [Batch, 2] (Consensus goal or individual goals)
    vae:      The VAE model
    """
    print(f"Calculating Riemannian metrics in chunks of {chunk_size}...")

    num_batches = z_pred.shape[0]
    num_time_steps = z_pred.shape[1]

    # Storage for chunk-wise results
    all_radial = []
    all_violation_rates = []
    all_energies_pred = []
    all_energies_gt = []

    ### To Correct MSE
    # Storage
    total_mse_sum = 0.0  # Changed from list to a running sum
    total_elements = 0  # Keep track of how many elements we've seen

    vae.eval()

    # Iterate through paths in chunks
    for i in range(0, num_batches, chunk_size):
        # 1. Slice Chunks
        end_idx = min(i + chunk_size, num_batches)
        zp_c = z_pred[i:end_idx]  # [Chunk, Time, 2]
        zr_c = z_real[i:end_idx]

        # Handle p2_goal if it's a single point or a list
        pg_c = p2_goal if p2_goal.dim() == 1 else p2_goal[i:end_idx]

        # 2. Basic Errors (MSE & Radial)
        chunk_sq_error_sum = torch.sum((zp_c - zr_c) ** 2).item()
        total_mse_sum += chunk_sq_error_sum
        total_elements += zp_c.numel()  # Counts all numbers in [Chunk, Time, 2]

        # Radial Goal Error
        z_final = zp_c[:, -1, :]
        radial_c = torch.norm(z_final - pg_c, dim=1)
        all_radial.append(radial_c)

        # 3. Energy & Violation Metrics (The Memory-Heavy Part)
        with torch.no_grad():
            # ==============================================================
            # PRESERVED FOR COLLEAGUE: DO NOT DELETE
            # ==============================================================
            # # Flatten Chunk for Jacobian [Chunk*Time, 2]
            # flat_p = zp_c.reshape(-1, 2)
            # flat_r = zr_c.reshape(-1, 2)
            #
            # # Get Metrics for Chunk
            # _, _, G_p = get_M(vae, flat_p.clone())
            # _, _, G_r = get_M(vae, flat_r.clone())
            #
            # G_p = torch.as_tensor(G_p)
            # G_r = torch.as_tensor(G_r)
            #
            # # Trace Density
            # dens_p = (G_p[:, 0, 0] + G_p[:, 1, 1]) / 2.0
            # dens_r = (G_r[:, 0, 0] + G_r[:, 1, 1]) / 2.0
            #
            # all_energies_pred.append(dens_p.mean().item())
            # all_energies_gt.append(dens_r.mean().item())
            #
            # # Violation Rate for this chunk
            # dens_p_bt = dens_p.reshape(-1, num_time_steps)  # [Chunk, Time]
            # violation_mask = dens_p_bt > (SAFE_BASELINE * density_threshold)
            # v_rate_c = (violation_mask.sum(dim=1).float() / num_time_steps)
            # ==============================================================

            v_rate_c = torch.zeros(zp_c.shape[0])  # Adjusted to match chunk batch size
            all_violation_rates.append(v_rate_c)

        # Clear GPU Cache after each chunk
        torch.cuda.empty_cache()

    # 4. Aggregate All Results
    radial_full = torch.cat(all_radial).detach().cpu().numpy()
    violation_full = torch.cat(all_violation_rates).detach().cpu().numpy()
    true_mse = total_mse_sum / total_elements

    # Fallbacks for empty energy lists
    avg_energy_pred = np.mean(all_energies_pred) if all_energies_pred else 0.0
    avg_energy_gt = np.mean(all_energies_gt) if all_energies_gt else 0.0

    report = {
        "MSE_Error": true_mse,
        "Radial_Error": radial_full.mean().item(),
        "Max_Radial": radial_full.max().item(),
        "Violation_Rate": violation_full.mean().item(),
        "Avg_Energy_Pred": avg_energy_pred,
        "Avg_Energy_GT": avg_energy_gt,
        "MSE_Mean": np.mean(true_mse),
        "MSE_Std": np.std(true_mse),
        "Radial_Mean": np.mean(radial_full),
        "Radial_Std": np.std(radial_full),
        "Violation_Mean": np.mean(violation_full),
        "Violation_Std": np.std(violation_full)
    }

    print(f"--- Final Metric Report ({num_batches} paths) ---")
    print(f"MSE = {report['MSE_Mean']:.4f} ± {report['MSE_Std']:.4f}")
    print(f"Radial = {report['Radial_Mean']:.4f} ± {report['Radial_Std']:.4f}")
    print(f"Radial Max = {report['Max_Radial']:.4f}")
    print(f"Violation = {report['Violation_Mean']:.4f} ± {report['Violation_Std']:.4f}")
    print(f"MSE: {report['MSE_Error']:.6f}")
    print(f"Radial Error: {report['Radial_Error']:.4f}")
    print(f"Violation Rate: {report['Violation_Rate'] * 100:.2f}%")
    print(f"Pred Energy: {report['Avg_Energy_Pred']:.4f} vs GT: {report['Avg_Energy_GT']:.4f}")

    return report

def calculate_euclidean_metrics(predicted_paths, ground_paths, predicted_goals, ground_goals):
    """
    Calculates standard physical metrics (MSE, radial error, final position error).
    """
    print("Calculating Euclidean metrics...")
    print(f"--- Final Euclidean Metric Report ({predicted_goals.shape[0]} paths) --- ")

    # Ensure tensors are safely converted to numpy for the np.mean math
    if torch.is_tensor(predicted_paths): predicted_paths = predicted_paths.detach().cpu().numpy()
    if torch.is_tensor(ground_paths): ground_paths = ground_paths.detach().cpu().numpy()
    if torch.is_tensor(predicted_goals): predicted_goals = predicted_goals.detach().cpu().numpy()
    if torch.is_tensor(ground_goals): ground_goals = ground_goals.detach().cpu().numpy()

    # ==========================================
    # 1. SPLIT DATA INTO POSITION & ORIENTATION
    # ==========================================
    # Trajectories: [N, T, 5]
    pred_pos_traj = predicted_paths[:, :, :2]  # First 2 columns (X, Y)
    true_pos_traj = ground_paths[:, :, :2]

    pred_ori_traj = predicted_paths[:, :, 2:]  # Last 3 columns (Orientation)
    true_ori_traj = ground_paths[:, :, 2:]

    # Goals: [N, 5]
    pred_pos_goal = predicted_goals[:, :2]
    true_pos_goal = ground_goals[:, :2]

    pred_ori_goal = predicted_goals[:, 2:]
    true_ori_goal = ground_goals[:, 2:]

    # ==========================================
    # 2. IMITATION LOSS (Full Trajectory Tracking)
    # ==========================================
    print(f"*** Imitation MSE in Euclidean Space ***")
    loss_imitation_pos = np.mean((pred_pos_traj - true_pos_traj) ** 2)
    print(f"Imitation Position MSE (Spatial): {loss_imitation_pos:.6f}")

    loss_imitation_ori = np.mean((pred_ori_traj - true_ori_traj) ** 2)
    print(f"Imitation Orientation Metric Value:  {loss_imitation_ori:.6f}")

    # ==========================================
    # 3. GOAL LOSS (The Final Destination Precision)
    # ==========================================
    print(f"*** Goal MSE in Euclidean Space ***")
    loss_goal_pos = np.mean((pred_pos_goal - true_pos_goal) ** 2)
    print(f"Goal Position MSE:         {loss_goal_pos:.6f}")

    loss_goal_ori = np.mean((pred_ori_goal - true_ori_goal) ** 2)
    print(f"Goal Orientation Metric:    {loss_goal_ori:.6f}")

    # NEW: Return the values so eval_rnode can save them!
    return {
        "Imitation_Pos_MSE": float(loss_imitation_pos),
        "Imitation_Ori_MSE": float(loss_imitation_ori),
        "Goal_Pos_MSE": float(loss_goal_pos),
        "Goal_Ori_MSE": float(loss_goal_ori)
    }



# ==========================================
# AMBIENT METRICS (against the raw demonstrations)
# ==========================================
def to_original_units(decoded_paths, xy_center, xy_scale):
    """Undo normalize_newVAE on decoded paths [..., dof]: xy * xy_scale + xy_center, rest unchanged."""
    paths = np.array(decoded_paths, dtype=np.float64, copy=True)
    paths[..., 0:2] = paths[..., 0:2] * float(xy_scale) + np.asarray(xy_center, dtype=np.float64)
    return paths


def resample_by_arc_length(paths, num_points=None):
    """Resample each path [B, T, d] to num_points points spaced uniformly in arc length."""
    B, T, d = paths.shape
    num_points = num_points or T
    s_new = np.linspace(0.0, 1.0, num_points)
    out = np.empty((B, num_points, d))
    for i in range(B):
        seg = np.linalg.norm(np.diff(paths[i], axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] <= 0:  # degenerate path that never moves
            out[i] = paths[i, :1]
            continue
        s = s / s[-1]
        for k in range(d):
            out[i, :, k] = np.interp(s_new, s, paths[i, :, k])
    return out


def orientation_angle(q_pred, q_true):
    """Angle in radians between orientations [..., k]: 2*arccos(|<q,q'>|) for quaternions (k=4,
    the absolute value handles the q/-q ambiguity), the geodesic arccos(<q,q'>) on S2 (k=3)."""
    q_pred = q_pred / np.linalg.norm(q_pred, axis=-1, keepdims=True)
    q_true = q_true / np.linalg.norm(q_true, axis=-1, keepdims=True)
    dot = np.sum(q_pred * q_true, axis=-1)
    if q_pred.shape[-1] == 4:
        return 2.0 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0))
    return np.arccos(np.clip(dot, -1.0, 1.0))


def calculate_ambient_metrics(decoded_paths, x_ambient, xy_center, xy_scale, pos_dof):
    """
    Goal / imitation errors of decoded paths against the raw demonstration segments, in original units.

    decoded_paths: [B, T, dof] decoder means (normalized units, decode_latente_paths)
    x_ambient:     [B, T, dof] raw demonstration segments (original units)
    Position errors are squared Euclidean norms over the pos_dof coordinates (summed, not averaged).
    """
    if torch.is_tensor(x_ambient): x_ambient = x_ambient.detach().cpu().numpy()
    pred = to_original_units(decoded_paths, xy_center, xy_scale)
    true = np.asarray(x_ambient, dtype=np.float64)

    pred_pos, true_pos = pred[:, :, :pos_dof], true[:, :, :pos_dof]
    pred_ori, true_ori = pred[:, :, pos_dof:], true[:, :, pos_dof:]

    goal_sq = np.sum((pred_pos[:, -1] - true_pos[:, -1]) ** 2, axis=-1)                  # [B]
    imit_sq = np.sum((pred_pos - true_pos) ** 2, axis=-1)                                  # [B, T]
    # Time-free: same MSE after both curves are resampled uniformly in arc length
    arc_sq = np.sum((resample_by_arc_length(pred_pos) - resample_by_arc_length(true_pos)) ** 2, axis=-1)

    goal_ang = np.degrees(orientation_angle(pred_ori[:, -1], true_ori[:, -1]))             # [B]
    imit_ang = np.degrees(orientation_angle(pred_ori, true_ori))                           # [B, T]

    report = {
        "Goal_MSE": float(goal_sq.mean()),
        "Imitation_MSE": float(imit_sq.mean()),
        "Imitation_MSE_ArcLength": float(arc_sq.mean()),
        "Goal_Ori_Angle_deg": float(goal_ang.mean()),
        "Imitation_Ori_Angle_deg": float(imit_ang.mean()),
        # Per-query spreads, for error bars
        "Goal_MSE_Std": float(goal_sq.std()),
        "Imitation_MSE_Std": float(imit_sq.mean(axis=1).std()),
        "Imitation_MSE_ArcLength_Std": float(arc_sq.mean(axis=1).std()),
    }

    print(f"--- Ambient Metric Report ({pred.shape[0]} paths, original units) ---")
    print(f"Goal MSE:                 {report['Goal_MSE']:.6f}")
    print(f"Imitation MSE:            {report['Imitation_MSE']:.6f}")
    print(f"Imitation MSE (arc len):  {report['Imitation_MSE_ArcLength']:.6f}")
    print(f"Goal orientation angle:   {report['Goal_Ori_Angle_deg']:.3f} deg")
    print(f"Imitation orient. angle:  {report['Imitation_Ori_Angle_deg']:.3f} deg")
    return report


# Keys of calculate_ground_metrics_chuncked that are placeholders until the density code is restored
_PLACEHOLDER_KEYS = ("Violation_Rate", "Violation_Mean", "Violation_Std", "Avg_Energy_Pred", "Avg_Energy_GT")


def evaluate_predictions(pred_latent, dense_latent, bench, vae_model):
    """
    All benchmark metrics of one method's latent paths.

    pred_latent:  [B, T, 2] latent paths at the ground truth's T time steps
    dense_latent: [B, S, 2] the same paths sampled densely (S = 200) for the off-manifold %
    bench:        the shared benchmark inputs built by run_benchmark.py
    """
    from benchmark.off_manifold import off_manifold_rates

    ground_truth = bench['ground_truth']
    pred_latent = torch.as_tensor(pred_latent, dtype=torch.float32).to(ground_truth.device)

    # A method that stopped early (e.g. stochman's break on failure) is scored on the queries it answered
    n = pred_latent.shape[0]
    if n < ground_truth.shape[0]:
        print(f"⚠️ Only {n}/{ground_truth.shape[0]} queries answered: metrics cover those only")

    print("\n📊 Calculating Latent Space Metrics...")
    latent_metrics = calculate_ground_metrics_chuncked(
        pred_latent, ground_truth[:n], bench['z2'][:n], vae_model, density_threshold=1.5, chunk_size=50
    )
    latent_metrics = {k: v for k, v in latent_metrics.items() if k not in _PLACEHOLDER_KEYS}

    print("\n📏 Decoding to Ambient Space...")
    ambient_metrics = calculate_ambient_metrics(
        decode_latente_paths(vae_model, pred_latent), bench['x_ambient'][:n],
        bench['xy_center'], bench['xy_scale'], vae_model.pos_dof
    )

    print("\n🧭 Calculating Off-manifold %...")
    off_manifold = off_manifold_rates(vae_model, dense_latent, bench['off_ref'])

    return {
        "num_queries_answered": n,
        "latent_metrics": latent_metrics,
        "ambient_metrics": ambient_metrics,
        "off_manifold": off_manifold,
    }
