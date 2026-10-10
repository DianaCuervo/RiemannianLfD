import os
import json
import numpy as np
import torch
from scipy.io import loadmat
import glob
import random

from node.data.preprocessing import get_demonstrations_paths, interpolate_trajectories, encode_demonstrations_paths, \
    create_universal_segmented_dataset, unify_time_steps, get_demonstrations_paths_newVAE
from node.utils.plots import plot_trajectories


# Load lasa demonstrations with the normalization constants
def load_lasa_demos(origin_dir, origin_file, trajectory_number):
    """Load LASA demos and return them normalized, plus the constants needed to undo it.

    Mirrors node/data/preprocessing.py's get_demonstrations_paths_newVAE + normalize_newVAE
    exactly -- one isotropic normalization shared across all demos -- but also hands back
    xy_center/xy_scale, which preprocessing computes locally and discards. Without them we
    cannot map a decoded trajectory back to real LASA units.
    """
    demoUQ = loadmat(f"{origin_dir}/{origin_file}")["demoUQ"]

    raw_positions = [demoUQ[0, i]["tsPos"][0, 0].T for i in range(trajectory_number)]
    raw_quats = [demoUQ[0, i]["quat"][0, 0].T for i in range(trajectory_number)]

    all_xy = np.vstack([pos[:, 0:2] for pos in raw_positions])
    xy_min, xy_max = all_xy.min(axis=0), all_xy.max(axis=0)
    xy_center = (xy_min + xy_max) / 2
    xy_scale = (xy_max - xy_min).max() / 2  # single scalar -> isotropic scaling

    demos = []
    for pos, quat in zip(raw_positions, raw_quats):
        xy_norm = (pos[:, 0:2] - xy_center) / xy_scale
        demos.append(np.hstack([xy_norm, pos[:, 2:3], quat]))

    return demos, xy_center, xy_scale

# Generate the benchmark datasets by dataset
def generate_benchmark_original_dataset(dataset_cfg, shape_name, vae_model, device):
    """
    Acts just like build_dataset_offline, but formats the output as a single benchmark
    reference: the latent ground truth plus the raw demonstration segment behind each query.

    Returns a dict with
        latent:     [B, T, latent_dim] encoded demonstration segments (z1/z2 are its ends)
        x_ambient:  [B, T, dof] the same segments in original units (normalization undone)
        xy_center, xy_scale: the normalize_newVAE constants, to un-normalize decoded paths
    """
    dataset_type = dataset_cfg['type']

    # 1. Load Real Paths (normalized, as the VAE saw them)
    if dataset_type == 'toy':
        real_paths = get_demonstrations_paths(
            origin_dir=dataset_cfg['origin_dir'],
            trajectory_number=dataset_cfg['trajectory_number'],
            test_id=dataset_cfg['test_id'],
            r2_letter=dataset_cfg['r2_letter'],
            s2_letter=dataset_cfg['s2_letter']
        )
        real_paths = interpolate_trajectories(real_paths,
                                              interpolation_points=dataset_cfg['interpolation_points'])
        # The toy demos are not normalized
        xy_center, xy_scale = np.zeros(2), 1.0

    elif dataset_type == 'lasa':
        origin_file = dataset_cfg['origin_file'].replace('{shape}', shape_name)
        real_paths = get_demonstrations_paths_newVAE(
            origin_dir=dataset_cfg['origin_dir'],
            origin_file=origin_file,
            trajectory_number=dataset_cfg['trajectory_number']
        )
        _, xy_center, xy_scale = load_lasa_demos(dataset_cfg['origin_dir'], origin_file,
                                                 dataset_cfg['trajectory_number'])

    plot_trajectories(real_paths)
    latent_paths = encode_demonstrations_paths(vae_model, real_paths, device)

    # 2. Segment and Unify latent and raw paths together, so both share the same indices
    joint_paths = []
    for latent, real in zip(latent_paths, real_paths):
        raw = torch.as_tensor(np.asarray(real), dtype=torch.float32).clone()
        raw[:, 0:2] = raw[:, 0:2] * float(xy_scale) + torch.as_tensor(xy_center, dtype=torch.float32)
        joint_paths.append(torch.cat([latent.detach().cpu(), raw], dim=1))
    latent_dim = latent_paths[0].shape[1]
    print(f'Joint_paths shape: {len(joint_paths)}')

    ### Encoded benchmark data creation
    # 1. Locate the directory where the Neural ODE training data was saved
    processed_dir = dataset_cfg['save_dir_node_processed'].replace('{shape}', shape_name)
    # 2. Load the (Start, Goal) pairs that the model has already seen
    print(f"\n--- 1. Loading Training Data Points from {processed_dir} ---")
    existing_starts, existing_goals = load_existing_start_goals(processed_dir)
    # 3. Generate the novel benchmark segments safely
    print("\n--- 3. Generating Benchmark Paths ---")
    benchmark_segments = create_benchmark_segments(
        encoded_paths=joint_paths,
        latent_dimension=latent_dim,
        existing_starts=existing_starts,
        existing_goals=existing_goals,
        min_window=dataset_cfg['min_window'],
        max_window=dataset_cfg['max_window'],
        samples_per_path=dataset_cfg['samples_per_path']
    )
    # 5. Unify time steps for the benchmark evaluation
    final_benchmark_paths = unify_time_steps(benchmark_segments, time_steps=dataset_cfg['time_steps'])

    final_paths = torch.stack(final_benchmark_paths)

    # 3. Split back into latent ground truth [B, T, 2] and ambient reference [B, T, dof]
    return {
        'latent': final_paths[:, :, :latent_dim],
        'x_ambient': final_paths[:, :, latent_dim:],
        'xy_center': torch.as_tensor(xy_center, dtype=torch.float32),
        'xy_scale': float(xy_scale),
    }

def load_existing_start_goals(processed_dir):
    existing_starts = []
    existing_goals = []

    file_paths = glob.glob(os.path.join(processed_dir, "*.pt"))

    for fp in file_paths:
        data = torch.load(fp, weights_only=True)
        if isinstance(data, dict) and 'z' in data:
            traj = data['z']
        elif isinstance(data, torch.Tensor):
            traj = data
        else:
            continue

        existing_starts.append(traj[0])  # Start point
        existing_goals.append(traj[-1])  # Goal point

    if not existing_starts:
        print(f"⚠️ No existing data found in {processed_dir}")
        return None, None

    starts_tensor = torch.stack(existing_starts)
    goals_tensor = torch.stack(existing_goals)

    print("\n" + "=" * 50)
    print(f"📥 LOADED TRAINING DATA:")
    print(f"   - Found {len(file_paths)} training files.")
    print(f"   - Extracted {starts_tensor.shape[0]} Start points.")
    print(f"   - Extracted {goals_tensor.shape[0]} Goal points.")
    print(f"   - Total (Start, Goal) pairs to check against: {starts_tensor.shape[0]}")
    print("=" * 50 + "\n")

    return starts_tensor, goals_tensor

def create_benchmark_segments(encoded_paths, latent_dimension=2, existing_starts=None, existing_goals=None,
                              min_window=150, max_window=950, samples_per_path=50, tolerance=1e-5):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    segmented_data = []
    total_duplicates_rejected = 0  # Tracker for rejected segments

    for path_idx, path in enumerate(encoded_paths):
        num_steps = path.shape[0]
        valid_samples_found = 0
        attempts = 0
        max_attempts = samples_per_path * 10

        while valid_samples_found < samples_per_path and attempts < max_attempts:
            attempts += 1

            roll = random.random()
            if roll < 0.30:
                win_size = random.randint(min_window, (num_steps // 2) - 150)
            elif roll < 0.65:
                win_size = random.randint(((num_steps // 2) - 150) + 1, (num_steps // 2) + 150)
            else:
                win_size = random.randint(((num_steps // 2) + 150) + 1, max_window)

            if num_steps <= win_size:
                continue

            start = np.random.randint(0, num_steps - win_size)
            end = start + win_size

            candidate_start = path[start]
            candidate_goal = path[end - 1]

            # VERIFICATION
            if existing_starts is not None and existing_goals is not None:
                start_dists = torch.norm(existing_starts - candidate_start[:latent_dimension], dim=1)
                goal_dists = torch.norm(existing_goals - candidate_goal[:latent_dimension], dim=1)

                # The '&' ensures they match on the exact same index (a tied pair)
                is_duplicate_task = ((start_dists < tolerance) & (goal_dists < tolerance)).any()

                if is_duplicate_task:
                    total_duplicates_rejected += 1
                    continue

            z_segment = path[start:end, :].clone()
            segmented_data.append(z_segment)
            valid_samples_found += 1

        print(f"Path {path_idx + 1}/{len(encoded_paths)}: Generated {valid_samples_found} pairs start-goal points.")

    print("\n" + "=" * 50)
    print(f"  BENCHMARK GENERATION COMPLETE:")
    print(f"   - Exact duplicate (Start, Goal) pairs rejected: {total_duplicates_rejected}")
    print(f"   - Final unseen benchmark pairs created: {len(segmented_data)}")
    print("=" * 50 + "\n")

    return segmented_data

def generate_benchmark_noisy_dataset(dataset_cfg, shape_name, vae_model, device):
    """
    Acts just like build_dataset_offline, but formats the output as a single benchmark
    reference: the latent ground truth plus the raw demonstration segment behind each query.

    Returns a dict with
        latent:     [B, T, latent_dim] encoded demonstration segments (z1/z2 are its ends)
        x_ambient:  [B, T, dof] the same segments in original units (normalization undone)
        xy_center, xy_scale: the normalize_newVAE constants, to un-normalize decoded paths
    """
    dataset_type = dataset_cfg['type']

    # 1. Load Real Paths (normalized, as the VAE saw them)
    if dataset_type == 'toy':
        real_paths = get_demonstrations_paths(
            origin_dir=dataset_cfg['origin_dir'],
            trajectory_number=dataset_cfg['trajectory_number'],
            test_id=dataset_cfg['test_id'],
            r2_letter=dataset_cfg['r2_letter'],
            s2_letter=dataset_cfg['s2_letter']
        )
        real_paths = interpolate_trajectories(real_paths,
                                              interpolation_points=dataset_cfg['interpolation_points'])
        # The toy demos are not normalized
        xy_center, xy_scale = np.zeros(2), 1.0

    elif dataset_type == 'lasa':
        origin_file = dataset_cfg['origin_file'].replace('{shape}', shape_name)
        real_paths = get_demonstrations_paths_newVAE(
            origin_dir=dataset_cfg['origin_dir'],
            origin_file=origin_file,
            trajectory_number=dataset_cfg['trajectory_number']
        )
        _, xy_center, xy_scale = load_lasa_demos(dataset_cfg['origin_dir'], origin_file,
                                                 dataset_cfg['trajectory_number'])

    # --- NEW: 2. Inject Noise to Generate Novel Trajectories ---
    # Pull the noise standard deviation from config, defaulting to 0.05 if not set
    noise_std = dataset_cfg.get('noise_std', 0.001)
    print(f"\n--- 2. Applying Gaussian Noise (std={noise_std}) to Original Paths ---")

    noisy_real_paths = []
    for path in real_paths:
        path_array = np.asarray(path)
        # Generate noise with the exact same shape as the trajectory
        noise = np.random.normal(loc=0.0, scale=noise_std, size=(1, path_array.shape[1]))
        noisy_real_paths.append(path_array + noise)

    # Replace the original paths with our new noisy variants
    real_paths = noisy_real_paths
    plot_trajectories(real_paths)
    latent_paths = encode_demonstrations_paths(vae_model, real_paths, device)

    # 2. Segment and Unify latent and raw paths together, so both share the same indices
    joint_paths = []
    for latent, real in zip(latent_paths, real_paths):
        raw = torch.as_tensor(np.asarray(real), dtype=torch.float32).clone()
        raw[:, 0:2] = raw[:, 0:2] * float(xy_scale) + torch.as_tensor(xy_center, dtype=torch.float32)
        joint_paths.append(torch.cat([latent.detach().cpu(), raw], dim=1))
    latent_dim = latent_paths[0].shape[1]
    print(f'Joint_paths shape: {len(joint_paths)}')

    ### Encoded benchmark data creation
    # 1. Locate the directory where the Neural ODE training data was saved
    processed_dir = dataset_cfg['save_dir_node_processed'].replace('{shape}', shape_name)
    # 2. Load the (Start, Goal) pairs that the model has already seen
    print(f"\n--- 1. Loading Training Data Points from {processed_dir} ---")
    existing_starts, existing_goals = load_existing_start_goals(processed_dir)
    # 3. Generate the novel benchmark segments safely
    print("\n--- 3. Generating Benchmark Paths ---")
    benchmark_segments = create_benchmark_segments(
        encoded_paths=joint_paths,
        latent_dimension=latent_dim,
        existing_starts=existing_starts,
        existing_goals=existing_goals,
        min_window=dataset_cfg['min_window'],
        max_window=dataset_cfg['max_window'],
        samples_per_path=dataset_cfg['samples_per_path']
    )
    # 5. Unify time steps for the benchmark evaluation
    final_benchmark_paths = unify_time_steps(benchmark_segments, time_steps=dataset_cfg['time_steps'])

    final_paths = torch.stack(final_benchmark_paths)

    # 3. Split back into latent ground truth [B, T, 2] and ambient reference [B, T, dof]
    return {
        'latent': final_paths[:, :, :latent_dim],
        'x_ambient': final_paths[:, :, latent_dim:],
        'xy_center': torch.as_tensor(xy_center, dtype=torch.float32),
        'xy_scale': float(xy_scale),
    }

# Export the benchmark datasets
def save_test_config(z1, z2, ground_truth, filename="ground_truth.pt", directory="./benchmark/data", extras=None):
    """
    Saves the essential test coordinates to disk.
    z1: [Batch, 2] or [2]
    z2: [Batch, 2] or [2]
    ground_truth: [Batch, Time, 2] or [Time, 2]
    extras: optional dict of further tensors/floats to store (e.g. x_ambient, xy_center, xy_scale)
    """
    # 1. Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"📁 Created new directory: {directory}")

    # 2. Prepare the full file path
    full_path = os.path.join(directory, filename)

    data_to_save = {
        'z1': z1.detach().cpu(),
        'z2': z2.detach().cpu(),
        'ground_truth': ground_truth.detach().cpu(),
        'description': "Test set for Riemannian Proxy-Geodesic comparison"
    }
    if extras:
        data_to_save.update(extras)
    # 4. Save
    torch.save(data_to_save, full_path)
    print(f"✅ Benchmark data successfully archived at: {full_path}")

# Load and preparation the benchmark data
def load_test_config(filename="ground_truth.pt", directory="/benchmark/data"):
    """
    Loads the saved test coordinates and moves them to the specified device.
    """
    full_path = os.path.join(directory, filename)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"❌ No file found at {full_path}")

    # 1. Load the dictionary
    data = torch.load(full_path, weights_only=True)

    # 2. Extract and move to device (GPU/CPU)
    z1 = data['z1']
    z2 = data['z2']
    ground_truth = data['ground_truth']

    print(f"✅ Loaded benchmark data from {full_path}")
    print(f"📊 Benchmark Ground Truth Shape: {ground_truth.shape}")

    return z1, z2, ground_truth

# Numpy-to-JSON Translator
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()  # Converts numpy float/int to standard Python float/int
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif torch.is_tensor(obj):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def save_json(report, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(report, f, indent=4, cls=NumpyEncoder)
    print(f"✅ Saved: {path}")

# Load the full benchmark reference (latent ground truth + ambient demonstrations)
def load_benchmark_reference(gt_data_path):
    """The benchmark file as a dict: z1, z2, ground_truth, x_ambient, xy_center, xy_scale."""
    if not os.path.exists(gt_data_path):
        raise FileNotFoundError(f"❌ No file found at {gt_data_path}")

    data = torch.load(gt_data_path, weights_only=True)
    if 'x_ambient' not in data:
        raise ValueError(f"❌ {gt_data_path} has no ambient reference (made by an older version): "
                         f"delete it so run_benchmark.py regenerates it")

    print(f"✅ Loaded benchmark reference from {gt_data_path}")
    print(f"📊 Latent ground truth: {tuple(data['ground_truth'].shape)} | "
          f"Ambient reference: {tuple(data['x_ambient'].shape)}")
    return data

# Function to save graph paths as .txt files or .csv files
def save_latent_paths_node(proxy_geodesics, dir_name="./benchmark/results", file_name="node_latent_paths_BM.pt"):
    """
    Consolidates 3,500 paths into a single structured file for efficiency.

    Args:
        proxy_geodesics: List of tensors or arrays.
        dir_name: Directory to store the bundle.
        file_name: The name of the aggregate file.
    """
    # 1. Ensure directory exists
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    processed_bundle = []

    for traj in proxy_geodesics:
        # Convert to torch tensor if it's numpy
        t = torch.as_tensor(traj)

        # 2. Reshape logic to ensure (1, 100, 2)
        # If input is [100, 2], it becomes [1, 100, 2]
        if t.dim() == 2:
            t = t.unsqueeze(0)
        # If it's already [1, 100, 2], this ensures it's correct
        elif t.dim() == 3 and t.shape[0] != 1:
            # Handle cases where batch might be squeezed
            t = t[0:1, :, :]

        processed_bundle.append(t.detach().cpu())

    # 3. Stack into a single giant tensor [3500, 100, 2]
    # OR keep as a list of [1, 100, 2] tensors.
    # Stacking is usually better for Batch processing later.
    final_data = torch.cat(processed_bundle, dim=0) # Result: [3500, 100, 2]

    # 4. Save to disk
    save_path = os.path.join(dir_name, file_name)
    torch.save(final_data, save_path)

    print(f"✅ NODE Proxy-Geodesics Exported! with Shape: {final_data.shape}")
    print(f"Successfully exported: {save_path}")

# Function to save stochman paths as .txt files or .csv files
def save_latent_paths_stochman(trajectories, dir_name="./benchmark/results", file_name="stochman_latent_paths_BM.pt"):

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    processed_bundle = []
    t_eval = torch.linspace(0, 1, 100)  # Standard 100 steps

    for traj in trajectories:
        # CHECK: If it's a spline object, evaluate it first
        if hasattr(traj, 'begin') or "CubicSpline" in str(type(traj)):
            t = traj(t_eval.to(traj.params.device))
        else:
            t = torch.as_tensor(traj)

        # Ensure (1, 100, 2) shape
        if t.dim() == 2:
            t = t.unsqueeze(0)

        processed_bundle.append(t.detach().cpu())

    final_data = torch.cat(processed_bundle, dim=0)
    torch.save(final_data, os.path.join(dir_name, file_name))
    print(f"✅ Saved {final_data.shape[0]} paths to {file_name}")

# Function to save graph paths as .txt files or .csv files
def save_latent_paths_graph(trajectories, dir_name="./benchmark/results", file_name="graph_latent_paths_BM.pt"):
    """
    Consolidates 3,500 paths into a single structured file for efficiency.

    Args:
        trajectories: List of tensors or arrays.
        dir_name: Directory to store the bundle.
        file_name: The name of the aggregate file.
    """
    # 1. Ensure directory exists
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    processed_bundle = []

    for traj in trajectories:
        # Convert to torch tensor if it's numpy
        t = torch.as_tensor(traj)

        # 2. Reshape logic to ensure (1, 100, 2)
        # If input is [100, 2], it becomes [1, 100, 2]
        if t.dim() == 2:
            t = t.unsqueeze(0)
        # If it's already [1, 100, 2], this ensures it's correct
        elif t.dim() == 3 and t.shape[0] != 1:
            # Handle cases where batch might be squeezed
            t = t[0:1, :, :]

        processed_bundle.append(t.detach().cpu())

    # 3. Stack into a single giant tensor [3500, 100, 2]
    # OR keep as a list of [1, 100, 2] tensors.
    # Stacking is usually better for Batch processing later.
    final_data = torch.cat(processed_bundle, dim=0) # Result: [3500, 100, 2]

    # 4. Save to disk
    save_path = os.path.join(dir_name, file_name)
    torch.save(final_data, save_path)

    print(f"✅ Discrete Latent Paths Exported! Shape: {final_data.shape}")
    print(f"Successfully exported: {save_path}")
