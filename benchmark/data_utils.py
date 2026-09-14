import os
import torch

from node.data.preprocessing import get_demonstrations_paths, interpolate_trajectories, encode_demonstrations_paths, \
    create_universal_segmented_dataset, unify_time_steps, get_demonstrations_paths_newVAE

# Generate the benchmark datasets by dataset
def generate_benchmark_dataset(dataset_cfg, shape_name, vae_model, device):
    """
    Acts just like build_dataset_offline, but formats and saves the output
    as a single ground_truth.pt file specifically for benchmarking.
    """
    dataset_type = dataset_cfg['type']

    # 1. Load Real Paths
    if dataset_type == 'toy':
        real_paths = get_demonstrations_paths(
            origin_dir=dataset_cfg['origin_dir'],
            trajectory_number=dataset_cfg['trajectory_number'],
            test_id=dataset_cfg['test_id'],
            r2_letter=dataset_cfg['r2_letter'],
            s2_letter=dataset_cfg['s2_letter']
        )
        interpolated_paths = interpolate_trajectories(real_paths,
                                                      interpolation_points=dataset_cfg['interpolation_points'])
        latent_paths = encode_demonstrations_paths(vae_model, interpolated_paths, device)

    elif dataset_type == 'lasa':
        origin_file = dataset_cfg['origin_file'].replace('{shape}', shape_name)
        real_paths = get_demonstrations_paths_newVAE(
            origin_dir=dataset_cfg['origin_dir'],
            origin_file=origin_file,
            trajectory_number=dataset_cfg['trajectory_number']
        )
        latent_paths = encode_demonstrations_paths(vae_model, real_paths, device)

    # 2. Segment and Unify (This uses the benchmark config to generate 3500 paths!)
    segmented_paths = create_universal_segmented_dataset(
        latent_paths,
        min_window=dataset_cfg['min_window'],
        max_window=dataset_cfg['max_window'],
        samples_per_path=dataset_cfg['samples_per_path']
    )
    final_paths = unify_time_steps(segmented_paths, time_steps=dataset_cfg['time_steps'])

    # 3. Format as a single Ground Truth Tensor [3500, Time, 2]
    new_ground = torch.stack(final_paths)

    return new_ground

# Export the benchmark datasets
def save_test_config(z1, z2, ground_truth, filename="ground_truth.pt", directory="./benchmark/data"):
    """
    Saves the essential test coordinates to disk.
    z1: [Batch, 2] or [2]
    z2: [Batch, 2] or [2]
    ground_truth: [Batch, Time, 2] or [Time, 2]
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
