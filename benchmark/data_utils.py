import os
import torch

from node.data.preprocessing import get_demonstrations_paths, interpolate_trajectories, encode_demonstrations_paths, \
    create_universal_segmented_dataset, unify_time_steps, get_demonstrations_paths_newVAE


# Data preparation for experiments
def get_ground_experiment_data(vae_model, ):
    print("Testing real demos...")
    real_space_paths = get_demonstrations_paths(vae_model)
    # print("# of Paths: " + str(len(real_space_paths)))
    # print("Paths shape: " + str(real_space_paths[0].shape))
    # print("Points values: " + str(real_space_paths[0][0]))
    # print("Points values: " + str(real_space_paths[0][-1]))
    #
    # print("Testing interpolation...")
    # print("Real paths shape: " + str(real_space_paths[0].shape))
    interpolated_paths = interpolate_trajectories(real_space_paths, 500)
    # print("# of paths: " + str(len(interpolated_paths)))
    # print("Interpolated Paths shape: " + str(interpolated_paths[0].shape))
    #
    # print("Testing latent demos...")
    # print("Interpolated Paths shape: " + str(interpolated_paths[0].shape))
    latent_space_paths = encode_demonstrations_paths(vae_model, interpolated_paths)
    # print("# of paths: " + str(len(latent_space_paths)))
    # print("Latent Paths shape: " + str(latent_space_paths[0].shape))

    # print("Encoding")
    # print("Testing latent demos unified time steps...")
    # latent_space_paths1 = encode_demonstrations_paths(vae_model, torch.tensor(real_space_paths))
    # latent_space_paths2 = encode_demonstrations_paths(vae_model, interpolated_paths)
    # print("# of paths: " + str(len(latent_space_paths1)))
    # print("Latent Paths shape: " + str(latent_space_paths1[0].shape))
    # print("# of paths: " + str(len(latent_space_paths2)))
    # print("Latent Paths shape: " + str(latent_space_paths2[0].shape))

    print("Segmentation")
    segmented_paths = create_universal_segmented_dataset(latent_space_paths, min_window=50, max_window=480, samples_per_path=50)
    #plot_trajectories(segmented_paths)
    print("# of paths: " + str(len(segmented_paths)))
    print("Latent Paths shape: " + str(segmented_paths[0].shape))
    #create_universal_segmented_dataset(encoded_paths, min_window=20, max_window=150, samples_per_path=499)

    print("DownSampling")
    unified = unify_time_steps(segmented_paths, time_steps=100)
    print("# of paths: " + str(len(unified)))
    print("Latent Paths shape: " + str(unified[0].shape))

    ground = unified
    new_ground = torch.stack(ground)  # Shape: [500, Variable, 2]
    #new_ground = torch.stack(latent_space_paths)
    test = latent_space_paths

    # plot_trajectories(real_space_paths)
    # plot_trajectories(interpolated_paths)
    # plot_trajectories(latent_space_paths)
    # plot_trajectories(test)
    return new_ground, test


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

# Export of the experiment predictions
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


def load_test_config(filename="experiment_data.pt", directory="../../RNODE/test_datasets/Experiment6"):
    """
    Loads the saved test coordinates and moves them to the specified device.
    """
    full_path = os.path.join(directory, filename)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"❌ No file found at {full_path}")

    # 1. Load the dictionary
    data = torch.load(full_path)

    # 2. Extract and move to device (GPU/CPU)
    z1 = data['z1']
    z2 = data['z2']
    ground_truth = data['ground_truth']

    print(f"✅ Loaded benchmark data from {full_path}")
    print(f"📊 Benchmark Ground Truth Shape: {ground_truth.shape}")

    return z1, z2, ground_truth
