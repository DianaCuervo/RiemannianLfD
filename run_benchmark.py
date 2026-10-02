import argparse
import os
import yaml
import torch
import copy

from benchmark.data_utils import generate_benchmark_dataset, save_test_config, load_test_config
from benchmark.eval_graph import run_graph_benchmark
from benchmark.eval_node import run_node_benchmark
from benchmark.eval_stochman import run_stochman_benchmark
from node.data.dataset import prepare_loaders
from node.utils.plots import visualize_metric, plot_trajectories_on_manifold, plot_trajectories
from vae.vae_model import load_pretrained_vae
import matplotlib.pyplot as plt


# ==========================================
# CONFIGURATION LOADERS
# ==========================================

def load_benchmark_config(config_path, dataset_type):
    """Loads the benchmark YAML structure and preserves root keys."""
    with open(config_path, 'r') as file:
        full_config = yaml.safe_load(file)

    if 'datasets' in full_config and dataset_type in full_config['datasets']:
        # 1. Get the specific dataset block (e.g., 'toy')
        dataset_cfg = full_config['datasets'][dataset_type]

        # 2. Attach all top-level keys (results_base_dir, frameworks, etc.)
        for key, value in full_config.items():
            if key != 'datasets':
                dataset_cfg[key] = value

        return dataset_cfg

    raise ValueError(f"❌ Dataset '{dataset_type}' not found in {config_path}")

def load_training_config(file_path, dataset, shape=None):
    """Loads VAE/NODE training configs (matching your main.py logic)."""
    with open(file_path, 'r') as file:
        full_config = yaml.safe_load(file)

    dataset_config = full_config.get(dataset)
    if dataset_config is None:
        raise ValueError(f"❌ Dataset '{dataset}' not found in {file_path}")

    if shape and shape in dataset_config:
        shape_cfg = dataset_config[shape]
        # Inherit the common 'dataset' settings!
        if 'dataset' in dataset_config:
            shape_cfg['dataset'] = copy.deepcopy(dataset_config['dataset'])
        return shape_cfg

    return dataset_config

# ==========================================
# VAE PREPARATION
# ==========================================
def load_respective_vae(args, device):
    # 1. Load VAE Config to find the model
    vae_cfg = load_training_config('config_files/vae_config.yaml', args.dataset, args.shape)
    original_path = vae_cfg['training_artifacts']['model_path']
    dataset_shape = args.shape + '-Shape' if args.shape not in ['None'] else args.shape
    vae_cfg['training_artifacts']['model_path'] = original_path.replace('{shape}', dataset_shape)
    vae_path = vae_cfg['training_artifacts']['model_path']
    print(f"Loading VAE from: {vae_path}")

    # 2. Instantiate and load VAE weights
    total_dof = vae_cfg['architecture']['pos_dof'] + vae_cfg['architecture']['qua_dof']
    dummy_data = torch.randn(100, total_dof)
    vae_model = load_pretrained_vae(vae_cfg, dummy_data)
    vae_model.to(device).eval()
    print(f"VAE Model initialized with DOF={total_dof} on {device.upper()}")

    ### Visualization of Manifold
    # Extract latent_frame from the config
    # space_title = ''
    # if args.dataset == 'toy':
    #     space_title = args.dataset.upper()
    # if args.dataset == 'lasa':
    #     space_title = args.dataset.upper() + ' ' +args.shape+ '-Shape'
    # if args.dataset == 'robot':
    #         space_title = args.dataset.upper() + ' Experiment'
    # l_max = vae_cfg['visualization']['latent_frame']
    # visualize_metric(vae_model, space_title, l_max)
    # plt.show()

    return vae_model

# ==========================================
# DATASET PREPARATION
# ==========================================
def prepare_benchmark_data(dataset_cfg, args, device):
    """Checks if the dataset exists; if not, generates it using the VAE."""
    gt_data_path = dataset_cfg['gt_data'].replace('{shape}', args.shape)
    directory = os.path.dirname(gt_data_path)
    filename = os.path.basename(gt_data_path)

    if os.path.exists(gt_data_path):
        print(f"✅ Found existing benchmark dataset at: {gt_data_path}")
        return

    print(f"⚠️ Benchmark dataset not found at {gt_data_path}. Generating now...")

    # 1. Load VAE
    print("Loading VAE for dataset generation...")
    vae_model = load_respective_vae(args, device)

    # 2. Generate the Data
    print("Slicing and processing benchmark segments...")
    dataset_shape = args.shape + '-Shape' if args.shape not in ['None'] else args.shape

    ground_truth = generate_benchmark_dataset(dataset_cfg, dataset_shape, vae_model, device)

    # 3. Save it
    z1 = ground_truth[:, 0, :]
    z2 = ground_truth[:, -1, :]
    save_test_config(z1, z2, ground_truth, filename=filename, directory=directory)
    print("✅ Benchmark dataset generated and secured!")

    # Ploting benchmark paths
    z1, z2, gt = load_test_config(filename=filename, directory=directory, )
    plot_trajectories(gt)

def main():
    print("\n" + "=" * 50)
    print("🚀 Riemannian LfD - Benchmarking Pipeline")
    print("=" * 50)

    # 1. Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="Evaluate frameworks on Riemannian LfD")
    parser.add_argument('--framework', type=str, required=True,
                        choices=['node', 'stochman', 'graph', 'all'],
                        help="Which framework to evaluate (or 'all')")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['toy', 'lasa' , 'robot'],
                        help="Which dataset to use")
    parser.add_argument('--shape', type=str, default='None',
                        help="Specific shape for LASA (e.g., N, Angle)")
    args = parser.parse_args()

    # 2. Load Configuration and Setup Device
    dataset_cfg = load_benchmark_config("config_files/benchmark_config.yaml", args.dataset)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 3. Trigger dataset safety check & generation
    print("\n Preparing datasets...")
    prepare_benchmark_data(dataset_cfg, args, device)
    print("\n Preparing Latent Space...")
    vae_model = load_respective_vae(args, device)

    print("\n✅ Setup complete! Ready to run framework evaluations.")

    # 4. Setup Results Directory
    dataset_name_with_shape = args.shape + '-Shape' if args.shape not in ['None'] else args.dataset
    results_dir = os.path.join(dataset_cfg['results_base_dir'], dataset_name_with_shape)
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n✅ Creating {results_dir} to save benchmark results.")

    # 5. Route to the correct evaluation script
    frameworks_to_run = ['node', 'stochman', 'graph'] if args.framework == 'all' else [args.framework]

    for fw in frameworks_to_run:
        print(f"\n--- Starting Benchmark for: {fw.upper()} ---")

        # Grab the specific framework settings from the config
        fw_cfg = dataset_cfg['frameworks'][fw]

        if fw == 'node':
            print("NODE evaluation...")
            run_node_benchmark(dataset_cfg, fw_cfg, args.dataset, args.shape, results_dir, device, vae_model)
        elif fw == 'graph':
            print("Graph evaluation...")
            run_graph_benchmark(dataset_cfg, fw_cfg, args.dataset, args.shape, results_dir, device, vae_model)
        elif fw == 'stochman':
            print("Stochman evaluation...")
            run_stochman_benchmark(dataset_cfg, fw_cfg, args.dataset, args.shape, results_dir, device, vae_model)

    print("\n" + "=" * 50)
    print("✅ All requested benchmarks completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()