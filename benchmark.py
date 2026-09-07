import argparse
import os
import yaml
import torch

from benchmark.data_utils import generate_benchmark_dataset, save_test_config
from vae.vae_model import VAE

def load_benchmark_config(config_path="config_files/benchmark_config.yaml"):
    """Loads the benchmark YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Cannot find benchmark config at {config_path}")
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_and_filter_config(config_path, dataset_type):
    """Helper to load your YAML files (just like in main.py)"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    # If your config has a top level 'datasets' key, return the specific one
    return config['datasets'][dataset_type] if 'datasets' in config else config


def prepare_benchmark_data(dataset_cfg, args, device):
    """Checks if the dataset exists; if not, generates it using the VAE."""
    gt_data_path = dataset_cfg['gt_data'].replace('{shape}', args.shape)
    directory = os.path.dirname(gt_data_path)
    filename = os.path.basename(gt_data_path)

    if os.path.exists(gt_data_path):
        print(f"✅ Found existing benchmark dataset at: {gt_data_path}")
        return

    print(f"⚠️ Benchmark dataset not found at {gt_data_path}. Generating now...")

    # 1. Load VAE Config to find the model
    print("Loading VAE for dataset generation...")
    vae_cfg = load_and_filter_config('config_files/vae_config.yaml', args.dataset)
    original_path = vae_cfg['training_artifacts']['model_path']
    dataset_shape = args.shape + '-Shape' if args.shape not in ['Angle', 'None'] else args.shape
    vae_path = original_path.replace('{shape}', dataset_shape)
    print(f"Loading VAE from: {vae_path}")

    # 2. Instantiate and load VAE weights
    vae_model = VAE(
        layers=vae_cfg['model_params']['layers'],
        sigma_z=float(vae_cfg['model_params']['sigma'])
    ).to(device)

    checkpoint = torch.load(vae_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    vae_model.load_state_dict(state_dict)
    vae_model.eval()

    # 3. Generate the Data using our new DRY function!
    print("Slicing and processing benchmark segments...")
    ground_truth = generate_benchmark_dataset(dataset_cfg, args.shape, vae_model, device)

    # 4. Save it
    z1 = ground_truth[:, 0, :]
    z2 = ground_truth[:, -1, :]
    save_test_config(z1, z2, ground_truth, filename=filename, directory=directory)

    print("✅ Benchmark dataset generated and secured!")

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

    # 2. Load Configuration
    config = load_and_filter_config("config_files/benchmark_config.yaml", args.dataset)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- NEW: DATASET GENERATION CHECK ---
    # Trigger dataset safety check & generation
    prepare_benchmark_data(config, args, device)
    # -------------------------------------

    config = load_and_filter_config("config_files/benchmark_config.yaml")
    dataset_cfg = config['datasets'][args.dataset]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Run the generator/checker
    prepare_benchmark_data(dataset_cfg, args, device)

    # --- Framework evaluation triggers will go here next ---
    print("\n✅ Setup complete! Ready to run framework evaluations.")

    # Load vae_config
    vae_cfg = load_and_filter_config('config_files/vae_config.yaml', args.dataset)
    original_path = vae_cfg['training_artifacts']['model_path']
    # Replace the '{shape}' placeholder with the actual shape from the command line
    dataset_shape = args.shape+'-Shape'
    #if args.shape != 'Angle' and args.shape != 'None':
    #    dataset_shape = dataset_shape+'-Shape'

    vae_cfg['training_artifacts']['model_path'] = original_path.replace('{shape}', dataset_shape)
    print(f"Loading VAE from: {vae_cfg['training_artifacts']['model_path']}")

    # # Ensure the requested dataset exists in the config
    # if args.dataset not in config['datasets']:
    #     raise ValueError(f"❌ Dataset '{args.dataset}' not found in benchmark_config.yaml")
    #
    # dataset_cfg = config['datasets'][args.dataset]
    #
    # # 3. Setup Directories
    # # E.g., ./benchmarks/results/exp6/
    # results_dir = os.path.join(config.get('results_base_dir', './benchmarks/results'), args.dataset)
    # os.makedirs(results_dir, exist_ok=True)
    #
    # # 4. Device Selection
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(f"⚙️  Hardware: {device.upper()}")
    # print(f"📊 Dataset:  {args.dataset}")
    # print(f"📂 Results will be saved to: {results_dir}\n")
    #
    # # 5. The Dispatcher (Routing to the correct evaluation script)
    # frameworks_to_run = ['rnode', 'discrete', 'iterative'] if args.framework == 'all' else [args.framework]
    #
    # for fw in frameworks_to_run:
    #     print(f"\n--- Starting Benchmark for: {fw.upper()} ---")
    #     fw_cfg = config['frameworks'].get(fw, {})
    #
    #     if fw == 'rnode':
    #         run_rnode_benchmark(dataset_cfg, fw_cfg, results_dir, device)
    #     elif fw == 'discrete':
    #         run_discrete_benchmark(dataset_cfg, fw_cfg, results_dir, device)
    #     elif fw == 'iterative':
    #         run_iterative_benchmark(dataset_cfg, fw_cfg, results_dir, device)
    #
    # print("\n" + "=" * 50)
    # print("✅ All requested benchmarks completed successfully!")
    # print("=" * 50)


if __name__ == "__main__":
    main()