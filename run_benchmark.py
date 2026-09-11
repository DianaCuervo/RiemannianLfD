import argparse
import os
import yaml
import torch
import copy

from benchmark.data_utils import generate_benchmark_dataset, save_test_config
from vae.vae_model import VAE


# ==========================================
# CONFIGURATION LOADERS
# ==========================================

def load_benchmark_config(config_path, dataset_type):
    """Loads specifically the benchmark YAML structure."""
    with open(config_path, 'r') as file:
        full_config = yaml.safe_load(file)

    if 'datasets' in full_config and dataset_type in full_config['datasets']:
        return full_config['datasets'][dataset_type]

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

    # 1. Load VAE Config to find the model
    print("Loading VAE for dataset generation...")
    vae_cfg = load_training_config('config_files/vae_config.yaml', args.dataset, args.shape)
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

    # 3. Generate the Data
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

    # 2. Load Configuration and Setup Device
    dataset_cfg = load_benchmark_config("config_files/benchmark_config.yaml", args.dataset)
    #dataset_cfg = config['datasets'][args.dataset]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 3. Trigger dataset safety check & generation
    prepare_benchmark_data(dataset_cfg, args, device)

    print("\n✅ Setup complete! Ready to run framework evaluations.")
    # (Evaluation script triggers will be uncommented here in the next step)

if __name__ == "__main__":
    main()