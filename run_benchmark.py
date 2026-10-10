import argparse
import os
import yaml
import torch
import copy

from benchmark.data_utils import generate_benchmark_original_dataset, generate_benchmark_noisy_dataset, save_test_config, load_benchmark_reference, save_json
from benchmark.metrics import decode_latente_paths, calculate_ambient_metrics
from benchmark.off_manifold import get_off_manifold_reference, reference_summary, off_manifold_rates
from benchmark.eval_graph import run_graph_benchmark
from benchmark.eval_node import run_node_benchmark
from benchmark.eval_stochman import run_stochman_benchmark
from node.data.dataset import prepare_loaders
from node.utils.plots import visualize_metric, plot_trajectories_on_manifold, plot_trajectories
from vae.vae_model import load_pretrained_vae
from vae.data.vae_dataset import load_vae_training_points
from vae.utils.config import load_dataset_config, beta_label
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
def resolve_beta(args):
    """The beta_scale of the VAE to benchmark: selected in vae_config.yaml for toy,
    from --beta_scale for lasa, None for datasets with a single VAE."""
    if args.dataset == 'toy':
        vae_cfg = load_training_config('config_files/vae_config.yaml', args.dataset)
        return beta_label(vae_cfg['architecture']['beta_scale'])
    if args.dataset == 'lasa':
        return args.beta_scale
    return None

def load_respective_vae(args, device):
    # 1. Load VAE Config to find the model
    vae_cfg = load_training_config('config_files/vae_config.yaml', args.dataset, args.shape)
    original_path = vae_cfg['training_artifacts']['model_path']
    dataset_shape = args.shape + '-Shape' if args.shape not in ['None'] else args.shape
    vae_cfg['training_artifacts']['model_path'] = original_path.replace('{shape}', dataset_shape)
    vae_cfg['training_artifacts']['cluster_path'] = vae_cfg['training_artifacts']['cluster_path'].replace('{shape}', dataset_shape)
    # One VAE per beta_scale (as main_new.py's main_iterative_vae)
    beta = resolve_beta(args)
    if beta is not None:
        vae_cfg['architecture']['beta_scale'] = float(beta)
        vae_cfg['training_artifacts']['model_path'] = vae_cfg['training_artifacts']['model_path'].replace('{beta}', str(beta))
        vae_cfg['training_artifacts']['cluster_path'] = vae_cfg['training_artifacts']['cluster_path'].replace('{beta}', str(beta))
    vae_path = vae_cfg['training_artifacts']['model_path']
    print(f"Loading VAE from: {vae_path}")

    # 2. Instantiate and load VAE weights
    total_dof = vae_cfg['architecture']['pos_dof'] + vae_cfg['architecture']['qua_dof']
    # Real training points, not noise: used to recompute beta if the checkpoint has no 'rbf_beta'
    training_points = load_vae_training_points(load_dataset_config(args.dataset, args.shape))
    vae_model = load_pretrained_vae(vae_cfg, training_points)
    vae_model.to(device).eval()
    print(f"VAE Model initialized with DOF={total_dof} on {device.upper()}")

    # ## Visualization of Manifold
    # #Extract latent_frame from the config
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
def prepare_benchmark_data(dataset_cfg, args, device, vae_model):
    """Checks if the dataset exists; if not, generates it using the VAE."""
    gt_data_path = dataset_cfg['gt_data']
    directory = os.path.dirname(gt_data_path)
    filename = os.path.basename(gt_data_path)

    if os.path.exists(gt_data_path):
        print(f"✅ Found existing benchmark dataset at: {gt_data_path}")
        return

    print(f"⚠️ Benchmark dataset not found at {gt_data_path}. Generating now...")

    # 1. Generate the Data (latent ground truth + raw demonstration segments)
    print("Slicing and processing benchmark segments...")
    dataset_shape = args.shape + '-Shape' if args.shape not in ['None'] else args.shape

    if args.benchmark_mode == 'original':
        reference = generate_benchmark_original_dataset(dataset_cfg, dataset_shape, vae_model, device)
    if args.benchmark_mode == 'noisy':
        reference = generate_benchmark_noisy_dataset(dataset_cfg, dataset_shape, vae_model, device)
    ground_truth = reference['latent']

    #2. Save it
    z1 = ground_truth[:, 0, :]
    z2 = ground_truth[:, -1, :]
    save_test_config(z1, z2, ground_truth, filename=filename, directory=directory,
                    extras={k: reference[k] for k in ('x_ambient', 'xy_center', 'xy_scale')})
    print("✅ Benchmark dataset generated and secured!")

    # Ploting benchmark paths
    plot_trajectories(ground_truth)

def save_reference_results(bench, vae_model, results_dir):
    """Method-independent rows: the VAE reconstruction floor and the off-manifold reference."""
    print("\n📏 VAE reconstruction error (the floor no method can beat)...")
    reconstruction = calculate_ambient_metrics(
        decode_latente_paths(vae_model, bench['ground_truth']), bench['x_ambient'],
        bench['xy_center'], bench['xy_scale'], vae_model.pos_dof
    )
    save_json({"shape": bench['shape'], "beta_scale": bench['beta'], "ambient_metrics": reconstruction},
              os.path.join(results_dir, "vae_reconstruction_metrics.json"))

    print("\n🧭 Off-manifold % of the latent ground truth (sanity row, at its T points)...")
    save_json({"shape": bench['shape'], "beta_scale": bench['beta'],
               **reference_summary(bench['off_ref']),
               "ground_truth_rates": off_manifold_rates(vae_model, bench['ground_truth'], bench['off_ref'])},
              os.path.join(results_dir, "off_manifold_reference.json"))

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
    parser.add_argument('--beta_scale', type=str, default='1',
                        help="Specific beta scale for the LASA VAE (e.g., 1, 5, 10); "
                             "the toy one is selected in vae_config.yaml")
    parser.add_argument('--node_variants', type=str, nargs='+', default=None,
                        help="NODE run_suffix values to evaluate (e.g., energy_imitation energy_only); "
                             "default: the run_suffix in node_config.yaml")
    parser.add_argument('--benchmark_mode', type=str, default='original',
                        choices=['original', 'noisy'],
                        help="Type of benchmark to be executed (e.g., original noise); "
                             "default: original")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                        help="Device for inference and timing")
    args = parser.parse_args()

    # 2. Load Configuration and Setup Device
    dataset_cfg = load_benchmark_config("config_files/benchmark_config.yaml", args.dataset)
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Resolve {shape}/{beta} once: every evaluator reads these paths as they are
    beta = resolve_beta(args)
    for key in ('benchmark_dir', 'gt_data'):
        dataset_cfg[key] = dataset_cfg[key].replace('{shape}', args.shape).replace('{beta}', str(beta))
        if key=='gt_data':
            dataset_cfg[key] = dataset_cfg[key].replace('{bm_mode}', args.benchmark_mode)

    # 3. Load the VAE, then trigger dataset safety check & generation
    print("\n Preparing Latent Space...")
    vae_model = load_respective_vae(args, device)
    print("\n Preparing datasets...")
    prepare_benchmark_data(dataset_cfg, args, device, vae_model)

    # 4. Setup Results Directory
    dataset_name_with_shape = args.shape + '-Shape' if args.shape not in ['None'] else args.dataset
    if beta is not None:
        dataset_name_with_shape = f"{dataset_name_with_shape}_beta_scale_{beta}"
    results_dir = os.path.join(dataset_cfg['results_base_dir'], dataset_name_with_shape)
    if args.benchmark_mode is not None:
        complement = f"{args.benchmark_mode}_ground_truth"
    results_dir = os.path.join(results_dir, complement)
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n✅ Creating {results_dir} to save benchmark results.")

    # 5. Shared inputs of every framework: reference data, off-manifold reference, NODE config
    reference = load_benchmark_reference(dataset_cfg['gt_data'])
    off_ref = get_off_manifold_reference(vae_model, args.dataset, args.shape,
                                         os.path.join(dataset_cfg['benchmark_dir'], complement+'_off_manifold_reference.pt'))
    title = args.dataset.upper() if args.shape in ['None'] else f"{args.shape}-Shape"
    if beta is not None:
        title = f"{title} beta_scale_{beta}"
    bench = {
        'shape': args.shape,
        'beta': beta,
        'title': title,
        'z1': reference['z1'].to(device),
        'z2': reference['z2'].to(device),
        'ground_truth': reference['ground_truth'].to(device),
        'x_ambient': reference['x_ambient'],
        'xy_center': reference['xy_center'],
        'xy_scale': reference['xy_scale'],
        'off_ref': off_ref,
        'node_cfg': load_training_config('config_files/node_config.yaml', args.dataset, args.shape),
    }
    save_reference_results(bench, vae_model, results_dir)

    print("\n✅ Setup complete! Ready to run framework evaluations.")

    # 6. Route to the correct evaluation script
    frameworks_to_run = ['node', 'stochman', 'graph'] if args.framework == 'all' else [args.framework]

    for fw in frameworks_to_run:
        print(f"\n--- Starting Benchmark for: {fw.upper()} ---")

        # Grab the specific framework settings from the config
        fw_cfg = dataset_cfg['frameworks'][fw]

        if fw == 'node':
            node_variants = args.node_variants or [bench['node_cfg'].get('run_suffix')]
            for variant in node_variants:
                print(f"NODE evaluation ({variant})...")
                run_node_benchmark(dataset_cfg, fw_cfg, bench, results_dir, device, vae_model, run_suffix=variant)
        elif fw == 'graph':
            print("Graph evaluation...")
            run_graph_benchmark(dataset_cfg, fw_cfg, bench, results_dir, device, vae_model)
        elif fw == 'stochman':
            print("Stochman evaluation...")
            run_stochman_benchmark(dataset_cfg, fw_cfg, bench, results_dir, device, vae_model)

    print("\n" + "=" * 50)
    print("✅ All requested benchmarks completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()
