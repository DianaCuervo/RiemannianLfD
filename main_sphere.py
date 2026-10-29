import argparse
import yaml
import torch
import copy
import sys
import os
import json
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchdiffeq import odeint

# --- YOUR EXISTING IMPORTS ---
from node.node_model import GoalConditionedNODE
from node.utils.logger import ConsoleLogger
from node.evaluation.node_test import load_trained_node
from node.data.dataset import prepare_loaders
from node.utils.plots import plot_sphere_trajectories_from_dataloader, plot_save_training_results

# --- NEW SPHERE IMPORTS ---
# SphereManifold class into sphere.py
from sphere_case.sphere import SphereManifold
from sphere_case.sphere_training import train_sphere_node_energy_goal_imitation_riemannianmse
from sphere_case.sphere_test import test_sphere_node


# Load config files fn
def load_and_filter_config(file_path, dataset, shape=None):
    """Loads the YAML and extracts only the relevant sub-dictionary."""
    with open(file_path, 'r') as file:
        full_config = yaml.safe_load(file)

    # 1. Drill down into the dataset (e.g., 'lasa' or 'toy')
    dataset_config = full_config.get(dataset)
    if dataset_config is None:
        raise ValueError(f"Dataset '{dataset}' not found in {file_path}")

    # If we are looking for a specific shape (like 'N' or 'Angle')
    if shape and shape in dataset_config:
        shape_cfg = dataset_config[shape]

        # THE FIX: Inherit the common 'dataset' settings!
        if 'dataset' in dataset_config:
            shape_cfg['dataset'] = copy.deepcopy(dataset_config['dataset'])

        return shape_cfg

    # 2. Drill down into the shape if applicable (for NODE config)
    #if shape and shape in dataset_config:
     #   return dataset_config[shape]


    # If no shape is needed (like for VAE), just return the dataset level
    return dataset_config

sys.stdout = ConsoleLogger()
def main():
    ### 1. CONFIGURATION
    print("\n--- Completing Configuration for Analytic Sphere ---")
    parser = argparse.ArgumentParser(description="Run RiemannianLfD on Analytic Sphere")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], help="Which mode to use")
    #parser.add_argument('--num_samples', type=int, default=1000, help="Number of synthetic SLERP demonstrations")
    #parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    #parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    args = parser.parse_args()

    dataset = 'sphere'

    # Load node_config
    node_cfg = load_and_filter_config('config_files/node_config.yaml', dataset, 'None')

    model_name = node_cfg['dataset'].get('model_name', 'NODE_AnalyticSphere')

    # Optional suffix to keep different loss setups apart (e.g. 'imitation_only'):
    # applies to the saved model, training log/curves and test results folder
    run_suffix = node_cfg.get('run_suffix')
    if run_suffix:
        model_name = f"{model_name}_{run_suffix}"

    node_cfg['dataset']['model_name'] = model_name

    # Log Initialization
    logs_dir = node_cfg['dataset'].get('model_train_logs', './node/training/training_logs')
    sys.stdout.set_log_file(save_dir=logs_dir, model_name=model_name)

    ### 2. INITIALIZATION
    print("\n--- Initializing Models ---")
    # Initialize the Analytic Manifold (Replaces the VAE)
    manifold = SphereManifold(radius=1.0)
    print("Analytic 2-Sphere Manifold initialized.")

    # Initialize the base NODE model
    # Pass the NODE config
    latent_d = node_cfg['architecture']['latent_dim']
    hidden_d = node_cfg['architecture']['hidden_dim']
    model = GoalConditionedNODE(latent_dim=latent_d, hidden_dim=hidden_d)
    print(f"NODE Model initialized with latent_dim={latent_d} and hidden_dim={hidden_d}")

    ### 3. PREPARE DATASETS
    print("\n--- Preparing Datasets (SLERP Generation) ---")
    # Generate perfect analytic ground truth paths on the sphere (Replaces build_dataset_offline)
    save_dir = manifold.build_sphere_dataset_offline(node_cfg)
    train_loader, val_loader, test_loader = prepare_loaders(
        save_dir,
        batch_size=node_cfg['training']['batch_size'],
        train_ratio=node_cfg['dataset']['train_rat'],
        val_ratio=node_cfg['dataset']['val_rat'],
    )

    # ## Visualization of Manifold with processed demonstrations
    # plot_sphere_trajectories_from_dataloader(train_loader, space_name=dataset)
    # plot_sphere_trajectories_from_dataloader(val_loader, space_name=dataset)
    # plot_sphere_trajectories_from_dataloader(test_loader, space_name=dataset)

    if args.mode == 'train':
        ### TRAIN NODE!
        # --- Load the saved baselines into the config ---
        metadata_path = os.path.join(save_dir, "dataset_baselines.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                baselines = json.load(f)
            print(f"✅ Successfully loaded dataset baselines from JSON.")
        else:
            print("⚠️ Warning: dataset_baselines.json not found! Using defaults.")
            baselines = {"safe_baseline": 1.0, "metric_scale_energy": 1e-5}

        # 4. OVERRIDE safe_baseline and metric_scale_energy values in YAML, use it. Otherwise, use JSON.
        yaml_baseline = node_cfg['training'].get('safe_baseline')
        yaml_scale = node_cfg['training'].get('metric_scale_energy')

        # --- THE BOUNCER FUNCTION ---
        def get_valid_number(val, default_val):
            if val is None:
                return default_val
            try:
                return float(val)
            except ValueError:
                return default_val  # Ignores 'safe_baseline_dummy'

        # --- THIS REPLACES YOUR OLD CRASHING LINES ---
        final_baseline = get_valid_number(yaml_baseline, baselines['safe_baseline'])
        final_scale = get_valid_number(yaml_scale, baselines['metric_scale_energy'])

        node_cfg['training']['safe_baseline'] = final_baseline
        node_cfg['training']['metric_scale_energy'] = final_scale

        # Check if yaml_baseline is actually a valid number to accurately print the source
        is_baseline_override = isinstance(yaml_baseline, (int, float)) or (
                isinstance(yaml_baseline, str) and yaml_baseline.replace('.', '', 1).replace('e', '', 1).replace('-',
                                                                                                                 '',
                                                                                                                 1).isdigit())
        is_scale_override = isinstance(yaml_scale, (int, float)) or (
                isinstance(yaml_scale, str) and yaml_scale.replace('.', '', 1).replace('e', '', 1).replace('-', '',
                                                                                                           1).isdigit())

        print(f"⚙️  Using Safe Baseline: {final_baseline:,.2f} " +
              ("(YAML Override)" if is_baseline_override else "(Auto-Calculated)"))
        print(f"⚙️  Using Energy Scale:  {final_scale:.2e} " +
              ("(YAML Override)" if is_scale_override else "(Auto-Calculated)"))

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("\n🚀 Launching NODE Training on Sphere...")

        # trained_model, history = train_sphere_node(
        #     ode_func=ode_func,
        #     manifold=manifold,
        #     train_loader=train_loader,
        #     val_loader=val_loader,
        #     epochs=args.epochs,
        #     lr=1e-3,
        #     device=device,
        #     t_steps=t_steps
        # )

        trained_model, history = train_sphere_node_energy_goal_imitation_riemannianmse(
            model=model,
            manifold=manifold,
            train_loader=train_loader,
            val_loader=val_loader,
            node_cfg=node_cfg,
            device=device
        )

        plot_save_training_results(
            history,
            save_dir=node_cfg['dataset'].get('model_train_curves_dir', './models/node/sphere'),
            filename=model_name + "_Training-Curves.svg"
        )

    elif args.mode == 'test':
        ### TEST NODE
        print("\n🚀 Launching NODE Testing on Sphere...")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 1. Figure out where the model is saved based on the config
        model_dir = node_cfg['dataset'].get('model_dir', f"./models/node/{dataset}")
        model_path = os.path.join(model_dir, f"{model_name}.pth")
        results_space = f"Analytic Sphere"

        test_results_dir = node_cfg['dataset'].get('model_test_results_dir', f"./results/node/{dataset}")
        results_path = os.path.join(test_results_dir, f"{model_name}")

        # 2. Load the trained NODE
        trained_model = load_trained_node(
            model_path=model_path,
            latent_dim=latent_d,
            hidden_dim=hidden_d,
            device=device
        )

        # 3. Test function
        test_sphere_node(
            trained_model=trained_model,
            manifold=manifold,
            test_loader=test_loader,
            save_dir=results_path,
            device=device,
            space_name=results_space.capitalize(),
        )

        # test_loss_total = 0.0
        # with torch.no_grad():
        #     for batch in test_loader:
        #         gt_traj = batch[0].to(device)
        #         start_points = gt_traj[:, 0, :]
        #         pred_traj = odeint(ode_func, start_points, t_steps, method='rk4').permute(1, 0, 2)
        #         pred_traj = manifold.project_manifold(pred_traj)
        #
        #         loss, _ = manifold.compute_composite_loss(
        #             pred_traj, gt_traj, t_steps, w_imit=1.0, w_energy=0.01, w_goal=1.0
        #         )
        #         test_loss_total += loss.item()
        #
        # print(f"✅ Test completed. Average Test Loss: {test_loss_total / len(test_loader):.6f}")

if __name__ == "__main__":
    main()