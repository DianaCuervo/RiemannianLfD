import argparse
import yaml
import torch
import copy
import sys
import os
import json

# --- NEW ABSOLUTE IMPORTS ---
from node.node_model import GoalConditionedNODE
from node.data.preprocessing import build_dataset_offline
from node.data.dataset import prepare_loaders
from vae.vae_model import VAE, load_pretrained_vae
from node.utils.plots import visualize_metric, plot_latent_dataloader, plot_trajectories_on_manifold, \
    plot_save_training_results
from node.training.node_train import train_node_energy_goal_imitation_riemannianmse
from node.utils.logger import ConsoleLogger  # (or wherever you saved it)
from node.evaluation.node_test import load_trained_node, test_node

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
    ### LOAD CONFIG FILES
    print("\n--- Completing Configuration ---")
    ## Load of the configuration file according to the specific dataset/experiment
    # Catch the Command Line Arguments
    parser = argparse.ArgumentParser(description="Run RiemannianLfD")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], help="Which mode to use")
    parser.add_argument('--dataset', type=str, required=True, choices=['toy', 'lasa' , 'robot'], help="Which dataset to use")
    parser.add_argument('--shape', type=str, default='None', help="Specific shape for LASA (e.g., N, Angle)")
    args = parser.parse_args()
    print(f"Starting run for Dataset: {args.dataset.upper()} | Shape: {args.shape}")

    # Load vae_config
    vae_cfg = load_and_filter_config('config_files/vae_config.yaml', args.dataset)
    original_path = vae_cfg['training_artifacts']['model_path']
    # Replace the '{shape}' placeholder with the actual shape from the command line
    dataset_shape = args.shape+'-Shape'
    #if args.shape != 'Angle' and args.shape != 'None':
    #    dataset_shape = dataset_shape+'-Shape'

    vae_cfg['training_artifacts']['model_path'] = original_path.replace('{shape}', dataset_shape)
    print(f"Loading VAE from: {vae_cfg['training_artifacts']['model_path']}")

    # Load node_config
    node_cfg = load_and_filter_config('config_files/node_config.yaml', args.dataset, args.shape)
    if args.dataset == 'lasa':
        node_cfg['dataset']['shape_name'] = node_cfg['dataset']['shape_name'].replace('{shape}', dataset_shape)
        node_cfg['dataset']['origin_file'] = node_cfg['dataset']['origin_file'].replace('{shape}', dataset_shape)
        node_cfg['dataset']['save_dir'] = node_cfg['dataset']['save_dir'].replace('{shape}', dataset_shape)
    dataset_type = node_cfg['dataset'].get('type', 'UnknownDataset')
    model_name_template = node_cfg['dataset'].get('model_name', 'NODE_{shape}RiemannianMSE_EGI')
    full_name = f"{dataset_type}_"
    if dataset_type == 'lasa':
        #shape_name = node_cfg['dataset'].get('shape_name', 'UnknownShape')
        full_name = f"{dataset_type}_{dataset_shape}_"
    model_name = model_name_template.replace('{shape}', full_name)
    node_cfg['dataset']['model_name'] = model_name

    # Log Initialization
    logs_dir = node_cfg['dataset'].get('model_train_logs', './node/training/training_logs')
    sys.stdout.set_log_file(save_dir=logs_dir, model_name=model_name)

    ### INITIALIZATION
    print("\n--- Initializing Models ---")
    # Pass the VAE config
    total_dof = vae_cfg['architecture']['pos_dof'] + vae_cfg['architecture']['qua_dof']
    dummy_data = torch.randn(100, total_dof)
    vae = load_pretrained_vae(vae_cfg, dummy_data)
    print(f"VAE Model initialized with DOF={total_dof}")

    ### Visualization of Manifold
    # # Extract latent_frame from the config
    # space_title = ''
    # if args.dataset == 'toy':
    #     space_title = args.dataset.upper()
    # if args.dataset == 'lasa':
    #     space_title = args.dataset.upper() + ' ' +args.shape+ '-Shape'
    # if args.dataset == 'robot':
    #         space_title = args.dataset.upper() + ' Experiment'
    # l_max = vae_cfg['visualization']['latent_frame']
    # visualize_metric(vae, space_title, l_max)

    # Pass the NODE config
    latent_d = node_cfg['architecture']['latent_dim']
    hidden_d = node_cfg['architecture']['hidden_dim']
    model = GoalConditionedNODE(latent_dim=latent_d, hidden_dim=hidden_d)
    print(f"NODE Model initialized with latent_dim={latent_d} and hidden_dim={hidden_d}")

    ### PREPARE DATASETS
    print("\n--- Preparing Datasets ---")
    save_dir = build_dataset_offline(node_cfg, vae, device='cpu')
    train_loader, val_loader, test_loader = prepare_loaders(
        save_dir,
        batch_size=node_cfg['training']['batch_size'],
        train_ratio=node_cfg['dataset']['train_rat'],
        val_ratio=node_cfg['dataset']['val_rat'],
    )
    ### Visualization of Manifold with processed demonstrations
    # l_max = vae_cfg['visualization']['latent_frame']
    # plot_trajectories_on_manifold(vae, train_loader, space_name=args.shape, latent_max=l_max)
    # plot_trajectories_on_manifold(vae, val_loader, space_name=args.shape, latent_max=l_max)
    # plot_trajectories_on_manifold(vae, test_loader, space_name=args.shape, latent_max=l_max)

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
            baselines = {"safe_baseline": 3867617.82, "metric_scale_energy": 1e-8}

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

        print("\n🚀 Launching NODE Training...")

        trained_model, history = train_node_energy_goal_imitation_riemannianmse(
            model=model,
            vae=vae,
            train_loader=train_loader,
            val_loader=val_loader,
            node_cfg=node_cfg,
            device=device
        )

        plot_save_training_results(
            history,
            save_dir=node_cfg['dataset'].get('model_train_curves_dir', './models/node/lasa'),
            filename=model_name + "_Training-Curves.svg"
        )
    elif args.mode == 'test':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 1. Figure out where the model is saved based on the config
        model_dir = node_cfg['dataset'].get('model_dir', f"./models/node/{args.dataset}")
        model_path = os.path.join(model_dir, f"{model_name}.pth")
        results_space = f"{dataset_type}"
        if dataset_type == "lasa":
            results_space = f"{dataset_shape}"


        l_max = vae_cfg['visualization']['latent_frame']
        test_results_dir = node_cfg['dataset'].get('model_test_results_dir', f"./results/node/{args.dataset}")
        results_path = os.path.join(test_results_dir, f"{model_name}")

        # 2. Load the trained NODE
        trained_model = load_trained_node(
            model_path=model_path,
            latent_dim=latent_d,
            hidden_dim=hidden_d,
            device=device
        )

        # 3. Test function
        test_node(
            trained_model=trained_model,
            vae_model=vae,
            test_loader=test_loader,
            save_dir=results_path,
            device=device,
            space_name=results_space.capitalize(),
            latent_frame=l_max
        )
    elif args.mode == 'benchmark':
        print("\nTo Be Completed!")

if __name__ == "__main__":
    print("Hello world!")
    main()
    print("Bye bye!")
