import argparse
import os
import copy
import yaml
import json
import optuna
import torch
import time

# --- 1. Import your existing project modules ---
# USING YOUR EXACT IMPORTS FROM MAIN.PY
from node.node_model import GoalConditionedNODE
from node.data.preprocessing import build_dataset_offline
from node.data.dataset import prepare_loaders
from vae.vae_model import load_pretrained_vae
from node.training.node_train import train_node_energy_goal_imitation_riemannianmse_optuna  # Ensure trial=None is in this fn def


def load_and_filter_base_config(file_path, dataset, shape=None):
    """Loads the YAML and extracts only the relevant sub-dictionary."""
    with open(file_path, 'r') as file:
        full_config = yaml.safe_load(file)

    dataset_config = full_config.get(dataset)
    if dataset_config is None:
        raise ValueError(f"Dataset '{dataset}' not found in {file_path}")

    if shape and shape in dataset_config:
        shape_cfg = dataset_config[shape]
        if 'dataset' in dataset_config:
            shape_cfg['dataset'] = copy.deepcopy(dataset_config['dataset'])
        return shape_cfg

    return dataset_config


# --- 2. Define the Objective Function ---
# We pass the initialized VAE into the objective so we don't reload it every trial
def objective(trial, base_node_cfg, vae, device):
    # A. Create a pristine copy of the config for this specific trial
    node_cfg = copy.deepcopy(base_node_cfg)

    # B. Define the Search Space (Overwrite static values with Optuna suggestions)
    node_cfg['training']['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    node_cfg['training']['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64])
    node_cfg['training']['scheduler']['w_goal_target'] = trial.suggest_float('w_goal_target', 10.0, 100.0)
    node_cfg['training']['scheduler']['w_ener_target'] = trial.suggest_float('w_ener_target', 0.1, 10.0, log=True)
    node_cfg['architecture']['hidden_dim'] = trial.suggest_categorical('hidden_dim', [128, 256, 512])

    # C. Initialize NODE with the (potentially altered) hidden_dim
    latent_d = node_cfg['architecture']['latent_dim']
    hidden_d = node_cfg['architecture']['hidden_dim']
    model = GoalConditionedNODE(latent_dim=latent_d, hidden_dim=hidden_d).to(device)

    # D. Prepare Datasets
    # We call this inside the objective because batch_size might have changed
    save_dir = build_dataset_offline(node_cfg, vae, device='cpu')
    train_loader, val_loader, test_loader = prepare_loaders(
        save_dir,
        batch_size=node_cfg['training']['batch_size'],
        train_ratio=node_cfg['dataset']['train_rat'],
        val_ratio=node_cfg['dataset']['val_rat'],
    )

    # E. Baseline Extraction (Copied from your main.py)
    metadata_path = os.path.join(save_dir, "dataset_baselines.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            baselines = json.load(f)
    else:
        baselines = {"safe_baseline": 3867617.82, "metric_scale_energy": 1e-8}

    def get_valid_number(val, default_val):
        if val is None: return default_val
        try:
            return float(val)
        except ValueError:
            return default_val

    yaml_baseline = node_cfg['training'].get('safe_baseline')
    yaml_scale = node_cfg['training'].get('metric_scale_energy')
    node_cfg['training']['safe_baseline'] = get_valid_number(yaml_baseline, baselines['safe_baseline'])
    node_cfg['training']['metric_scale_energy'] = get_valid_number(yaml_scale, baselines['metric_scale_energy'])

    # F. Run Training Loop
    # NOTE: Ensure your train_node_energy_goal_imitation_riemannianmse function
    # accepts a 'trial' keyword argument!
    trained_model, history = train_node_energy_goal_imitation_riemannianmse_optuna(
        model=model,
        vae=vae,
        train_loader=train_loader,
        val_loader=val_loader,
        node_cfg=node_cfg,
        device=device,
        trial=trial  # Passed for pruning
    )

    # G. Return Validation Loss to Minimize
    final_val_loss = history["val_loss"][-1]
    return final_val_loss


# --- 3. Main Execution Block ---
if __name__ == "__main__":
    print("\n--- Completing Configuration ---")
    parser = argparse.ArgumentParser(description="Optuna Tuning for Neural ODE")
    parser.add_argument('--n_trials', type=int, default=50, help="How many Optuna trials to run")
    parser.add_argument('--dataset', type=str, required=True, choices=['toy', 'lasa', 'robot'])
    parser.add_argument('--shape', type=str, default='None', help="Specific shape for LASA")
    args = parser.parse_args()
    print(
        f"Starting NODE optimization with {args.n_trials} trials for Dataset: {args.dataset.upper()} | Shape: {args.shape}")

    # 1. Load VAE Config and Initialize VAE (Only needs to happen once!)
    vae_cfg = load_and_filter_base_config('config_files/vae_config.yaml', args.dataset)
    dataset_shape = args.shape + '-Shape'
    vae_cfg['training_artifacts']['model_path'] = vae_cfg['training_artifacts']['model_path'].replace('{shape}',
                                                                                                      dataset_shape)
    # Pass the VAE config
    total_dof = vae_cfg['architecture']['pos_dof'] + vae_cfg['architecture']['qua_dof']
    dummy_data = torch.randn(100, total_dof)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize the VAE once and pass it to the objective
    vae = load_pretrained_vae(vae_cfg, dummy_data).to(device)
    vae.eval()  # Keep it in eval mode

    # 2. Load Base NODE Config
    node_cfg_path = "config_files/node_config.yaml"
    base_node_cfg = load_and_filter_base_config(node_cfg_path, args.dataset, args.shape)

    # Apply dynamic naming overrides (copied from main.py)
    if args.dataset == 'lasa':
        base_node_cfg['dataset']['shape_name'] = base_node_cfg['dataset']['shape_name'].replace('{shape}',
                                                                                                dataset_shape)
        base_node_cfg['dataset']['origin_file'] = base_node_cfg['dataset']['origin_file'].replace('{shape}',
                                                                                                  dataset_shape)
        base_node_cfg['dataset']['save_dir'] = base_node_cfg['dataset']['save_dir'].replace('{shape}', dataset_shape)
    dataset_type = base_node_cfg['dataset'].get('type', 'UnknownDataset')
    model_name_template = base_node_cfg['dataset'].get('model_name', 'NODE_{shape}RiemannianMSE_EGI')
    full_name = f"{dataset_type}_{dataset_shape}_" if dataset_type == 'lasa' else f"{dataset_type}_"

    # Append _Optuna so tuning runs don't overwrite your main model weights
    base_node_cfg['dataset']['model_name'] = model_name_template.replace('{shape}', full_name) + "_Optuna"

    # 3. Setup Optuna
    storage_name = f"sqlite:///node_tuning_{full_name}DB.db"
    study_name = f"NODE_{full_name}Study"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50)
    )

    print(f"Resuming optimization. Currently completed {len(study.trials)} trials.")

    # Start the timer
    start_time = time.time()

    # 4. Run Optimization
    # Notice we pass the initialized VAE and device into the objective!
    study.optimize(lambda trial: objective(trial, base_node_cfg, vae, device), n_trials=args.n_trials)

    # Stop the timer and calculate duration
    end_time = time.time()
    total_duration_seconds = end_time - start_time
    total_duration_minutes = total_duration_seconds / 60.0
    total_duration_hours = total_duration_minutes / 60.0

    # --- 5. Print Results ---
    print("\n" + "=" * 50)
    print("Optimization Finished!")
    # 5. Print the formatted total time
    if total_duration_hours > 1:
        print(f"Total Optimization Time: {total_duration_hours:.2f} hours")
    elif total_duration_minutes > 1:
        print(f"Total Optimization Time: {total_duration_minutes:.2f} minutes")
    else:
        print(f"Total Optimization Time: {total_duration_seconds:.2f} seconds")

    print(f"Best Trial Score (Val Loss): {study.best_value:.5f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("=" * 50)