import os
import time
import json
import torch
import numpy as np

from benchmark.data_utils import load_test_config, save_latent_paths_node
from benchmark.metrics import (
    calculate_ground_metrics_chuncked,
    calculate_euclidean_metrics,
    decode_latente_paths,
    decode_latent_goals
)
from node.evaluation.node_test import load_trained_node

def run_node_benchmark(dataset_cfg, fw_cfg, dataset_type, shape_name, results_dir, device, vae_model):
    """
    Evaluates the NODE framework against the benchmark test set.
    """
    title_complement = '' if shape_name in ['None'] else '('+shape_name+')'
    print("\n" + "-" * 40)
    print(f"🧠 NODE Benchmark Evaluation - {dataset_type.upper()} {title_complement}")
    print("-" * 40)

    # ==========================================
    # 1. LOAD DATA
    # ==========================================
    """Checks if the dataset exists; if not, generates it using the VAE."""
    gt_data_path = dataset_cfg['gt_data'].replace('{shape}', shape_name)
    directory = os.path.dirname(gt_data_path)
    filename = os.path.basename(gt_data_path)
    z1, z2, ground_truth = load_test_config(filename=filename, directory=directory)

    z1 = z1.to(device)
    z2 = z2.to(device)
    ground_truth = ground_truth.to(device)

    # Recreate the time steps array used during training/generation
    t_resolution = ground_truth.shape[1]
    t_steps = torch.linspace(0, 1, t_resolution).to(device)
    print(f"Time Resolution: {t_resolution}")

    # ==========================================
    # 2. LOAD NODE
    # ==========================================
    print("Loading NODE...")
    dataset_fw_cfg = fw_cfg[dataset_type]
    dataset_shape = shape_name+'-Shape'
    model_name_template = dataset_fw_cfg.get('model_name', 'NODE_{shape}RiemannianMSE_EGI')
    full_name = f"{dataset_type}_"
    if dataset_type == 'lasa':
        #shape_name = node_cfg['dataset'].get('shape_name', 'UnknownShape')
        full_name = f"{dataset_type}_{dataset_shape}_"
    model_name = model_name_template.replace('{shape}', full_name)

    if dataset_type == 'lasa':
        latent_d = dataset_fw_cfg[shape_name]['latent_dim']
        hidden_d = dataset_fw_cfg[shape_name]['hidden_dim']
    else:
        latent_d = dataset_fw_cfg['latent_dim']
        hidden_d = dataset_fw_cfg['hidden_dim']

    # 1. Figure out where the model is saved based on the config
    model_dir = dataset_fw_cfg.get('model_dir', f"./models/node/{dataset_type}")
    model_path = os.path.join(model_dir, f"{model_name}.pth")

    # 2. Load the trained NODE
    my_model = load_trained_node(
        model_path=model_path,
        latent_dim=latent_d,
        hidden_dim=hidden_d,
        device=device
    )

    print(f"NODE Model initialized with latent_dim={latent_d} and hidden_dim={hidden_d}")

    # ==========================================
    # 3. INFERENCE
    # ==========================================
    print("\nRunning Inference...")
    start_time = time.time()

    with torch.no_grad():
        pred_traj = my_model(z1, z2, t_steps)
        z_pred = pred_traj[:, :, 0, :]
        v_pred = pred_traj[:, :, 1, :]

    end_time = time.time()

    proxy_geos = z_pred.permute(1, 0, 2)
    ground_data = ground_truth
    print(f"Ground Shape: {ground_truth.shape}")
    print(f"Proxy Shape: {proxy_geos.shape}")

    inference_time = end_time - start_time
    print(f"⏱️ Inference Time ({proxy_geos.shape[0]} paths): {inference_time:.4f} seconds.")

    # ==========================================
    # 4. CALCULATE METRICS
    # ==========================================
    print("\n📊 Calculating Latent Space Metrics...")
    riemannian_metrics = calculate_ground_metrics_chuncked(
        proxy_geos, ground_data, z2, vae_model, density_threshold=1.5, chunk_size=50
    )

    print("\n📏 Decoding to Euclidean Space...")
    euclidean_paths_g = decode_latente_paths(vae_model, ground_data)
    euclidean_paths_p = decode_latente_paths(vae_model, proxy_geos)
    euclidean_goals_g = decode_latent_goals(vae_model, z2)
    predicted_goals = euclidean_paths_p[:, -1, :]

    euclidean_metrics = calculate_euclidean_metrics(
        euclidean_paths_p, euclidean_paths_g, predicted_goals, euclidean_goals_g
    )

    # ==========================================
    # 5. SAVE RESULTS
    # ==========================================
    metrics_report = {
        "framework": "NODE",
        "dataset": dataset_type,
        "shape": shape_name,
        "inference_time_seconds": inference_time,
        "riemannian_metrics": riemannian_metrics,
        "euclidean_metrics": euclidean_metrics
    }

    # Numpy-to-JSON Translator
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.generic):
                return obj.item()  # Converts numpy float/int to standard Python float/int
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    json_path = os.path.join(results_dir, "node_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(metrics_report, f, indent=4, cls=NumpyEncoder)

    print(f"\n💾 Metrics saved to: {json_path}")

    save_latent_paths_node(proxy_geos, dir_name=results_dir, file_name="node_proxy-geodesics.pt")