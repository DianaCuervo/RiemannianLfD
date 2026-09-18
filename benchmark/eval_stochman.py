import os
import time
import json
import torch
import numpy as np

from stochman.manifold import Manifold
from stochman.curves import CubicSpline
from benchmark.data_utils import load_test_config, save_latent_paths_stochman
from benchmark.metrics import calculate_ground_metrics_chuncked, decode_latente_paths, decode_latent_goals, \
    calculate_euclidean_metrics
from node.utils.plots import visualize_gtvspred_comparison_multiple, save_test_plot


def stochman_benchmark(model , z1_test, z2_test):
    """Computing geodesic between two points on the manifold
            Input:
                model:              an instance of the class VAE()
                z1_test:            set of starting points
                z2_test:            set of ending points
            output:
                all_geodesics       geodesic in latent space
            """
    device = next(model.parameters()).device

    p0 = z1_test
    p1 = z2_test
    all_geodesics = []
    success_count = 0

    for i in range(p0.shape[0]):
        # Extract single points: [1,2]
        p0_single = p0[i].unsqueeze(0)
        p1_single = p1[i].unsqueeze(0)

        # Calculate one geodesic
        try:
            # FIX: Create the manifold curve here and push it to the GPU
            init_curve = CubicSpline(p0_single, p1_single).to(device)

            C, success = Manifold.connecting_geodesic(model, p0_single, p1_single, init_curve=init_curve)
            #if success:
            all_geodesics.append(C)
            success_count += 1
        except Exception as e:
            print(f"Failed at index {i}: {e}")

    print(f"Successfully calculated {success_count}/{p0.shape[0]} geodesics.")
    return all_geodesics


def run_stochman_benchmark(dataset_cfg, fw_cfg,  dataset_type, shape_name, results_dir, device, vae_model):
    """
    Evaluates the NODE framework against the benchmark test set.
    """
    title_complement = '' if shape_name in ['None'] else '('+shape_name+')'
    print("\n" + "-" * 40)
    print(f"🧠 Stochman Benchmark Evaluation - {dataset_type.upper()} {title_complement}")
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
    # 2. INFERENCE
    # ==========================================
    print("\nRunning Inference...")
    start_time = time.time()

    stochman_latent_paths_BM = stochman_benchmark(vae_model, z1, z2)

    end_time = time.time()

    all_geodesic_coords = []
    for spline in stochman_latent_paths_BM:
        # A CubicSpline object is 'callable'. Passing t_eval returns the [100, 2] path.
        coords = spline(t_steps)
        all_geodesic_coords.append(coords)

    trajectories = all_geodesic_coords

    stochman_predictions = torch.stack(trajectories)
    ground_truth = ground_truth
    print(f"Ground Shape: {ground_truth.shape}")
    print(f"Stochman predictions shape: {stochman_predictions.shape}")

    inference_time = end_time - start_time
    print(f"⏱️ Inference Time ({stochman_predictions.shape[0]} paths): {inference_time:.4f} seconds.")

    # ==========================================
    # 3. CALCULATE METRICS
    # ==========================================
    print("\n📊 Calculating Latent Space Metrics...")
    riemannian_metrics = calculate_ground_metrics_chuncked(
        stochman_predictions, ground_truth, z2, vae_model, density_threshold=1.5, chunk_size=50
    )

    print("\n📏 Decoding to Euclidean Space...")
    euclidean_paths_g = decode_latente_paths(vae_model, ground_truth)
    euclidean_paths_p = decode_latente_paths(vae_model, stochman_predictions)
    euclidean_goals_g = decode_latent_goals(vae_model, z2)
    predicted_goals = euclidean_paths_p[:, -1, :]

    euclidean_metrics = calculate_euclidean_metrics(
        euclidean_paths_p, euclidean_paths_g, predicted_goals, euclidean_goals_g
    )

    # ==========================================
    # 4. SAVE RESULTS
    # ==========================================
    print(f"\n💾 Saving evidences...")
    metrics_report = {
        "framework": "Stochman",
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

    json_path = os.path.join(results_dir, "stochman_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(metrics_report, f, indent=4, cls=NumpyEncoder)

    print(f"\n✅ Metrics saved to: {json_path}")

    save_latent_paths_stochman(trajectories, dir_name=results_dir, file_name="stochman_predictions.pt")

    print(f"Ploting in progress...")
    results_space = f"{dataset_type.upper()}"
    dataset_shape = shape_name+'-Shape'
    if dataset_type == "lasa":
        results_space = f"{dataset_shape}"
    points_dim = fw_cfg['axis_points']
    dataset_fw_cfg = fw_cfg[dataset_type]
    latent_max = dataset_fw_cfg.get('latent_frame', '10')
    number_samples = stochman_predictions.shape[0]
    visualize_gtvspred_comparison_multiple(stochman_predictions, ground_truth, vae_model, device, results_space, points_dim, latent_max,
                                           num_samples=number_samples, model_name="Stochman")
    save_test_plot(save_dir=results_dir, filename=f"stochman_plots.svg")