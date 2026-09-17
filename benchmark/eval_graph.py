import os
import time
import json
import torch
import numpy as np

from GeodesicMotionSkills.Experiments.Utils import discretized_manifold
from benchmark.data_utils import load_test_config, save_latent_paths_stochman
from benchmark.metrics import calculate_ground_metrics_chuncked, decode_latente_paths, decode_latent_goals, \
    calculate_euclidean_metrics
from node.utils.plots import visualize_gtvspred_comparison_multiple, save_test_plot


def graph_benchmark(model , z1_test, z2_test, time_steps, discrete_model=None):
    """Computing geodesic between two points on the manifold
            Input:
                model:              an instance of the class VAE()
                ref_trajectory:     a reference trajectory selected from the test_set/training_set
                                    (It can be any trajectory but it has been selected from training
                                    set to make sure the points are already on the manifold)
                discrete_model:     a discrete graph representing the manifold
            output:
                lt:                 geodesic in latent space
            """
    lt = []

    for i in range(z1_test.shape[0]):
        p0 = z1_test[i]
        p1 = z2_test[i]

        # Compute geodesic between p0 and p1
        curve = discrete_model.connecting_geodesic(p0, p1, model,
                                                   model.time_step)  # ---> Using Dijkstra's algorithm on a set G of nodes.

        # print("Compute visualization data...")
        alpha = time_steps.to(curve.device).reshape((-1, 1))
        # print("end of alpha")

        latent_curves = curve(alpha.transpose(1, 0)).detach().numpy()  # Latent geodesic
        lt.append(latent_curves)

    print(f"Successfully calculated {len(lt)}/{z1_test.shape[0]} geodesics.")
    return lt


def run_graph_benchmark(dataset_cfg, fw_cfg, dataset_type, shape_name, results_dir, device, vae_model):
    """
    Evaluates the NODE framework against the benchmark test set.
    """
    title_complement = '' if shape_name in ['None'] else '('+shape_name+')'
    print("\n" + "-" * 40)
    print(f"🧠 Graph Benchmark Evaluation - {dataset_type.upper()} {title_complement}")
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
    t_steps = torch.linspace(0, 1, t_resolution)
    print(f"Time Resolution: {t_resolution}")

    # ==========================================
    # 2. PREPARE DISCRETE MANIFOLD
    # ==========================================
    graph_size = fw_cfg['graph_size']
    dataset_fw_cfg = fw_cfg[dataset_type]
    latent_max = dataset_fw_cfg.get('latent_frame', '10')

    ran = torch.linspace(-latent_max, latent_max, graph_size)
    x, y = torch.meshgrid(ran, ran)
    grid = torch.cat((x.unsqueeze(0), y.unsqueeze(0)))
    print(f"Graph-size for manifold: {graph_size}")
    print("Compute graph-based manifold ...")
    discrete_model = discretized_manifold.DiscretizedManifold(vae_model, grid)

    # ==========================================
    # 3. INFERENCE
    # ==========================================
    print("\nRunning Inference...")

    start_time = time.time()

    graph_latent_paths_BM = graph_benchmark(vae_model, z1, z2, t_steps, discrete_model=discrete_model)

    end_time = time.time()

    graph_predictions = torch.stack([
        torch.as_tensor(path, dtype=torch.float32, device=device)
        for path in graph_latent_paths_BM
    ])
    graph_predictions = graph_predictions.squeeze()
    ground_truth = ground_truth
    print(f"Ground Shape: {ground_truth.shape}")
    print(f"Graph predictions shape: {graph_predictions.shape}")

    inference_time = end_time - start_time
    print(f"⏱️ Inference Time ({graph_predictions.shape[0]} paths): {inference_time:.4f} seconds.")

    # ==========================================
    # 4. CALCULATE METRICS
    # ==========================================
    print("\n📊 Calculating Latent Space Metrics...")
    riemannian_metrics = calculate_ground_metrics_chuncked(
        graph_predictions, ground_truth, z2, vae_model, density_threshold=1.5, chunk_size=50
    )

    print("\n📏 Decoding to Euclidean Space...")
    euclidean_paths_g = decode_latente_paths(vae_model, ground_truth)
    euclidean_paths_p = decode_latente_paths(vae_model, graph_predictions)
    euclidean_goals_g = decode_latent_goals(vae_model, z2)
    predicted_goals = euclidean_paths_p[:, -1, :]

    euclidean_metrics = calculate_euclidean_metrics(
        euclidean_paths_p, euclidean_paths_g, predicted_goals, euclidean_goals_g
    )

    # ==========================================
    # 5. SAVE RESULTS
    # ==========================================
    print(f"\n💾 Saving evidences...")
    metrics_report = {
        "framework": "Graph",
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

    json_path = os.path.join(results_dir, "graph_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(metrics_report, f, indent=4, cls=NumpyEncoder)

    print(f"\n✅ Metrics saved to: {json_path}")

    save_latent_paths_stochman(graph_latent_paths_BM, dir_name=results_dir, file_name="graph_predictions.pt")

    print(f"Ploting in progress...")
    results_space = f"{dataset_type.upper()}"
    dataset_shape = shape_name+'-Shape'
    if dataset_type == "lasa":
        results_space = f"{dataset_shape}"
    points_dim = fw_cfg['axis_points']
    dataset_fw_cfg = fw_cfg[dataset_type]
    latent_max = dataset_fw_cfg.get('latent_frame', '10')
    number_samples = graph_predictions.shape[0]
    visualize_gtvspred_comparison_multiple(graph_predictions, ground_truth, vae_model, device, results_space, points_dim, latent_max,
                                           num_samples=number_samples, model_name="Graph")
    save_test_plot(save_dir=results_dir, filename=f"graph_plots.svg")