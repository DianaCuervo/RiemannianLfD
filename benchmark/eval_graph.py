import os
import torch

from GeodesicMotionSkills.Experiments.Utils import discretized_manifold
from benchmark.data_utils import save_latent_paths_stochman, save_json
from benchmark.metrics import evaluate_predictions
from benchmark.off_manifold import DENSE_SAMPLES
from benchmark.timing import QueryTimer, summarize, device_label, WARMUP_QUERIES
from node.utils.plots import visualize_gtvspred_comparison_multiple, save_test_plot


def graph_benchmark(model , z1_test, z2_test, time_steps, discrete_model=None, dense_time_steps=None, device='cpu'):
    """Computing geodesic between two points on the manifold
            Input:
                model:              an instance of the class VAE()
                z1_test, z2_test:   start and goal points of every query
                time_steps:         the T evaluation times of each latent path
                discrete_model:     a discrete graph representing the manifold
                dense_time_steps:   the S evaluation times for the off-manifold % (untimed)
            output:
                lt:                 geodesics in latent space, each [T, 2]
                lt_dense:           the same geodesics at dense_time_steps, each [S, 2]
                times:              per-query wall-clock times (clock stops at the latent path)
            """
    def solve(i, alpha):
        p0 = z1_test[i]
        p1 = z2_test[i]

        # Compute geodesic between p0 and p1
        curve = discrete_model.connecting_geodesic(p0, p1, model,
                                                   model.time_step)  # ---> Using Dijkstra's algorithm on a set G of nodes.
        alpha = alpha.to(curve.device).reshape((-1, 1))
        return curve, curve(alpha.transpose(1, 0)).detach().reshape(alpha.shape[0], -1)  # Latent geodesic

    for i in range(min(WARMUP_QUERIES, z1_test.shape[0])):
        solve(i, time_steps)

    lt, lt_dense, times = [], [], []
    for i in range(z1_test.shape[0]):
        with QueryTimer(device) as timer:
            curve, latent_curve = solve(i, time_steps)
        times.append(timer.elapsed)
        lt.append(latent_curve)

        alpha_dense = dense_time_steps.to(curve.device).reshape((1, -1))
        lt_dense.append(curve(alpha_dense).detach().reshape(alpha_dense.shape[1], -1))

    print(f"Successfully calculated {len(lt)}/{z1_test.shape[0]} geodesics.")
    return lt, lt_dense, times


def run_graph_benchmark(dataset_cfg, fw_cfg, bench, results_dir, device, vae_model):
    """
    Evaluates the Graph framework against the benchmark test set.
    """
    dataset_type = dataset_cfg['type']
    print("\n" + "-" * 40)
    print(f"🧠 Graph Benchmark Evaluation - {bench['title']}")
    print("-" * 40)

    # ==========================================
    # 1. LOAD DATA
    # ==========================================
    z1, z2, ground_truth = bench['z1'], bench['z2'], bench['ground_truth']

    # Recreate the time steps array used during training/generation
    t_resolution = ground_truth.shape[1]
    t_steps = torch.linspace(0, 1, t_resolution)
    t_dense = torch.linspace(0, 1, DENSE_SAMPLES)
    print(f"Time Resolution: {t_resolution}")

    # ==========================================
    # 2. PREPARE DISCRETE MANIFOLD (one-off cost)
    # ==========================================
    graph_size = fw_cfg['graph_size']
    dataset_fw_cfg = fw_cfg[dataset_type]
    latent_max = dataset_fw_cfg.get('latent_frame', '10')

    ran = torch.linspace(-latent_max, latent_max, graph_size, device=device)
    x, y = torch.meshgrid(ran, ran)
    grid = torch.cat((x.unsqueeze(0), y.unsqueeze(0)))
    print(f"Graph-size for manifold: {graph_size}")
    print("Compute graph-based manifold ...")
    with QueryTimer(device) as timer:
        discrete_model = discretized_manifold.DiscretizedManifold(vae_model, grid)
    graph_construction_time = timer.elapsed
    print(f"⏱️ Graph construction: {graph_construction_time:.2f} s")

    # ==========================================
    # 3. INFERENCE
    # ==========================================
    print("\nRunning Inference...")
    graph_latent_paths_BM, graph_dense_paths, times = graph_benchmark(
        vae_model, z1, z2, t_steps, discrete_model=discrete_model, dense_time_steps=t_dense, device=device)

    graph_predictions = torch.stack([path.to(device) for path in graph_latent_paths_BM])
    graph_dense = torch.stack([path.cpu() for path in graph_dense_paths])
    timing = summarize(times)
    print(f"Ground Shape: {ground_truth.shape}")
    print(f"Graph predictions shape: {graph_predictions.shape}")
    print(f"⏱️ Per-query time: median {timing['median_s']:.4f} s (IQR {timing['iqr_s']:.4f} s)")

    # ==========================================
    # 4. CALCULATE METRICS
    # ==========================================
    metrics = evaluate_predictions(graph_predictions, graph_dense, bench, vae_model)

    # ==========================================
    # 5. SAVE RESULTS
    # ==========================================
    print(f"\n💾 Saving evidences...")
    metrics_report = {
        "framework": "Graph",
        "variant": "default",
        "dataset": dataset_type,
        "shape": bench['shape'],
        "beta_scale": bench['beta'],
        "graph_size": graph_size,
        **metrics,
        "one_off_costs": {"graph_construction_s": graph_construction_time},
    }
    save_json(metrics_report, os.path.join(results_dir, "graph_metrics.json"))

    label = device_label(device)
    timing_report = {
        "framework": "Graph",
        "variant": "default",
        "device": label,
        "warmup_queries": WARMUP_QUERIES,
        "per_query": timing,
        "graph_construction_s": graph_construction_time,
    }
    save_json(timing_report, os.path.join(results_dir, f"graph_timing_{label}.json"))

    save_latent_paths_stochman(graph_latent_paths_BM, dir_name=results_dir, file_name="graph_predictions.pt")

    print(f"Ploting in progress...")
    points_dim = fw_cfg['axis_points']
    number_samples = graph_predictions.shape[0]
    visualize_gtvspred_comparison_multiple(graph_predictions, ground_truth, vae_model, device, bench['title'], points_dim,
                                           latent_max, num_samples=number_samples, model_name="Graph")
    save_test_plot(save_dir=results_dir, filename=f"graph_plots.svg")
