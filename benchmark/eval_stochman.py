import os
import torch

from stochman.manifold import Manifold
from stochman.curves import CubicSpline
from benchmark.data_utils import save_latent_paths_stochman, save_json
from benchmark.metrics import evaluate_predictions
from benchmark.off_manifold import DENSE_SAMPLES
from benchmark.timing import QueryTimer, summarize, device_label, WARMUP_QUERIES
from node.utils.plots import visualize_gtvspred_comparison_multiple, save_test_plot


def stochman_benchmark(model , z1_test, z2_test, time_steps, dense_time_steps):
    """Computing geodesic between two points on the manifold
            Input:
                model:              an instance of the class VAE()
                z1_test:            set of starting points
                z2_test:            set of ending points
                time_steps:         the T evaluation times of each latent path
                dense_time_steps:   the S evaluation times for the off-manifold % (untimed)
            output:
                all_geodesics       geodesics in latent space, each [T, 2]
                all_dense           the same geodesics at dense_time_steps, each [S, 2]
                times               per-query wall-clock times (clock stops at the latent path)
            """
    device = next(model.parameters()).device

    p0 = z1_test
    p1 = z2_test

    def solve(i):
        # Extract single points: [1,2]
        p0_single = p0[i].unsqueeze(0)
        p1_single = p1[i].unsqueeze(0)

        # FIX: Create the manifold curve here and push it to the GPU
        init_curve = CubicSpline(p0_single, p1_single).to(device)

        C, success = Manifold.connecting_geodesic(model, p0_single, p1_single, init_curve=init_curve)
        # A CubicSpline object is 'callable'. Passing t_eval returns the [100, 2] path.
        return C, C(time_steps).detach().reshape(time_steps.shape[0], -1)

    for i in range(min(WARMUP_QUERIES, p0.shape[0])):
        solve(i)

    all_geodesics, all_dense, times = [], [], []
    success_count = 0

    for i in range(p0.shape[0]):
        # Calculate one geodesic
        try:
            with QueryTimer(device) as timer:
                C, coords = solve(i)
            #if success:
            times.append(timer.elapsed)
            all_geodesics.append(coords)
            all_dense.append(C(dense_time_steps).detach().reshape(dense_time_steps.shape[0], -1))
            success_count += 1
        except Exception as e:
            print(f"Failed at index {i}: {e}")
            break

    print(f"Successfully calculated {success_count}/{p0.shape[0]} geodesics.")
    return all_geodesics, all_dense, times


def run_stochman_benchmark(dataset_cfg, fw_cfg, bench, results_dir, device, vae_model):
    """
    Evaluates the Stochman framework against the benchmark test set.
    """
    dataset_type = dataset_cfg['type']
    print("\n" + "-" * 40)
    print(f"🧠 Stochman Benchmark Evaluation - {bench['title']}")
    print("-" * 40)

    # ==========================================
    # 1. LOAD DATA
    # ==========================================
    z1, z2, ground_truth = bench['z1'], bench['z2'], bench['ground_truth']

    # Recreate the time steps array used during training/generation
    t_resolution = ground_truth.shape[1]
    t_steps = torch.linspace(0, 1, t_resolution).to(device)
    t_dense = torch.linspace(0, 1, DENSE_SAMPLES).to(device)
    print(f"Time Resolution: {t_resolution}")

    # ==========================================
    # 2. INFERENCE
    # ==========================================
    print("\nRunning Inference...")
    trajectories, dense_trajectories, times = stochman_benchmark(vae_model, z1, z2, t_steps, t_dense)

    stochman_predictions = torch.stack(trajectories)
    stochman_dense = torch.stack(dense_trajectories)
    timing = summarize(times)
    print(f"Ground Shape: {ground_truth.shape}")
    print(f"Stochman predictions shape: {stochman_predictions.shape}")
    print(f"⏱️ Per-query time: median {timing['median_s']:.4f} s (IQR {timing['iqr_s']:.4f} s)")

    # ==========================================
    # 3. CALCULATE METRICS
    # ==========================================
    metrics = evaluate_predictions(stochman_predictions, stochman_dense, bench, vae_model)

    # ==========================================
    # 4. SAVE RESULTS
    # ==========================================
    print(f"\n💾 Saving evidences...")
    metrics_report = {
        "framework": "Stochman",
        "variant": "default",
        "dataset": dataset_type,
        "shape": bench['shape'],
        "beta_scale": bench['beta'],
        **metrics,
        "one_off_costs": {},
    }
    save_json(metrics_report, os.path.join(results_dir, "stochman_metrics.json"))

    label = device_label(device)
    timing_report = {
        "framework": "Stochman",
        "variant": "default",
        "device": label,
        "warmup_queries": WARMUP_QUERIES,
        "per_query": timing,
    }
    save_json(timing_report, os.path.join(results_dir, f"stochman_timing_{label}.json"))

    save_latent_paths_stochman(trajectories, dir_name=results_dir, file_name="stochman_predictions.pt")

    print(f"Ploting in progress...")
    points_dim = fw_cfg['axis_points']
    dataset_fw_cfg = fw_cfg[dataset_type]
    latent_max = dataset_fw_cfg.get('latent_frame', '10')
    number_samples = stochman_predictions.shape[0]
    visualize_gtvspred_comparison_multiple(stochman_predictions, ground_truth, vae_model, device, bench['title'],
                                           points_dim, latent_max, num_samples=number_samples, model_name="Stochman")
    save_test_plot(save_dir=results_dir, filename=f"stochman_plots.svg")
