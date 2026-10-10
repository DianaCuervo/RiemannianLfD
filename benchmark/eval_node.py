import os
import torch

from benchmark.data_utils import save_latent_paths_node, save_json
from benchmark.metrics import evaluate_predictions
from benchmark.off_manifold import DENSE_SAMPLES
from benchmark.timing import QueryTimer, summarize, device_label, WARMUP_QUERIES
from node.evaluation.node_test import load_trained_node
from node.utils.plots import save_test_plot, visualize_gtvspred_comparison_multiple


def node_model_name(model_name_template, dataset_type, shape=None, task=None, beta=None, run_suffix=None):
    """Name of a trained NODE, as main_new.py's main_iterative_vae builds it (keep the two in sync).

    model_name_template: frameworks.node.<dataset>.model_name from benchmark_config.yaml.
    e.g. lasa, N, beta 5, energy_imitation -> NODE_lasa_N-Shape_beta_scale_5_RiemannianMSE_EGI_energy_imitation
         toy, beta 5, energy_imitation    -> NODE_toy_beta_scale_5_RiemannianMSE_EGI_energy_imitation
    """
    if dataset_type == 'lasa':
        complement = f"{shape}-Shape"
        if beta is not None:
            complement = f"{complement}_beta_scale_{beta}"
        model_name = model_name_template.replace('{shape}', f"{dataset_type}_{complement}_")
    elif dataset_type == 'lerobot':
        model_name = model_name_template.replace('{task}', f"{dataset_type}_{task}_")
    else:
        complement = f"{dataset_type}_beta_scale_{beta}_" if beta is not None else f"{dataset_type}_"
        model_name = model_name_template.replace('{shape}', complement)

    if run_suffix:
        model_name = f"{model_name}_{run_suffix}"
    return model_name


def run_node_benchmark(dataset_cfg, fw_cfg, bench, results_dir, device, vae_model, run_suffix=None):
    """
    Evaluates one NODE variant (run_suffix) against the benchmark test set.
    """
    dataset_type = dataset_cfg['type']
    shape_name = bench['shape']
    variant = run_suffix or 'default'
    tag = f"node_{run_suffix}" if run_suffix else "node"
    print("\n" + "-" * 40)
    print(f"🧠 NODE Benchmark Evaluation - {bench['title']} | Variant: {variant}")
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
    # 2. LOAD NODE
    # ==========================================
    print("Loading NODE...")
    dataset_fw_cfg = fw_cfg[dataset_type]
    model_name_template = dataset_fw_cfg.get('model_name', 'NODE_{shape}RiemannianMSE_EGI')
    model_name = node_model_name(model_name_template, dataset_type, shape_name, beta=bench['beta'],
                                 run_suffix=run_suffix)

    if dataset_type == 'lasa':
        latent_d = dataset_fw_cfg[shape_name]['latent_dim']
        hidden_d = dataset_fw_cfg[shape_name]['hidden_dim']
    else:
        latent_d = dataset_fw_cfg['latent_dim']
        hidden_d = dataset_fw_cfg['hidden_dim']

    # 1. Figure out where the model is saved based on the config
    model_dir = dataset_fw_cfg.get('model_dir', f"./models/node/{dataset_type}")
    model_path = os.path.join(model_dir, f"{model_name}.pth")

    my_model = load_trained_node(
        model_path=model_path,
        latent_dim=latent_d,
        hidden_dim=hidden_d,
        device=device
    )

    print(f"NODE Model initialized with latent_dim={latent_d} and hidden_dim={hidden_d}")

    # ==========================================
    # 3. INFERENCE (per query, batch size 1, clock stops at the latent path)
    # ==========================================
    print("\nRunning Inference...")
    num_queries = z1.shape[0]

    def solve(i, t):
        return my_model(z1[i:i + 1], z2[i:i + 1], t)[:, 0, 0, :]  # [T, 2]

    proxy_list, times = [], []
    with torch.no_grad():
        for i in range(min(WARMUP_QUERIES, num_queries)):
            solve(i, t_steps)

        for i in range(num_queries):
            with QueryTimer(device) as timer:
                path = solve(i, t_steps)
            times.append(timer.elapsed)
            proxy_list.append(path)

        # Batched throughput: all queries in one call (a NODE advantage, but not the per-query time)
        with QueryTimer(device) as timer:
            my_model(z1, z2, t_steps)
        batched_time = timer.elapsed

        # Dense paths for the off-manifold %, untimed
        dense_geos = my_model(z1, z2, t_dense)[:, :, 0, :].permute(1, 0, 2)

    proxy_geos = torch.stack(proxy_list)
    timing = summarize(times)
    print(f"Ground Shape: {ground_truth.shape}")
    print(f"Proxy Shape: {proxy_geos.shape}")
    print(f"⏱️ Per-query time: median {timing['median_s']:.4f} s (IQR {timing['iqr_s']:.4f} s) | "
          f"batched ({num_queries} paths): {batched_time:.4f} s")

    # ==========================================
    # 4. CALCULATE METRICS
    # ==========================================
    metrics = evaluate_predictions(proxy_geos, dense_geos, bench, vae_model)

    # ==========================================
    # 5. SAVE RESULTS
    # ==========================================
    print(f"\n💾 Saving evidences...")
    metrics_report = {
        "framework": "NODE",
        "variant": variant,
        "model_name": model_name,
        "dataset": dataset_type,
        "shape": shape_name,
        "beta_scale": bench['beta'],
        "latent_dim": latent_d,
        "hidden_dim": hidden_d,
        **metrics,
    }
    save_json(metrics_report, os.path.join(results_dir, f"{tag}_metrics.json"))

    label = device_label(device)
    timing_report = {
        "framework": "NODE",
        "variant": variant,
        "device": label,
        "warmup_queries": WARMUP_QUERIES,
        "per_query": timing,
        "batched_throughput": {
            "num_queries": num_queries,
            "total_s": batched_time,
            "per_query_equivalent_s": batched_time / num_queries,
        },
    }
    save_json(timing_report, os.path.join(results_dir, f"{tag}_timing_{label}.json"))

    save_latent_paths_node(proxy_geos, dir_name=results_dir, file_name=f"{tag}_proxy-geodesics.pt")

    print(f"Ploting in progress...")
    points_dim = fw_cfg['axis_points']
    dataset_fw_cfg = fw_cfg[dataset_type]
    latent_max = dataset_fw_cfg.get('latent_frame', '10')
    number_samples = proxy_geos.shape[0]
    visualize_gtvspred_comparison_multiple(proxy_geos, ground_truth, vae_model, device, bench['title'], points_dim,
                                           latent_max, num_samples=number_samples, model_name=f"NODE ({variant})")
    save_test_plot(save_dir=results_dir, filename=f"{tag}_plots.svg")
