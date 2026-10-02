import os
import json
import torch
import numpy as np

from benchmark.data_utils import load_test_config
from benchmark.metrics import calculate_ground_metrics_chuncked, decode_latente_paths, decode_latent_goals, \
    calculate_euclidean_metrics


class NumpyEncoder(json.JSONEncoder):
    """Numpy/Torch-to-JSON translator."""
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        return super().default(obj)


def evaluate_saved_stochman(dataset_cfg, dataset_type, shape_name, results_dir, device, vae_model,
                            predictions_file="stochman_predictions.pt",
                            output_file="stochman_metrics_recomputed.json",
                            density_threshold=1.5, chunk_size=50):
    """
    Skips inference: loads previously saved Stochman latent paths and only
    recomputes the metrics + saves the report.
    The output goes to a separate JSON so the original stochman_metrics.json is not overwritten.
    """
    print("\n" + "-" * 40)
    print(f"🧪 Stochman metrics from saved paths - {dataset_type.upper()} ({shape_name})")
    print("-" * 40)

    # ==========================================
    # 1. LOAD DATA + SAVED PREDICTIONS
    # ==========================================
    gt_data_path = dataset_cfg['gt_data'].replace('{shape}', shape_name)
    z1, z2, ground_truth = load_test_config(filename=os.path.basename(gt_data_path),
                                            directory=os.path.dirname(gt_data_path))
    z2 = z2.to(device)
    ground_truth = ground_truth.to(device)

    pred_path = os.path.join(results_dir, predictions_file)
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"❌ No saved predictions at {pred_path}")
    stochman_predictions = torch.load(pred_path, map_location=device, weights_only=True)

    if stochman_predictions.shape != ground_truth.shape:
        # Failed geodesics in stochman_benchmark() stop the loop early -> fewer paths than test pairs.
        n = min(stochman_predictions.shape[0], ground_truth.shape[0])
        print(f"⚠️ Shape mismatch: preds {tuple(stochman_predictions.shape)} vs GT {tuple(ground_truth.shape)}. "
              f"Truncating both to first {n} paths.")
        stochman_predictions, ground_truth, z2 = stochman_predictions[:n], ground_truth[:n], z2[:n]
    print(f"Predictions: {tuple(stochman_predictions.shape)} | Ground truth: {tuple(ground_truth.shape)}")

    # Reuse the timing from the original run (inference is not repeated here).
    inference_time = None
    old_json = os.path.join(results_dir, "stochman_metrics.json")
    if os.path.exists(old_json):
        with open(old_json) as f:
            inference_time = json.load(f).get("inference_time_seconds")

    # ==========================================
    # 2. CALCULATE METRICS
    # ==========================================
    print("\n📊 Calculating Latent Space Metrics...")
    riemannian_metrics = calculate_ground_metrics_chuncked(
        stochman_predictions, ground_truth, z2, vae_model,
        density_threshold=density_threshold, chunk_size=chunk_size
    )

    print("\n📏 Decoding to Euclidean Space...")
    with torch.no_grad():  # decoding only, no gradients needed
        euclidean_paths_g = decode_latente_paths(vae_model, ground_truth)
        euclidean_paths_p = decode_latente_paths(vae_model, stochman_predictions)
        euclidean_goals_g = decode_latent_goals(vae_model, z2)
        predicted_goals = euclidean_paths_p[:, -1, :]

        euclidean_metrics = calculate_euclidean_metrics(
            euclidean_paths_p, euclidean_paths_g, predicted_goals, euclidean_goals_g
        )

    # ==========================================
    # 3. SAVE RESULTS
    # ==========================================
    metrics_report = {
        "framework": "Stochman",
        "dataset": dataset_type,
        "shape": shape_name,
        "num_paths": int(stochman_predictions.shape[0]),
        "inference_time_seconds": inference_time,
        "riemannian_metrics": riemannian_metrics,
        "euclidean_metrics": euclidean_metrics,
    }

    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, output_file)
    with open(json_path, 'w') as f:
        json.dump(metrics_report, f, indent=4, cls=NumpyEncoder)
    print(f"\n✅ Metrics saved to: {json_path}")

    return metrics_report


def main():
    import argparse
    # Reuse the exact setup helpers of the main pipeline (config + VAE loading)
    from run_benchmark import load_benchmark_config, load_respective_vae

    parser = argparse.ArgumentParser(description="Recompute Stochman metrics from saved predictions")
    parser.add_argument('--dataset', type=str, required=True, choices=['toy', 'lasa', 'robot'])
    parser.add_argument('--shape', type=str, default='None', help="Specific shape for LASA (e.g., N, Angle)")
    parser.add_argument('--predictions_file', type=str, default="stochman_predictions.pt")
    parser.add_argument('--output_file', type=str, default="stochman_metrics_recomputed.json")
    parser.add_argument('--chunk_size', type=int, default=50)
    parser.add_argument('--density_threshold', type=float, default=1.5)
    args = parser.parse_args()

    dataset_cfg = load_benchmark_config("config_files/benchmark_config.yaml", args.dataset)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae_model = load_respective_vae(args, device)

    dataset_name_with_shape = args.shape + '-Shape' if args.shape not in ['None'] else args.dataset
    results_dir = os.path.join(dataset_cfg['results_base_dir'], dataset_name_with_shape)

    evaluate_saved_stochman(dataset_cfg, args.dataset, args.shape, results_dir, device, vae_model,
                            predictions_file=args.predictions_file, output_file=args.output_file,
                            density_threshold=args.density_threshold, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
