import torch
from torch.utils.data import Dataset, DataLoader
import os
import json

from node.utils.plots import visualize_sphere_metric, visualize_sphere_path_comparison, visualize_sphere_random_points_comparison, \
    save_test_plot


def calculate_test_metrics(model, test_loader, device='cpu'):
    """
    Calculates Euclidean MSE and true Geodesic MSE across the entire test set.
    """
    model.eval()

    total_euclid_mse = 0.0
    total_geodesic_mse = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            # Smart Unpacking
            if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                p1, p2, z_target = batch[0], batch[1], batch[2]
            else:
                z_target = batch[0]
                p1, p2 = z_target[:, 0, :], z_target[:, -1, :]

            p1, p2, z_target = p1.to(device), p2.to(device), z_target.to(device)
            B, T_steps, D = z_target.shape

            t_steps = torch.linspace(0, 1, T_steps).to(device)

            # Predict
            pred_traj = model(p1, p2, t_steps)
            z_pred = pred_traj[:, :, 0, :]  # Shape: [T, B, 3]
            y_pred = z_pred.permute(1, 0, 2)  # Shape: [B, T, 3]

            # 1. Euclidean MSE (Straight line distance squared in 3D)
            batch_euclid_mse = torch.mean((y_pred - z_target) ** 2).item()
            total_euclid_mse += batch_euclid_mse

            # 2. Geodesic MSE (Arc-length distance squared on the surface)
            # Normalize just in case the unconstrained model drifted slightly off radius=1.0
            y_pred_unit = y_pred / torch.norm(y_pred, dim=-1, keepdim=True).clamp(min=1e-8)
            z_target_unit = z_target / torch.norm(z_target, dim=-1, keepdim=True).clamp(min=1e-8)

            # Dot product for cosine similarity, clamped for numerical stability
            dot_prod = torch.sum(y_pred_unit * z_target_unit, dim=-1)
            dot_prod = torch.clamp(dot_prod, -1.0 + 1e-7, 1.0 - 1e-7)

            # Arc length is arccos(dot_product)
            arc_lengths = torch.acos(dot_prod)
            batch_geodesic_mse = torch.mean(arc_lengths ** 2).item()
            total_geodesic_mse += batch_geodesic_mse

            total_batches += 1

    avg_euclid_mse = total_euclid_mse / total_batches if total_batches > 0 else 0.0
    avg_geodesic_mse = total_geodesic_mse / total_batches if total_batches > 0 else 0.0

    print(f"\n📊 TEST SET RESULTS:")
    print(f"  Euclidean MSE: {avg_euclid_mse:.6f}")
    print(f"  Geodesic MSE:  {avg_geodesic_mse:.6f}\n")

    return avg_euclid_mse, avg_geodesic_mse

def test_sphere_node(trained_model, manifold, test_loader, save_dir, device='cpu', space_name=''):
    """
    Takes the already-loaded models and data, and runs the visualizations.
    """
    print("\n--- Starting Testing & Visualizations ---")

    # Ensure models are in eval mode and on the right device
    trained_model.eval()
    trained_model.to(device)

    # --- 1. Run Numerical Metrics ---
    print("Calculating Test Metrics...")
    euclid_mse, geodesic_mse = calculate_test_metrics(trained_model, test_loader, device)

    # Save metrics to file
    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, "test_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({
            "euclidean_mse": euclid_mse,
            "geodesic_mse": geodesic_mse
        }, f, indent=4)
    print(f"💾 Saved test metrics to {metrics_path}")

    # --- 2. Run Plotting Functions ---
    print("Generating Manifold Visualization...")
    visualize_sphere_metric(manifold, space_name)
    save_test_plot(save_dir=save_dir, filename=f"{space_name} Manifold.svg")

    print("Generating Path Comparisons...")
    number_samples = 5
    visualize_sphere_path_comparison(trained_model, manifold, test_loader, space_name, num_samples=number_samples)
    save_test_plot(save_dir=save_dir, filename=f"{number_samples}-Path Comparisons.svg")
    number_samples = 15
    visualize_sphere_path_comparison(trained_model, manifold, test_loader, space_name, num_samples=number_samples)
    save_test_plot(save_dir=save_dir, filename=f"{number_samples}-Path Comparisons.svg")

    print("Generating Random Points Comparison...")
    visualize_sphere_random_points_comparison(trained_model, manifold, test_loader, space_name)
    save_test_plot(save_dir=save_dir, filename=f"Trajectory with random start-end point.svg")

    print("✅ Testing complete!")