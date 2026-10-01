import time

import torch
import torch.nn as nn

from benchmark.sim_utils import as_embedded_manifold
from GeodesicMotionSkills.Experiments.Utils import discretized_manifold
from node.utils.plots import visualize_metric, visualize_path_comparison, visualize_random_points_comparison


################################################### Testing the VAE with the graph-based planner
class GraphPlanner(nn.Module):
    """Graph-based geodesics behind the same interface as GoalConditionedNODE.

    forward(p1, p2, t_steps) returns [T, Batch, 2 (pos/vel), latent_dim] for batched p1/p2,
    or [T, latent_dim, 2 (pos/vel)] for single points -- exactly what the NODE returns -- so
    the NODE plotting helpers in node/utils/plots.py can be reused unchanged.
    """

    def __init__(self, vae_model, discrete_model):
        super().__init__()
        self.vae_model = vae_model
        self.discrete_model = discrete_model

    def geodesic(self, p1, p2, t_steps):
        # Shortest path on the graph, smoothed by a spline (as benchmark/eval_graph.py does)
        curve = self.discrete_model.connecting_geodesic(p1, p2, self.vae_model, self.vae_model.time_step)
        z_path = curve(t_steps.to(curve.device).reshape(1, -1)).detach()
        return z_path.reshape(-1, p1.shape[-1])  # [T, latent_dim]

    def forward(self, p1, p2, t_steps):
        batched = p1.dim() > 1
        p1_b, p2_b = (p1, p2) if batched else (p1.unsqueeze(0), p2.unsqueeze(0))
        z_traj = torch.stack([self.geodesic(a, b, t_steps) for a, b in zip(p1_b, p2_b)], dim=1)  # [T, B, d]
        # Velocities only to match the NODE's output layout; the plots use positions only
        v_traj = torch.gradient(z_traj, spacing=(t_steps.to(z_traj.device),), dim=0)[0]
        if batched:
            return torch.stack([z_traj, v_traj], dim=2)  # [T, B, 2, d]
        # torch.stack([z, v], dim=2) on [T, d] tensors, as the NODE does for single points
        return torch.stack([z_traj[:, 0], v_traj[:, 0]], dim=2)  # [T, d, 2]


#Building the graph planner (counterpart of load_trained_node)
def build_graph_planner(vae_model, latent_frame=15, graph_size=100):
    # DiscretizedManifold builds CPU tensors and calls .numpy() on intermediates
    vae_model = as_embedded_manifold(vae_model.cpu())

    ran = torch.linspace(-latent_frame, latent_frame, graph_size)
    x, y = torch.meshgrid(ran, ran, indexing='ij')
    grid = torch.cat((x.unsqueeze(0), y.unsqueeze(0)))

    print(f"Building graph-based manifold: {graph_size}x{graph_size} nodes over +/-{latent_frame} ...")
    start_time = time.time()
    discrete_model = discretized_manifold.DiscretizedManifold(vae_model, grid)
    print(f"Graph built in {time.time() - start_time:.1f} seconds")

    planner = GraphPlanner(vae_model, discrete_model)
    planner.eval()
    return planner

def test_vae_with_graph(vae_model, test_loader, space_name='', latent_frame=15, graph_size=100, num_samples=5):
    print("Testing VAE with the graph-based planner...")
    planner = build_graph_planner(vae_model, latent_frame=latent_frame, graph_size=graph_size)
    visualize_metric(vae_model, space_name, latent_frame)
    visualize_path_comparison(planner, vae_model, test_loader, num_samples=num_samples)
    visualize_random_points_comparison(planner, vae_model, test_loader)
