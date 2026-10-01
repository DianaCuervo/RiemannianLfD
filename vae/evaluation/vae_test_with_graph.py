import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from benchmark.sim_utils import as_embedded_manifold
from GeodesicMotionSkills.Experiments.Utils import discretized_manifold
from vae.vae_model import get_M
from vae.utils.plots import plot_metric_with_latent_points, plot_measure_and_mf, plot_task_space


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
    # DiscretizedManifold stores the decoded grid in a hard-coded [N, 3] buffer
    # (discretized_manifold.py:64), so it only accepts 3-D positions (lasa, lerobot)
    if vae_model.pos_dof != 3:
        raise ValueError(f"The graph planner needs a VAE with 3-D positions, got pos_dof={vae_model.pos_dof} "
                         f"(the toy VAE is R2 x S2)")
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


def decode_latent_path(vae_model, z_path):
    """Decode a [T, latent_dim] path to positions [T, pos_dof] and unit quaternions [T, qua_dof].

    Uses the vMF `loc` (unit norm) rather than `mean` (loc scaled by A(kappa)), as
    simulation_benchmark.decode_path does.
    """
    with torch.no_grad():
        pos_dist, qua_dist = vae_model.decode(z_path, train_rbf=True)
    positions = pos_dist.mean.reshape(-1, vae_model.pos_dof).cpu().numpy()
    quaternions = qua_dist.loc.reshape(-1, vae_model.qua_dof).cpu().numpy()
    quaternions = quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)
    return positions, quaternions


def compute_measure_and_mf(vae_model, latent_frame, point_dim=100):
    """Variance measure and magnification factor over a [point_dim x point_dim] latent grid.

    Port of auxiliary_tests.py::Tests.geodesic_computation, vectorised. The embedding is
    [decoded mean (p) | decoded std / inverse concentration (p)], so the variance measure
    sums from index p onwards -- the original hard-coded 5, which only fits the toy VAE.
    Both grids are [x, y]-indexed.
    """
    ran = torch.linspace(-latent_frame, latent_frame, point_dim)
    x, y = torch.meshgrid(ran, ran, indexing='ij')
    grid_pts = torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)

    embedded, _, metric = get_M(vae_model, grid_pts)
    embedded = embedded.reshape(-1, embedded.shape[-1])
    metric = metric.reshape(-1, grid_pts.shape[1], grid_pts.shape[1]).double()

    mf = 0.5 * torch.log(torch.abs(torch.det(metric)))
    measure = torch.log(embedded[:, vae_model.p:].sum(dim=-1))
    return (measure.reshape(point_dim, point_dim).cpu().numpy(),
            mf.reshape(point_dim, point_dim).cpu().numpy())


def sample_segments(num_points, num_random_segments, seed=None):
    """The full demo (first -> last point), plus random sub-segments of it."""
    rng = np.random.default_rng(seed)
    segments = [(0, num_points - 1)]
    for _ in range(num_random_segments):
        start, end = sorted(rng.choice(num_points, size=2, replace=False).tolist())
        segments.append((start, end))
    return segments


def test_vae_with_graph(vae_model, trajectories, test_demo, results_dir, space_name='', latent_frame=15,
                        graph_size=100, num_random_segments=5, time_steps=500, seed=None):
    """Graph geodesics between points of a held-out demo, as toy_example.py's test_model.

    trajectories: all VAE training demos (load_vae_trajectories), for the latent scatter.
    test_demo: one raw demo [T, dof], whose encoded points give the geodesic end-points.
    """
    print("Testing VAE with the graph-based planner...")
    planner = build_graph_planner(vae_model, latent_frame=latent_frame, graph_size=graph_size)
    vae_model = planner.vae_model
    pos_dof = vae_model.pos_dof

    with torch.no_grad():
        z_train = vae_model.encode(torch.from_numpy(np.vstack(trajectories)).float(), train_rbf=True)[1]
        z_demo = vae_model.encode(torch.from_numpy(test_demo).float(), train_rbf=True)[1]

    segments = sample_segments(test_demo.shape[0], num_random_segments, seed)
    t_steps = torch.linspace(0, 1, time_steps)

    z_paths, decoded_positions, decoded_quaternions, segment_results = [], [], [], []
    for start, end in segments:
        t0 = time.perf_counter()
        z_path = planner.geodesic(z_demo[start], z_demo[end], t_steps)
        plan_time = time.perf_counter() - t0
        positions, quaternions = decode_latent_path(vae_model, z_path)

        demo_part = test_demo[start:end + 1, :pos_dof]
        segment_results.append({
            "start_index": start,
            "end_index": end,
            "planning_time_seconds": plan_time,
            # The geodesic is pinned to encode(demo) in latent space, so these are the VAE's
            # encode/decode reconstruction errors at the two end-points
            "start_error": float(np.linalg.norm(positions[0] - test_demo[start, :pos_dof])),
            "end_error": float(np.linalg.norm(positions[-1] - test_demo[end, :pos_dof])),
            "geodesic_length": float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
            "demo_length": float(np.linalg.norm(np.diff(demo_part, axis=0), axis=1).sum()),
        })
        print(f"Segment {start}->{end}: planned in {plan_time:.3f}s | "
              f"length {segment_results[-1]['geodesic_length']:.3f} "
              f"(demo {segment_results[-1]['demo_length']:.3f})")

        z_paths.append(z_path.cpu().numpy())
        decoded_positions.append(positions)
        decoded_quaternions.append(quaternions)

    z_train, z_demo = z_train.cpu().numpy(), z_demo.cpu().numpy()

    print("Computing variance measure and magnification factor...")
    measure, mf = compute_measure_and_mf(vae_model, latent_frame)

    plot_metric_with_latent_points(vae_model, z_train, space_name, latent_frame, results_dir,
                                   f"{space_name} Manifold with Graph Geodesics.svg",
                                   z_demo=z_demo, z_paths=z_paths)
    plot_measure_and_mf(measure, mf, latent_frame, z_train, z_paths, results_dir)
    plot_task_space(test_demo[:, :pos_dof], decoded_positions, segments, results_dir)

    os.makedirs(results_dir, exist_ok=True)
    np.savez(os.path.join(results_dir, "geodesics.npz"),
             segments=np.array(segments), z_paths=np.stack(z_paths),
             positions=np.stack(decoded_positions), quaternions=np.stack(decoded_quaternions),
             test_demo=test_demo, z_demo=z_demo)
    with open(os.path.join(results_dir, "summary.json"), 'w') as f:
        json.dump({"latent_frame": latent_frame, "graph_size": graph_size, "time_steps": time_steps,
                   "seed": seed, "segments": segment_results}, f, indent=4)
    print(f"✅ Graph test results saved to {results_dir}")
    return segment_results
