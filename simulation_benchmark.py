"""Replay planned latent-space geodesics on a simulated Franka Panda.

Combines the graph-based geodesic computation from benchmark/eval_graph.py with the
MuJoCo replay from GeodesicMotionSkills' toy_example.py: pick A-E waypoints from a LASA
demonstration, plan a geodesic between each consecutive pair, decode it to Cartesian
poses, and drive the arm through them with IK.

Usage:
    python simulation_benchmark.py --framework graph --dataset lasa --shape N
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import yaml

from benchmark import sim_utils
from GeodesicMotionSkills.Experiments.Utils import discretized_manifold
from vae.vae_model import load_pretrained_vae


class NumpyEncoder(json.JSONEncoder):
    """Numpy-to-JSON translator, matching the one used across benchmark/eval_*.py."""

    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ==========================================
# CONFIGURATION
# ==========================================
def load_training_config(file_path, dataset, shape=None):
    """Load VAE/NODE training configs (matching main.py's logic)."""
    with open(file_path, 'r') as file:
        full_config = yaml.safe_load(file)

    dataset_config = full_config.get(dataset)
    if dataset_config is None:
        raise ValueError(f"Dataset '{dataset}' not found in {file_path}")

    if shape and shape in dataset_config:
        shape_cfg = dataset_config[shape]
        if 'dataset' in dataset_config:
            shape_cfg['dataset'] = copy.deepcopy(dataset_config['dataset'])
        return shape_cfg

    return dataset_config


def load_vae(args, device):
    """Instantiate the VAE and load its trained weights (as run_benchmark.py does)."""
    vae_cfg = load_training_config('config_files/vae_config.yaml', args.dataset, args.shape)
    dataset_shape = args.shape + '-Shape' if args.shape not in ['None'] else args.shape
    vae_cfg['training_artifacts']['model_path'] = \
        vae_cfg['training_artifacts']['model_path'].replace('{shape}', dataset_shape)

    total_dof = vae_cfg['architecture']['pos_dof'] + vae_cfg['architecture']['qua_dof']
    dummy_data = torch.randn(100, total_dof)
    vae_model = load_pretrained_vae(vae_cfg, dummy_data, device=device)
    vae_model.to(device).eval()
    print(f"VAE Model initialized with DOF={total_dof} on {device.upper()}")

    # Supply the Manifold-side attributes DiscretizedManifold reads. See sim_utils.
    return sim_utils.as_graph_manifold(vae_model)


# ==========================================
# WAYPOINTS
# ==========================================
def select_waypoints(vae_model, data_cfg, shape, device):
    """Sample A-E waypoints from one demonstration and encode them into latent space.

    benchmark/eval_graph.py draws its z1/z2 pairs from a pre-generated ground_truth.pt.
    We instead reproduce toy_example.py's simulate_model(): evenly spaced points along a
    single demo, which gives an ordered A->B->C->D->E chain to drive the robot through.
    """
    demos, xy_center, xy_scale = sim_utils.load_lasa_demos(
        data_cfg['origin_dir'],
        data_cfg['origin_file'].replace('{shape}', shape + '-Shape'),
        data_cfg['trajectory_number'],
    )
    demo = demos[data_cfg['test_id']]

    indices = np.linspace(0, demo.shape[0] - 1, data_cfg['num_waypoints'], dtype=int)
    waypoints = demo[indices, :]
    print(f"{len(indices)} waypoints selected from demo {data_cfg['test_id']} "
          f"(indices {indices.tolist()})")

    with torch.no_grad():
        z_waypoints = vae_model.encode(
            torch.tensor(waypoints, dtype=torch.float32, device=device), train_rbf=True
        )[1]

    return waypoints, z_waypoints, xy_center, xy_scale


# ==========================================
# PLANNING
# ==========================================
def build_discrete_manifold(vae_model, fw_cfg, dataset_type):
    """Discretize the latent space into a graph (benchmark/eval_graph.py:76-85)."""
    graph_size = fw_cfg['graph_size']
    latent_max = fw_cfg[dataset_type]['latent_frame']

    ran = torch.linspace(-latent_max, latent_max, graph_size)
    x, y = torch.meshgrid(ran, ran)
    grid = torch.cat((x.unsqueeze(0), y.unsqueeze(0)))

    print(f"Graph-size for manifold: {graph_size} (latent frame +/-{latent_max})")
    print("Compute graph-based manifold ...")
    return discretized_manifold.DiscretizedManifold(vae_model, grid)


def plan_geodesic(vae_model, discrete_model, p0, p1, time_steps):
    """Plan one segment, following benchmark/eval_graph.py's graph_benchmark()."""
    curve = discrete_model.connecting_geodesic(p0, p1, vae_model, vae_model.time_step)
    alpha = time_steps.to(curve.device).reshape((-1, 1))
    return curve(alpha.transpose(1, 0)).detach()


def decode_path(vae_model, z_path):
    """Decode a latent path to Cartesian positions and unit quaternions.

    Uses the vMF `loc` rather than `mean`: loc is the unit-norm direction, while mean is
    loc scaled by A(kappa) and so is NOT a valid rotation. MuJoCo needs a unit quaternion.
    train_rbf=True routes the scales through the RBF networks, which accept any batch
    size -- the fixed-128 padding in benchmark/metrics.py exists only because that path
    uses decoder_scale_qua, whose first dimension is the training batch size.
    """
    with torch.no_grad():
        pos_dist, qua_dist = vae_model.decode(z_path, train_rbf=True)

    positions = pos_dist.mean.squeeze().cpu().numpy()
    quaternions = qua_dist.loc.squeeze().cpu().numpy()
    quaternions = quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)
    return positions, quaternions


# ==========================================
# METRICS
# ==========================================
def quaternion_steps_deg(quaternions):
    """Rotation angle between consecutive decoded orientations, in degrees.

    A large value means the decoded path crossed a region where the orientation decoder
    swings abruptly. Because preprocessing trains on both +q and -q (the '+/- horsefeet'
    augmentation in get_demonstrations_paths_newVAE), the latent space contains both a +q
    and a -q copy of the data, and a geodesic that crosses between them sweeps the
    end-effector through orientations that appear in no demonstration. IK cannot follow
    that, which shows up as a large max (not mean) pose error during replay.

    abs() on the dot product quotes the true rotation, ignoring the q/-q double cover.
    """
    dots = np.einsum('ij,ij->i', quaternions[:-1], quaternions[1:])
    return np.degrees(2 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0)))


def segment_metrics(index, plan_time, ik_errors, world_path, achieved_pos, target_pos,
                    quaternions=None):
    ep = np.array([e[0] for e in ik_errors])
    eo = np.degrees([e[1] for e in ik_errors])
    steps = np.diff(world_path, axis=0)
    path_length = float(np.linalg.norm(steps, axis=1).sum())
    jerk = float(np.linalg.norm(np.diff(world_path, n=3, axis=0), axis=1).mean()) \
        if world_path.shape[0] > 3 else 0.0

    return {
        "segment": index,
        "planning_time_seconds": plan_time,
        "ik_pos_err_mm": {"mean": float(ep.mean() * 1000), "max": float(ep.max() * 1000)},
        "ik_ori_err_deg": {"mean": float(eo.mean()), "max": float(eo.max())},
        # How well the arm tracked the path it was actually given.
        "tracking_err_mm": float(np.linalg.norm(achieved_pos - world_path[-1]) * 1000),
        # End-to-end error against the original waypoint: tracking error PLUS the VAE's
        # encode/decode reconstruction error, since the geodesic ends at decode(encode(B))
        # rather than at B itself.
        "goal_err_mm": float(np.linalg.norm(achieved_pos - target_pos) * 1000),
        "path_length_m": path_length,
        "mean_jerk": jerk,
        # Large values explain a large *max* ik error: see quaternion_steps_deg().
        "max_quat_step_deg": float(quaternion_steps_deg(quaternions).max())
        if quaternions is not None and len(quaternions) > 1 else 0.0,
    }


# ==========================================
# MAIN
# ==========================================
def main():
    print("\n" + "=" * 50)
    print("Riemannian LfD - Simulation Benchmark")
    print("=" * 50)

    parser = argparse.ArgumentParser(description="Replay planned geodesics on a simulated Panda")
    parser.add_argument('--framework', type=str, default='graph', choices=['graph'],
                        help="Which planner to use")
    parser.add_argument('--dataset', type=str, default='lasa', choices=['lasa'],
                        help="Which dataset to use")
    parser.add_argument('--shape', type=str, required=True,
                        help="LASA shape (e.g. N, Angle, P, Leaf_1)")
    parser.add_argument('--no_viewer', action='store_true',
                        help="Run headless (metrics only, no live viewer)")
    parser.add_argument('--graph_size', type=int, default=None,
                        help="Override nodes per latent axis; small values give a much "
                             "faster but coarser manifold, useful for smoke tests")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help="Defaults to cpu: DiscretizedManifold builds its grid and "
                             "decoded_positions as CPU tensors and calls .numpy() on "
                             "intermediates, so a CUDA VAE trips a device mismatch. The "
                             "VAE is small and MuJoCo is CPU-bound, so there is little "
                             "to gain here anyway.")
    parser.add_argument('--seed', type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open('config_files/simulation_config.yaml', 'r') as f:
        sim_cfg = yaml.safe_load(f)
    data_cfg = sim_cfg['datasets'][args.dataset]
    mj_cfg = sim_cfg['mujoco']
    ik_cfg = sim_cfg['ik']

    device = args.device

    # ---- 1. VAE ----
    print("\n--- Loading VAE ---")
    vae_model = load_vae(args, device)

    # ---- 2. Waypoints ----
    print("\n--- Selecting waypoints ---")
    waypoints, z_waypoints, xy_center, xy_scale = select_waypoints(
        vae_model, data_cfg, args.shape, device)

    marker_points_raw = sim_utils.to_workspace(
        waypoints[:, :3], xy_center, xy_scale, mj_cfg['pos_xy_scale'])
    offset = sim_utils.compute_workspace_offset(marker_points_raw, mj_cfg['workspace_target'])
    marker_points = marker_points_raw + offset

    # ---- 3. Simulation ----
    # Started BEFORE the manifold build, which takes ~a minute: the window comes up right
    # away showing the arm at home with the A-E waypoints marked, instead of appearing
    # only once planning is done and vanishing seconds later.
    print("\n--- Initializing MuJoCo ---")
    mj_model, mj_data, site_id, qpos_ids, dof_ids, jnt_ids = sim_utils.build_mujoco_model(mj_cfg)
    sim_utils.reset_to_home(mj_model, mj_data)

    mj_viewer = None
    if not args.no_viewer:
        import mujoco.viewer
        mj_viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
        sim_utils.add_labeled_markers(mj_viewer, marker_points)

    # ---- 4. Planner ----
    print("\n--- Building discrete manifold ---")
    fw_cfg = copy.deepcopy(sim_cfg['frameworks'][args.framework])
    if args.graph_size is not None:
        fw_cfg['graph_size'] = args.graph_size
    t0 = time.perf_counter()
    discrete_model = build_discrete_manifold(vae_model, fw_cfg, args.dataset)
    graph_build_time = time.perf_counter() - t0
    print(f"[Timer] manifold construction: {graph_build_time:.4f} seconds")

    time_steps = torch.linspace(0, 1, data_cfg['time_steps'])
    results = []

    try:
        for i in range(z_waypoints.shape[0] - 1):
            print(f"\n--- Processing segment {i} ({chr(65+i)} -> {chr(66+i)}) ---")

            t0 = time.perf_counter()
            z_path = plan_geodesic(vae_model, discrete_model,
                                   z_waypoints[i], z_waypoints[i + 1], time_steps)
            plan_time = time.perf_counter() - t0
            print(f"[Timer] geodesic planning: {plan_time:.4f} seconds")

            positions, quaternions = decode_path(vae_model, z_path)
            world_path = sim_utils.to_workspace(
                positions, xy_center, xy_scale, mj_cfg['pos_xy_scale'], offset)

            t0 = time.perf_counter()
            ik_errors, achieved_pos, _ = sim_utils.execute_trajectory_in_mujoco(
                mj_model, mj_data, site_id, qpos_ids, dof_ids, jnt_ids,
                world_path, quaternions, ik_cfg, mj_cfg['waypoint_substeps'],
                viewer=mj_viewer, settle_steps=mj_cfg.get('settle_steps', 0),
            )
            replay_time = time.perf_counter() - t0
            print(f"[Timer] mujoco replay: {replay_time:.4f} seconds")

            metrics = segment_metrics(i, plan_time, ik_errors, world_path,
                                      achieved_pos, marker_points[i + 1, :3],
                                      quaternions=quaternions)
            metrics["replay_time_seconds"] = replay_time
            if metrics["max_quat_step_deg"] > 10.0:
                print(f"  [WARNING] orientation jumps {metrics['max_quat_step_deg']:.1f} deg "
                      f"between consecutive points -- the geodesic crosses a region the "
                      f"orientation decoder does not model; IK cannot follow it.")
            results.append(metrics)

        # The replay is only a few seconds long, so without this the window would close
        # almost as soon as it finished moving. Hold it until it is closed by hand.
        if mj_viewer is not None and mj_viewer.is_running():
            print("\nReplay finished -- close the viewer window to exit.")
            while mj_viewer.is_running():
                time.sleep(0.1)
    finally:
        if mj_viewer is not None:
            mj_viewer.close()

    # ---- 5. Results ----
    results_dir = os.path.join(sim_cfg['results_base_dir'], f"{args.shape}-Shape")
    os.makedirs(results_dir, exist_ok=True)

    report = {
        "framework": args.framework,
        "dataset": args.dataset,
        "shape": args.shape,
        "num_waypoints": int(data_cfg['num_waypoints']),
        "time_steps": int(data_cfg['time_steps']),
        "seed": args.seed,
        # load_pretrained_vae derives the RBF bandwidth from a RANDOM dummy tensor and it
        # is not part of the checkpoint, so it changes with the seed -- and it feeds
        # embed(), the metric, and therefore which path the planner returns. Recorded so a
        # set of numbers can be tied back to the metric that produced them.
        "rbf_beta": float(vae_model.dec_std_qua[0].beta.flatten()[0]),
        "graph_build_time_seconds": graph_build_time,
        "total_planning_time_seconds": sum(r["planning_time_seconds"] for r in results),
        "segments": results,
    }

    json_path = os.path.join(results_dir, f"sim_{args.framework}_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=4, cls=NumpyEncoder)

    print("\n" + "=" * 50)
    print(f"Metrics saved to: {json_path}")
    plan_times = [r["planning_time_seconds"] for r in results]
    print(f"Planning: {sum(plan_times):.4f}s total over {len(plan_times)} segments "
          f"(mean {np.mean(plan_times):.4f}s)")
    print("=" * 50)


if __name__ == "__main__":
    main()
