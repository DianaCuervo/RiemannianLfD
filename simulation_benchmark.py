"""Replay planned latent-space paths on a simulated Franka Panda.

Combines the latent planners benchmarked in RiemannianLfD-benchmark (graph, NODE and
stochman -- see its benchmark/eval_*.py) with the MuJoCo replay from GeodesicMotionSkills'
toy_example.py: pick A-E waypoints from a LASA demonstration, plan a path between each
consecutive pair, decode it to Cartesian poses, and drive the arm through them with IK.

Each segment starts where the previous plan ended in latent space. For graph and stochman
that is the waypoint itself (both pin the curve's end-points); NODE only approaches its
goal, so starting the next segment from the true waypoint would make the arm jump.

Usage:
    python simulation_benchmark.py --framework graph --dataset lasa --shape N
    python simulation_benchmark.py --framework node --dataset lasa --shape N
    python simulation_benchmark.py --framework stochman --dataset lasa --shape N
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
from node.node_model import GoalConditionedNODE
from stochman.curves import CubicSpline
from vae.vae_model import load_pretrained_vae
from vae.data.vae_dataset import load_vae_training_points


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


def load_vae(args, data_cfg, device):
    """Instantiate the VAE and load its trained weights (as run_benchmark.py does)."""
    vae_cfg = load_training_config('config_files/vae_config.yaml', args.dataset, args.shape)
    dataset_shape = args.shape + '-Shape' if args.shape not in ['None'] else args.shape
    vae_cfg['training_artifacts']['model_path'] = \
        vae_cfg['training_artifacts']['model_path'].replace('{shape}', dataset_shape)
    vae_cfg['training_artifacts']['cluster_path'] = \
        vae_cfg['training_artifacts']['cluster_path'].replace('{shape}', dataset_shape)

    # The VAE's own training points (both quaternion signs), from which load_pretrained_vae
    # recomputes the RBF beta -- not the single-sign demos select_waypoints uses.
    training_points = load_vae_training_points({
        'type': args.dataset,
        'origin_dir': data_cfg['origin_dir'],
        'origin_file': data_cfg['origin_file'].replace('{shape}', dataset_shape),
        'trajectory_number': data_cfg['trajectory_number'],
    })

    total_dof = vae_cfg['architecture']['pos_dof'] + vae_cfg['architecture']['qua_dof']
    vae_model = load_pretrained_vae(vae_cfg, training_points, device=device)
    vae_model.to(device).eval()
    print(f"VAE Model initialized with DOF={total_dof} on {device.upper()}")

    # Supply the Manifold-side attributes the graph and stochman planners read. See sim_utils.
    return sim_utils.as_embedded_manifold(vae_model)


def load_node(fw_cfg, dataset_type, shape, device):
    """Instantiate the goal-conditioned NODE and load its trained weights.

    Builds GoalConditionedNODE directly rather than via node/evaluation/node_test.py's
    load_trained_node: that helper takes no hidden_dim, so the 512-wide N/P/Leaf_1 models
    would not load, and importing it drags in the matplotlib plotting module.
    """
    dataset_fw_cfg = fw_cfg[dataset_type]
    arch = dataset_fw_cfg[shape]
    model_path = os.path.join(dataset_fw_cfg['model_dir'],
                              dataset_fw_cfg['model_name'].replace('{shape}', shape))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find saved NODE at: {model_path}")

    node_model = GoalConditionedNODE(latent_dim=arch['latent_dim'], hidden_dim=arch['hidden_dim'])
    node_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    node_model.to(device).eval()
    print(f"NODE loaded from {model_path} "
          f"(latent_dim={arch['latent_dim']}, hidden_dim={arch['hidden_dim']})")
    return node_model, model_path


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


# Every planner returns (z_path [1, T, latent_dim], info dict), so decoding and chaining
# need not know which framework produced the path.
def plan_geodesic(vae_model, discrete_model, p0, p1, time_steps):
    """Plan one segment, following benchmark/eval_graph.py's graph_benchmark()."""
    curve = discrete_model.connecting_geodesic(p0, p1, vae_model, vae_model.time_step)
    alpha = time_steps.to(curve.device).reshape((-1, 1))
    z_path = curve(alpha.transpose(1, 0)).detach()
    return z_path.reshape(1, -1, p0.shape[-1]), {}


def plan_node(node_model, p0, p1, time_steps):
    """Roll out the goal-conditioned NODE from p0 towards p1 (benchmark/eval_node.py).

    The rollout starts exactly at p0 but is only trained to approach p1, so unlike the
    other planners its last point is not guaranteed to be the goal.
    """
    with torch.no_grad():
        pred_traj = node_model(p0.unsqueeze(0), p1.unsqueeze(0), time_steps.to(p0.device))
    # [T, Batch, (pos, vel), latent_dim] -> positions of the single batch element
    z_path = pred_traj[:, 0, 0, :]
    return z_path.unsqueeze(0), {}


def plan_stochman(vae_model, p0, p1, time_steps, fw_cfg):
    """Fit a geodesic by minimising curve energy (benchmark/eval_stochman.py).

    A copy of stochman.geodesic.geodesic_minimizing_energy -- which eval_stochman reaches
    through Manifold.connecting_geodesic -- with lr/thresh taken from the config instead
    of hardcoded, so the local stochman package stays untouched. The spline's begin/end
    are buffers, so only the interior nodes move and the end-points stay exact.
    """
    curve = CubicSpline(p0.unsqueeze(0), p1.unsqueeze(0), num_nodes=fw_cfg['num_nodes'])
    alpha = torch.linspace(0, 1, fw_cfg['eval_grid'], dtype=curve.begin.dtype,
                           device=curve.device)
    opt = torch.optim.Adam(curve.parameters(), lr=fw_cfg['lr'])

    def closure():
        opt.zero_grad()
        loss = vae_model.curve_energy(curve(alpha)).mean()
        loss.backward()
        return loss

    for k in range(fw_cfg['max_iter']):
        opt.step(closure=closure)
        max_grad = max(p.grad.abs().max() for p in curve.parameters())
        if max_grad < fw_cfg['thresh']:
            break

    with torch.no_grad():
        z_path = curve(time_steps.to(curve.device))
    info = {"converged": bool(max_grad < fw_cfg['thresh']), "iterations": k + 1,
            "final_max_grad": float(max_grad)}
    return z_path.reshape(1, -1, p0.shape[-1]), info


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
                    quaternions=None, start_gap_latent=0.0, end_gap_latent=0.0,
                    end_gap_mm=0.0):
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
        # Chaining: how far this segment's start is from the true waypoint (non-zero only
        # when the previous plan missed its goal, i.e. for NODE) ...
        "start_gap_latent": start_gap_latent,
        # ... and how far the plan itself ended from its goal, in latent space and decoded
        # into the workspace. Isolates the planner's miss from IK/tracking error and the
        # VAE reconstruction error that goal_err_mm also includes. ~0 for graph/stochman.
        "end_gap_latent": end_gap_latent,
        "end_gap_mm": end_gap_mm,
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
    parser.add_argument('--framework', type=str, default='graph',
                        choices=['graph', 'node', 'stochman'],
                        help="Which planner to use")
    parser.add_argument('--dataset', type=str, default='lasa', choices=['lasa'],
                        help="Which dataset to use")
    parser.add_argument('--shape', type=str, required=True,
                        help="LASA shape (e.g. N, Angle, P, Leaf_1)")
    parser.add_argument('--no_viewer', action='store_true',
                        help="Run headless (metrics only, no live viewer)")
    parser.add_argument('--graph_size', type=int, default=None,
                        help="Graph framework only: override nodes per latent axis; small "
                             "values give a much faster but coarser manifold, useful for "
                             "smoke tests")
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
    vae_model = load_vae(args, data_cfg, device)

    # ---- 2. Waypoints ----
    print("\n--- Selecting waypoints ---")
    waypoints, z_waypoints, xy_center, xy_scale = select_waypoints(
        vae_model, data_cfg, args.shape, device)

    marker_points_raw = sim_utils.to_workspace(
        waypoints[:, :3], xy_center, xy_scale, mj_cfg['pos_xy_scale'])
    offset = sim_utils.compute_workspace_offset(marker_points_raw, mj_cfg['workspace_target'])
    marker_points = marker_points_raw + offset

    # ---- 3. Simulation ----
    # Started BEFORE planner setup -- the graph manifold build takes ~a minute: the window
    # comes up right away showing the arm at home with the A-E waypoints marked, instead
    # of appearing only once planning is done and vanishing seconds later.
    print("\n--- Initializing MuJoCo ---")
    mj_model, mj_data, site_id, qpos_ids, dof_ids, jnt_ids = sim_utils.build_mujoco_model(mj_cfg)
    sim_utils.reset_to_home(mj_model, mj_data)

    mj_viewer = None
    if not args.no_viewer:
        import mujoco.viewer
        mj_viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
        sim_utils.add_labeled_markers(mj_viewer, marker_points)

    # ---- 4. Planner ----
    print(f"\n--- Setting up {args.framework} planner ---")
    fw_cfg = copy.deepcopy(sim_cfg['frameworks'][args.framework])
    if args.graph_size is not None:
        if args.framework == 'graph':
            fw_cfg['graph_size'] = args.graph_size
        else:
            print("  [NOTE] --graph_size ignored: it only applies to the graph framework")
    time_steps = torch.linspace(0, 1, data_cfg['time_steps'])
    node_path = None

    t0 = time.perf_counter()
    if args.framework == 'graph':
        discrete_model = build_discrete_manifold(vae_model, fw_cfg, args.dataset)
        plan = lambda p0, p1: plan_geodesic(vae_model, discrete_model, p0, p1, time_steps)
    elif args.framework == 'node':
        node_model, node_path = load_node(fw_cfg, args.dataset, args.shape, device)
        plan = lambda p0, p1: plan_node(node_model, p0, p1, time_steps)
    else:  # stochman: optimises each curve from scratch, nothing to prepare
        plan = lambda p0, p1: plan_stochman(vae_model, p0, p1, time_steps, fw_cfg)
    setup_time = time.perf_counter() - t0
    print(f"[Timer] planner setup: {setup_time:.4f} seconds")

    results = []

    try:
        # Chained: each segment starts where the previous plan ended in latent space, not at
        # the true waypoint, so the arm never jumps between segments. The goal is always the
        # true next waypoint. Only NODE can end off-goal; see the module docstring.
        z_start = z_waypoints[0]
        for i in range(z_waypoints.shape[0] - 1):
            print(f"\n--- Processing segment {i} ({chr(65+i)} -> {chr(66+i)}) ---")
            z_goal = z_waypoints[i + 1]

            t0 = time.perf_counter()
            z_path, plan_info = plan(z_start, z_goal)
            plan_time = time.perf_counter() - t0
            print(f"[Timer] {args.framework} planning: {plan_time:.4f} seconds")
            if plan_info.get("converged") is False:
                print(f"  [NOTE] stochman stopped after {plan_info['iterations']} iterations "
                      f"without converging (max |grad| {plan_info['final_max_grad']:.2e})")

            positions, quaternions = decode_path(vae_model, z_path)
            world_path = sim_utils.to_workspace(
                positions, xy_center, xy_scale, mj_cfg['pos_xy_scale'], offset)

            z_end = z_path.reshape(-1, z_path.shape[-1])[-1]
            start_gap_latent = float(torch.norm(z_start - z_waypoints[i]))
            end_gap_latent = float(torch.norm(z_end - z_goal))
            gap_pos, _ = decode_path(vae_model, torch.stack([z_end, z_goal]).unsqueeze(0))
            gap_world = sim_utils.to_workspace(gap_pos, xy_center, xy_scale,
                                               mj_cfg['pos_xy_scale'])
            end_gap_mm = float(np.linalg.norm(gap_world[0] - gap_world[1]) * 1000)

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
                                      quaternions=quaternions,
                                      start_gap_latent=start_gap_latent,
                                      end_gap_latent=end_gap_latent,
                                      end_gap_mm=end_gap_mm)
            metrics["replay_time_seconds"] = replay_time
            metrics.update(plan_info)
            if metrics["max_quat_step_deg"] > 10.0:
                print(f"  [WARNING] orientation jumps {metrics['max_quat_step_deg']:.1f} deg "
                      f"between consecutive points -- the geodesic crosses a region the "
                      f"orientation decoder does not model; IK cannot follow it.")
            if end_gap_mm > 5.0:
                print(f"  [WARNING] plan ends {end_gap_mm:.1f} mm short of waypoint "
                      f"{chr(66+i)}; the next segment starts from there.")
            results.append(metrics)

            z_start = z_end

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
        # The RBF bandwidth is not part of the checkpoint; load_pretrained_vae restores the
        # training-time value. It feeds embed(), the metric, and therefore which path the
        # graph/stochman planners return (NODE never reads the metric). Recorded so a set of
        # numbers can be tied back to the metric that produced them.
        "rbf_beta": float(vae_model.dec_std_qua[0].beta.flatten()[0]),
        "framework_config": fw_cfg,
        "node_checkpoint": node_path,
        # Graph: manifold construction. NODE: checkpoint load. Stochman: none.
        "setup_time_seconds": setup_time,
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
