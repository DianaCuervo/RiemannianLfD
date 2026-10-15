"""Replay planned latent-space paths on a simulated Franka Panda.

Combines the latent planners benchmarked in RiemannianLfD-benchmark (graph, NODE and
stochman -- see its benchmark/eval_*.py) with the MuJoCo replay from GeodesicMotionSkills'
toy_example.py: pick A-E waypoints from a LASA demonstration, plan a path between each
consecutive pair, decode it to Cartesian poses, and drive the arm through them with IK.

Each segment starts where the previous plan ended in latent space. For graph and stochman
that is the waypoint itself (both pin the curve's end-points); NODE only approaches its
goal, so starting the next segment from the true waypoint would make the arm jump.

The VAE and NODE are the ones trained for --shape and --beta_scale, resolved from
vae_config.yaml and node_config.yaml as run_benchmark.py does.

Planning time is per segment and covers only the planner, from start/goal latents to the
latent path: graph search + spline fit, the NODE rollout, or the stochman energy
minimisation. Graph construction is logged as setup time and is not part of it.

Usage:
    python simulation_benchmark.py --framework graph --shape N --beta_scale 5
    python simulation_benchmark.py --framework node --shape N --beta_scale 5 [--node_variant energy_only]
    python simulation_benchmark.py --framework stochman --shape N --beta_scale 5
"""
import argparse
import copy
import glob
import json
import math
import os
import re
import time

import mujoco
import networkx as nx
import numpy as np
import torch
import yaml

from benchmark import sim_utils
from benchmark.eval_node import node_model_name
from benchmark.timing import QueryTimer
from GeodesicMotionSkills.Experiments.Utils import discretized_manifold
from node.node_model import GoalConditionedNODE
from stochman.curves import CubicSpline
from vae.utils.config import NODE_CONFIG_PATH, beta_label, load_vae_config
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
def load_node_config(dataset, shape):
    """The 'dataset' section and the per-shape section of node_config.yaml for one shape."""
    with open(NODE_CONFIG_PATH, 'r') as file:
        dataset_config = yaml.safe_load(file)[dataset]
    if shape not in dataset_config:
        raise ValueError(f"Shape '{shape}' not found under '{dataset}' in {NODE_CONFIG_PATH}")
    return dataset_config['dataset'], dataset_config[shape]


def lasa_shapes():
    """The LASA shapes node_config.yaml defines (one VAE/NODE family per shape)."""
    with open(NODE_CONFIG_PATH, 'r') as file:
        lasa_config = yaml.safe_load(file)['lasa']
    return [key for key in lasa_config if key != 'dataset']


def available_betas(pattern):
    """beta_scale values of the files matching a glob with '*' in place of the beta."""
    regex = re.compile(re.escape(pattern).replace(r'\*', r'([^_/]+)'))
    found = {m.group(1) for path in glob.glob(pattern) if (m := regex.fullmatch(path))}
    return sorted(found, key=float)


def load_vae(args, data_cfg, device):
    """Instantiate the VAE for (shape, beta_scale) and load its trained weights.

    Returns the model, its training points (also used to size the graph grid) and the
    checkpoint path.
    """
    vae_cfg = load_vae_config(args.dataset, shape=args.shape, beta=args.beta_scale)
    model_path = vae_cfg['training_artifacts']['model_path']
    if not os.path.exists(model_path):
        beta = beta_label(args.beta_scale)
        on_disk = available_betas(model_path.replace(f"_beta_scale_{beta}", "_beta_scale_*"))
        raise FileNotFoundError(f"No VAE for {args.shape}-Shape at beta_scale {beta}: {model_path} "
                                f"(beta_scale on disk: {', '.join(on_disk) or 'none'})")

    # The VAE's own training points (both quaternion signs), from which load_pretrained_vae
    # recomputes the RBF beta -- not the single-sign demos select_waypoints uses.
    training_points = load_vae_training_points({
        'type': args.dataset,
        'origin_dir': data_cfg['origin_dir'],
        'origin_file': data_cfg['origin_file'].replace('{shape}', f"{args.shape}-Shape"),
        'trajectory_number': data_cfg['trajectory_number'],
    })

    total_dof = vae_cfg['architecture']['pos_dof'] + vae_cfg['architecture']['qua_dof']
    vae_model = load_pretrained_vae(vae_cfg, training_points, device=device)
    vae_model.to(device).eval()
    print(f"VAE Model initialized with DOF={total_dof} on {device.upper()}")

    # Supply the Manifold-side attributes the graph and stochman planners read. See sim_utils.
    return sim_utils.as_embedded_manifold(vae_model), training_points, model_path


def load_node(args, device):
    """Instantiate the goal-conditioned NODE for (shape, beta_scale, variant) and load it.

    Architecture, checkpoint name and default variant (run_suffix) come from node_config.yaml,
    the config the NODE was trained with; the file name is built by the same
    node_model_name() run_benchmark.py uses. The checkpoint's layer shape is checked against
    the config first, so a mismatch fails with both values instead of a state-dict error.
    """
    dataset_cfg, shape_cfg = load_node_config(args.dataset, args.shape)
    variant = args.node_variant or shape_cfg.get('run_suffix')
    arch = shape_cfg['architecture']
    model_name = node_model_name(dataset_cfg['model_name'], args.dataset, args.shape,
                                 beta=beta_label(args.beta_scale), run_suffix=variant)
    model_path = os.path.join(dataset_cfg['model_dir'], f"{model_name}.pth")
    if not os.path.exists(model_path):
        on_disk = sorted(os.path.basename(p) for p in glob.glob(
            os.path.join(dataset_cfg['model_dir'], f"NODE_{args.dataset}_{args.shape}-Shape_*.pth")))
        raise FileNotFoundError(f"Could not find saved NODE at: {model_path}\n"
                                f"NODE checkpoints for {args.shape}-Shape: {on_disk or 'none'}")

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    expected = (arch['hidden_dim'], 2 * arch['latent_dim'])
    found = tuple(state_dict['ode_func.net.0.weight'].shape)
    if found != expected:
        raise ValueError(f"{model_path} has hidden_dim={found[0]}, latent_dim={found[1] // 2}, but "
                         f"{NODE_CONFIG_PATH} lasa.{args.shape}.architecture says "
                         f"hidden_dim={arch['hidden_dim']}, latent_dim={arch['latent_dim']}")

    node_model = GoalConditionedNODE(latent_dim=arch['latent_dim'], hidden_dim=arch['hidden_dim'])
    node_model.load_state_dict(state_dict)
    node_model.to(device).eval()
    print(f"NODE loaded from {model_path} "
          f"(latent_dim={arch['latent_dim']}, hidden_dim={arch['hidden_dim']}, variant={variant})")
    return node_model, model_path, variant


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
class FastDiscretizedManifold(discretized_manifold.DiscretizedManifold):
    """DiscretizedManifold whose per-query planning skips the full-graph copy.

    Upstream connecting_geodesic starts with copy.deepcopy(self.mem_g) so obstacles can
    re-weight edges without spoiling the original graph. With no obstacles and no via-points
    that copy is pure overhead (about half the per-query time on a 100x100 grid) and would be
    counted as planning time. This runs the same no-obstacle branch on mem_g directly, so the
    path is identical; anything else falls back to the upstream method.
    """

    def connecting_geodesic(self, p1, p2, model, graph_id, curve=None):
        if model.env.obstacles or model.env.via_points:
            return super().connecting_geodesic(p1, p2, model, graph_id, curve)

        idx1, idx2 = self.grid_point(p1), self.grid_point(p2)
        path = nx.shortest_path(self.mem_g, source=idx1, target=idx2, weight='weight')
        weights = [self.mem_g.edges[path[k], path[k + 1]]['weight'] for k in range(len(path) - 1)]
        self.coordinates = (self.grid.view(self.grid.shape[0], -1)[:, path[1:-1]]).t()
        t = torch.tensor(weights[:-1], device=p1.device).cumsum(dim=0) / sum(weights)

        if curve is None:
            curve = CubicSpline(p1, p2).to(p1.device)
        else:
            curve.begin = p1
            curve.end = p2
        curve.fit(t, self.coordinates)
        return curve


def latent_grid(fw_cfg, dataset_type, z_points, graph_size=None):
    """Latent grid for the graph planner, as a [2, nx, ny] meshgrid, and its description.

    grid: data covers the bounding box of z_points (encoded training points and waypoints)
    plus latent_margin, with square cells of cell_size, so nx and ny can differ. The old
    fixed +/-15 frame left N's waypoint A (x = 18.6) and the training latents of N, Angle
    and P outside the grid, and the planner snapped them to its border.
    graph_size, if given, sets the node count on the longer axis instead of cell_size.
    """
    z_points = z_points.detach().cpu()
    if fw_cfg['grid'] == 'data':
        lo = z_points.min(dim=0).values - fw_cfg['latent_margin']
        hi = z_points.max(dim=0).values + fw_cfg['latent_margin']
    elif fw_cfg['grid'] == 'fixed':
        frame = float(fw_cfg[dataset_type]['latent_frame'])
        lo, hi = torch.full((2,), -frame), torch.full((2,), frame)
    else:
        raise ValueError(f"Unknown graph grid '{fw_cfg['grid']}' (expected 'data' or 'fixed')")

    extent = hi - lo
    cell = float(extent.max()) / (graph_size - 1) if graph_size else float(fw_cfg['cell_size'])
    sizes = [int(math.ceil(float(e) / cell - 1e-6)) + 1 for e in extent]
    axes = [lo[k] + cell * torch.arange(sizes[k], dtype=torch.float32) for k in range(2)]
    x, y = torch.meshgrid(axes[0], axes[1], indexing='ij')

    info = {"grid": fw_cfg['grid'], "cell_size": cell, "nodes": sizes,
            "lower": [float(a[0]) for a in axes], "upper": [float(a[-1]) for a in axes]}
    return torch.stack((x, y)), info


def build_discrete_manifold(vae_model, grid, info):
    """Discretize the latent space into a graph (benchmark/eval_graph.py:76-85)."""
    print(f"Graph grid ({info['grid']}): {info['nodes'][0]} x {info['nodes'][1]} nodes, "
          f"cell {info['cell_size']:.3f}, x [{info['lower'][0]:.2f}, {info['upper'][0]:.2f}], "
          f"y [{info['lower'][1]:.2f}, {info['upper'][1]:.2f}]")
    print("Compute graph-based manifold ...")
    return FastDiscretizedManifold(vae_model, grid)


def warn_outside_grid(z_points, info):
    lower, upper = torch.tensor(info['lower']), torch.tensor(info['upper'])
    outside = ((z_points.cpu() < lower) | (z_points.cpu() > upper)).any(dim=1)
    for i in torch.nonzero(outside).flatten().tolist():
        print(f"  [WARNING] waypoint {chr(65 + i)} at {z_points[i].cpu().numpy().round(2)} lies outside "
              f"the graph grid; the planner will snap it to the nearest border node")


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
    """Fit a geodesic by minimising curve energy over a CubicSpline.

    The loop of stochman.geodesic.geodesic_minimizing_energy (which eval_stochman reaches
    through Manifold.connecting_geodesic), with its settings taken from the config instead
    of hardcoded, so the local stochman package stays untouched. Its hardcoded lr 1e-3 /
    150 iterations barely move the curve off the straight line, so the default here stops
    on an energy plateau instead (stop: plateau); stop: grad keeps the original rule.
    The spline's begin/end are buffers, so only the interior nodes move and the end-points
    stay exact.
    """
    curve = CubicSpline(p0.unsqueeze(0), p1.unsqueeze(0), num_nodes=fw_cfg['num_nodes']).to(p0.device)
    alpha = torch.linspace(0, 1, fw_cfg['eval_grid'], dtype=curve.begin.dtype, device=curve.device)
    opt = torch.optim.Adam(curve.parameters(), lr=fw_cfg['lr'])

    best, stall, stop_reason = math.inf, 0, "max_iter"
    for k in range(fw_cfg['max_iter']):
        opt.zero_grad()
        loss = vae_model.curve_energy(curve(alpha)).mean()
        loss.backward()
        opt.step()
        max_grad = float(max(p.grad.abs().max() for p in curve.parameters()))

        if fw_cfg['stop'] == 'grad':
            if max_grad < fw_cfg['thresh']:
                stop_reason = "grad"
                break
        else:
            if loss.item() < best * (1 - fw_cfg['rtol']):
                best, stall = loss.item(), 0
            else:
                stall += 1
            if stall >= fw_cfg['patience']:
                stop_reason = "plateau"
                break

    with torch.no_grad():
        z_path = curve(time_steps.to(curve.device))
    info = {"stop_reason": stop_reason, "converged": stop_reason != "max_iter",
            "iterations": k + 1, "final_max_grad": max_grad}
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
def path_energy(vae_model, z_path):
    """Curve energy of a discrete latent path [1, T, d].

    EmbeddedManifold.curve_energy scales by T-1, so it approximates the continuous energy
    and every planner is scored on the same T points along its own path.
    """
    with torch.no_grad():
        return float(vae_model.curve_energy(z_path.reshape(1, -1, z_path.shape[-1]).clone()).sum())


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


def _stats(values, scale=1.0):
    v = np.asarray(values, dtype=np.float64) * scale
    return {"mean": float(v.mean()), "max": float(v.max()), "rms": float(np.sqrt((v ** 2).mean()))}


def _path_shape(path):
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    jerk = float(np.linalg.norm(np.diff(path, n=3, axis=0), axis=1).mean()) if path.shape[0] > 3 else 0.0
    return length, jerk


def segment_metrics(index, plan_time, replay, world_path, target_pos, quaternions,
                    start_gap_latent, end_gap_latent, end_gap_mm, energy, waypoint_dt):
    planned_length, planned_jerk = _path_shape(world_path)
    planned_speed = np.linalg.norm(np.diff(world_path, axis=0), axis=1) / waypoint_dt
    executed_length, executed_jerk = _path_shape(replay["achieved_path"])
    ik_collisions = replay["ik_collision_points"]

    return {
        "segment": index,
        "planning_time_seconds": plan_time,
        # Where the arm actually was after each waypoint's substeps, against that waypoint.
        "exec_pos_err_mm": _stats(replay["exec_err"][:, 0], 1000),
        "exec_ori_err_deg": _stats(np.degrees(replay["exec_err"][:, 1])),
        # How closely the reference IK solved each pose -- not where the arm went.
        "ik_residual_pos_mm": _stats(replay["ik_residual"][:, 0], 1000),
        "ik_residual_ori_deg": _stats(np.degrees(replay["ik_residual"][:, 1])),
        # After settling: against the end of the path the arm was given ...
        "end_err_mm": float(np.linalg.norm(replay["end_pos"] - world_path[-1]) * 1000),
        # ... and end-to-end against the original waypoint: that error PLUS the VAE's
        # encode/decode reconstruction error, since the geodesic ends at decode(encode(B))
        # rather than at B itself.
        "goal_err_mm": float(np.linalg.norm(replay["end_pos"] - target_pos) * 1000),
        # Chaining: how far this segment's start is from the true waypoint (non-zero only
        # when the previous plan missed its goal, i.e. for NODE) ...
        "start_gap_latent": start_gap_latent,
        # ... and how far the plan itself ended from its goal, in latent space and decoded
        # into the workspace. Isolates the planner's miss from IK/tracking error and the
        # VAE reconstruction error that goal_err_mm also includes. ~0 for graph/stochman.
        "end_gap_latent": end_gap_latent,
        "end_gap_mm": end_gap_mm,
        # The same yardstick for all planners: Riemannian energy of the planned latent path,
        # against the straight line between the same start and goal.
        **energy,
        "planned_path_length_m": planned_length,
        "planned_mean_jerk": planned_jerk,
        # Commanded end-effector speed: the arm replays one path point per waypoint_dt, so a
        # planner whose path speeds up (NODE's rollouts are not constant-speed) is tracked
        # with more lag there. Explains where a large exec_pos_err max comes from.
        "planned_speed_m_s": {"median": float(np.median(planned_speed)), "max": float(planned_speed.max())},
        "executed_path_length_m": executed_length,
        "executed_mean_jerk": executed_jerk,
        # Large values explain a large *max* tracking error: see quaternion_steps_deg().
        "max_quat_step_deg": float(quaternion_steps_deg(quaternions).max())
        if quaternions is not None and len(quaternions) > 1 else 0.0,
        # Physics steps in which two robot bodies touched, and which ones.
        "self_collision_steps": replay["self_collision_steps"],
        "self_contact_pairs": replay["self_contact_pairs"],
        # Waypoints whose final IK solution still collided; IK retries from other seeds;
        # retries that found no clean solution; largest joint change between two commands.
        "ik_collision_points": {"count": len(ik_collisions),
                                "first_index": ik_collisions[0] if ik_collisions else None},
        "ik_retries": replay["ik_retries"],
        "ik_unresolved": replay["ik_unresolved"],
        "max_cmd_jump_deg": float(np.degrees(replay["max_cmd_jump_rad"])),
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
                        help="LASA shape, as defined in node_config.yaml (e.g. N, Angle, Leaf_1, Sine)")
    parser.add_argument('--beta_scale', type=str, required=True,
                        help="beta scale of the VAE (and NODE) to use (e.g. 5, 10)")
    parser.add_argument('--node_variant', type=str, default=None,
                        help="NODE only: run_suffix of the NODE to load (e.g. energy_imitation, "
                             "energy_only); default: the shape's run_suffix in node_config.yaml")
    parser.add_argument('--no_viewer', action='store_true',
                        help="Run headless (metrics only, no live viewer)")
    parser.add_argument('--graph_size', type=int, default=None,
                        help="Graph framework only: override the node count on the longer grid "
                             "axis (the cell size follows); small values give a much faster but "
                             "coarser manifold, useful for smoke tests")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help="Defaults to cpu: DiscretizedManifold builds its grid and "
                             "decoded_positions as CPU tensors and calls .numpy() on "
                             "intermediates, so a CUDA VAE trips a device mismatch. The "
                             "VAE is small and MuJoCo is CPU-bound, so there is little "
                             "to gain here anyway.")
    parser.add_argument('--seed', type=int, default=4)
    args = parser.parse_args()

    shapes = lasa_shapes()
    if args.shape not in shapes:
        parser.error(f"--shape {args.shape} is not defined in {NODE_CONFIG_PATH} (choose from {shapes})")
    if args.node_variant and args.framework != 'node':
        print("  [NOTE] --node_variant ignored: it only applies to the node framework")
    beta = beta_label(args.beta_scale)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open('config_files/simulation_config.yaml', 'r') as f:
        sim_cfg = yaml.safe_load(f)
    data_cfg = sim_cfg['datasets'][args.dataset]
    mj_cfg = sim_cfg['mujoco']
    ik_cfg = sim_cfg['ik']

    device = args.device

    # ---- 1. VAE ----
    print(f"\n--- Loading VAE ({args.shape}-Shape, beta_scale {beta}) ---")
    vae_model, training_points, vae_path = load_vae(args, data_cfg, device)

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
    replay = sim_utils.PandaReplay(mj_cfg, ik_cfg, seed=args.seed)

    mj_viewer = None
    if not args.no_viewer:
        from mujoco import viewer as mujoco_viewer
        mj_viewer = mujoco_viewer.launch_passive(replay.model, replay.data)
        sim_utils.add_labeled_markers(mj_viewer, marker_points)
        replay.viewer = mj_viewer

    # ---- 4. Planner ----
    print(f"\n--- Setting up {args.framework} planner ---")
    fw_cfg = copy.deepcopy(sim_cfg['frameworks'][args.framework])
    if args.graph_size is not None and args.framework != 'graph':
        print("  [NOTE] --graph_size ignored: it only applies to the graph framework")
    time_steps = torch.linspace(0, 1, data_cfg['time_steps'])
    node_path = node_variant = grid_info = None

    t0 = time.perf_counter()
    if args.framework == 'graph':
        with torch.no_grad():
            z_train = vae_model.encode(training_points.to(device), train_rbf=True)[1]
        grid, grid_info = latent_grid(fw_cfg, args.dataset, torch.cat([z_train, z_waypoints]),
                                      args.graph_size)
        warn_outside_grid(z_waypoints, grid_info)
        discrete_model = build_discrete_manifold(vae_model, grid.to(device), grid_info)
        plan = lambda p0, p1: plan_geodesic(vae_model, discrete_model, p0, p1, time_steps)
    elif args.framework == 'node':
        node_model, node_path, node_variant = load_node(args, device)
        plan = lambda p0, p1: plan_node(node_model, p0, p1, time_steps)
    else:  # stochman: optimises each curve from scratch, nothing to prepare
        plan = lambda p0, p1: plan_stochman(vae_model, p0, p1, time_steps, fw_cfg)
    setup_time = time.perf_counter() - t0
    print(f"[Timer] planner setup: {setup_time:.4f} seconds (not part of planning time)")

    # One untimed query first, as benchmark/timing.py's WARMUP_QUERIES: the first call pays
    # one-off costs (lazy torch/networkx initialisation; ~0.5 s for graph against ~0.03 s per
    # later query) that would otherwise be booked as segment 0's planning time. Stochman's
    # warm-up is capped at a few iterations -- that already initialises everything it uses.
    if args.framework == 'stochman':
        plan_stochman(vae_model, z_waypoints[0], z_waypoints[1], time_steps, dict(fw_cfg, max_iter=5))
    else:
        plan(z_waypoints[0], z_waypoints[1])

    results = []
    approach_report = None

    try:
        # Every planner's path starts at decode(z_A), so the approach is the same for all.
        print("\n--- Approaching waypoint A ---")
        pos_a, quat_a = decode_path(vae_model, z_waypoints[:1].unsqueeze(0))
        world_a = sim_utils.to_workspace(np.atleast_2d(pos_a), xy_center, xy_scale,
                                         mj_cfg['pos_xy_scale'], offset)[0]
        approach_report = replay.approach(world_a, np.atleast_2d(quat_a)[0])

        # Chained: each segment starts where the previous plan ended in latent space, not at
        # the true waypoint, so the arm never jumps between segments. The goal is always the
        # true next waypoint. Only NODE can end off-goal; see the module docstring.
        z_start = z_waypoints[0]
        for i in range(z_waypoints.shape[0] - 1):
            print(f"\n--- Processing segment {i} ({chr(65+i)} -> {chr(66+i)}) ---")
            z_goal = z_waypoints[i + 1]

            with QueryTimer(device) as timer:
                z_path, plan_info = plan(z_start, z_goal)
            plan_time = timer.elapsed
            print(f"[Timer] {args.framework} planning: {plan_time:.4f} seconds")
            if plan_info.get("stop_reason") == "max_iter":
                print(f"  [NOTE] stochman stopped at max_iter={plan_info['iterations']} without "
                      f"converging (max |grad| {plan_info['final_max_grad']:.2e})")

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

            ts = time_steps.to(z_start.device).reshape(-1, 1)
            z_line = (z_start + ts * (z_goal - z_start)).unsqueeze(0)
            energy = {
                "latent_curve_energy": path_energy(vae_model, z_path),
                "straight_line_energy": path_energy(vae_model, z_line),
                "max_dev_from_line_latent": float(torch.norm(z_path - z_line, dim=-1).max()),
            }

            t0 = time.perf_counter()
            replay_out = replay.execute(world_path, quaternions)
            replay_time = time.perf_counter() - t0
            print(f"[Timer] mujoco replay: {replay_time:.4f} seconds")

            metrics = segment_metrics(i, plan_time, replay_out, world_path,
                                      marker_points[i + 1, :3], quaternions,
                                      start_gap_latent, end_gap_latent, end_gap_mm, energy,
                                      waypoint_dt=mj_cfg['waypoint_substeps'] * mj_cfg['timestep'])
            metrics["replay_time_seconds"] = replay_time
            metrics.update(plan_info)

            if metrics["max_quat_step_deg"] > 10.0:
                print(f"  [WARNING] orientation jumps {metrics['max_quat_step_deg']:.1f} deg "
                      f"between consecutive points -- the geodesic crosses a region the "
                      f"orientation decoder does not model; IK cannot follow it.")
            if end_gap_mm > 5.0:
                print(f"  [WARNING] plan ends {end_gap_mm:.1f} mm short of waypoint "
                      f"{chr(66+i)}; the next segment starts from there.")
            if args.framework == 'stochman' and \
                    energy["latent_curve_energy"] > 0.9 * energy["straight_line_energy"]:
                print(f"  [WARNING] stochman barely moved off the straight line (energy "
                      f"{energy['latent_curve_energy']:.3g} vs {energy['straight_line_energy']:.3g})")
            if metrics["self_collision_steps"]:
                print(f"  [WARNING] self-collision during {metrics['self_collision_steps']} physics "
                      f"steps: {metrics['self_contact_pairs']}")
            if metrics["ik_unresolved"]:
                print(f"  [WARNING] {metrics['ik_unresolved']} waypoint(s) had no collision-free "
                      f"IK solution within reseed_tol_pos")
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
    results_dir = os.path.join(sim_cfg['results_base_dir'], f"{args.shape}-Shape_beta_scale_{beta}")
    os.makedirs(results_dir, exist_ok=True)

    report = {
        "framework": args.framework,
        "dataset": args.dataset,
        "shape": args.shape,
        "beta_scale": beta,
        "node_variant": node_variant,
        "num_waypoints": int(data_cfg['num_waypoints']),
        "time_steps": int(data_cfg['time_steps']),
        "seed": args.seed,
        # The RBF bandwidth is not part of the checkpoint; load_pretrained_vae restores the
        # training-time value. It feeds embed(), the metric, and therefore which path the
        # graph/stochman planners return (NODE never reads the metric). Recorded so a set of
        # numbers can be tied back to the metric that produced them.
        "rbf_beta": float(vae_model.dec_std_qua[0].beta.flatten()[0]),
        "vae_checkpoint": vae_path,
        "node_checkpoint": node_path,
        "framework_config": fw_cfg,
        "graph_grid": grid_info,
        "simulation": {"mujoco_version": mujoco.__version__, "mujoco": mj_cfg, "ik": ik_cfg},
        "approach": approach_report,
        # Graph: manifold construction. NODE: checkpoint load. Stochman: none.
        # Logged for information only -- not part of the planning time.
        "setup_time_seconds": setup_time,
        "warmup_queries": 1,
        "total_planning_time_seconds": sum(r["planning_time_seconds"] for r in results),
        "segments": results,
    }

    file_tag = f"{args.framework}_{node_variant}" if node_variant else args.framework
    json_path = os.path.join(results_dir, f"sim_{file_tag}_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=4, cls=NumpyEncoder)

    print("\n" + "=" * 50)
    print(f"Metrics saved to: {json_path}")
    plan_times = [r["planning_time_seconds"] for r in results]
    print(f"Planning: {sum(plan_times):.4f}s total over {len(plan_times)} segments "
          f"(mean {np.mean(plan_times):.4f}s)")
    exec_max = [r["exec_pos_err_mm"]["max"] for r in results]
    print(f"Executed pos error: max {max(exec_max):.2f} mm over all segments; "
          f"self-collision steps: {sum(r['self_collision_steps'] for r in results)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
