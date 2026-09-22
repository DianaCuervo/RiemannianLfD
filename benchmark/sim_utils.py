"""MuJoCo replay helpers for simulation_benchmark.py.

Ports the Panda IK/replay code from GeodesicMotionSkills' toy_example.py and adds the
small runtime adapter that lets our plain-nn.Module VAE drive DiscretizedManifold.
"""
import types
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.io import loadmat

from stochman.manifold import EmbeddedManifold
from GeodesicMotionSkills.Experiments.Utils.environment import Environment


# ==========================================
# VAE ADAPTER
# ==========================================
def as_graph_manifold(vae):
    """Attach the three attributes DiscretizedManifold expects, without touching vae_model.py.

    vae_model.py's VAE subclasses plain nn.Module, so it has embed() but none of the
    Manifold interface built on top of it. Rather than change that class, we bind what
    the graph planner actually reads onto the instance -- the same trick
    load_pretrained_vae already uses when it sets vae.obstacle_input_space from outside.

        curve_length  weights every edge of the graph (discretized_manifold.py:77-98).
                      EmbeddedManifold's implementation only calls self.embed().
        env           read for obstacles/via-points (discretized_manifold.py:186-187).
        time_step     passed straight through as `graph_id`, which is only used by a
                      commented-out draw_graph() call -- carried so our call site reads
                      identically to benchmark/eval_graph.py.
    """
    if not hasattr(vae, "curve_length"):
        vae.curve_length = types.MethodType(EmbeddedManifold.curve_length, vae)
    if not hasattr(vae, "env"):
        vae.env = Environment()
    if not hasattr(vae, "time_step"):
        vae.time_step = 0
    return vae


# ==========================================
# LASA DEMONSTRATIONS
# ==========================================
def load_lasa_demos(origin_dir, origin_file, trajectory_number):
    """Load LASA demos and return them normalized, plus the constants needed to undo it.

    Mirrors node/data/preprocessing.py's get_demonstrations_paths_newVAE + normalize_newVAE
    exactly -- one isotropic normalization shared across all demos -- but also hands back
    xy_center/xy_scale, which preprocessing computes locally and discards. Without them we
    cannot map a decoded trajectory back to real LASA units for the robot.
    """
    demoUQ = loadmat(f"{origin_dir}/{origin_file}")["demoUQ"]

    raw_positions = [demoUQ[0, i]["tsPos"][0, 0].T for i in range(trajectory_number)]
    raw_quats = [demoUQ[0, i]["quat"][0, 0].T for i in range(trajectory_number)]

    all_xy = np.vstack([pos[:, 0:2] for pos in raw_positions])
    xy_min, xy_max = all_xy.min(axis=0), all_xy.max(axis=0)
    xy_center = (xy_min + xy_max) / 2
    xy_scale = (xy_max - xy_min).max() / 2  # single scalar -> isotropic scaling

    demos = []
    for pos, quat in zip(raw_positions, raw_quats):
        xy_norm = (pos[:, 0:2] - xy_center) / xy_scale
        demos.append(np.hstack([xy_norm, pos[:, 2:3], quat]))

    return demos, xy_center, xy_scale


def to_workspace(pos_norm, xy_center, xy_scale, pos_xy_scale, offset=None):
    """Map normalized VAE positions into Panda workspace coordinates.

    This is the ONLY place the normalization/scaling/offset chain is applied. toy_example.py
    split it across two call sites -- the A-E markers were de-normalized while the executed
    trajectory was not -- so the spheres never sat where the arm actually went.

    The inverse of normalize_newVAE is `xy * xy_scale + xy_center`; note that xy_scale is a
    single isotropic scalar, so the per-axis `(x+1)/2*(max-min)+min` form used for the old
    markers was only correct on the wider of the two axes.

    z is passed through unnormalized and unscaled, matching preprocessing (which normalizes
    xy only). For LASA this is moot -- tsPos z is identically 0 -- so the workspace offset
    alone sets the working height.
    """
    pos_norm = np.asarray(pos_norm, dtype=np.float64)
    world = np.empty_like(pos_norm)
    world[:, :2] = (pos_norm[:, :2] * xy_scale + xy_center) * pos_xy_scale
    world[:, 2] = pos_norm[:, 2]
    if offset is not None:
        world = world + offset
    return world


def compute_workspace_offset(pos, target):
    """Shift a set of workspace points so their centroid lands on `target`."""
    target = np.asarray(target, dtype=np.float64)
    offset = target - pos.mean(axis=0)
    reach = np.linalg.norm(pos + offset, axis=1)
    print(f"  Workspace offset applied: {offset.round(3)} m")
    print(f"  Shifted reach: [{reach.min():.3f}, {reach.max():.3f}] m (Panda max ~0.855 m)")
    if reach.max() > 0.85:
        print("  [WARNING] Some waypoints may be near or beyond the workspace boundary")
    return offset


# ==========================================
# MUJOCO MODEL
# ==========================================
def build_mujoco_model(cfg):
    xml_path = Path(cfg["xml_path"]).resolve()
    print(f"Loading MuJoCo model: {xml_path}")
    xml_text = xml_path.read_text()

    # The XML declares a relative meshdir, which only resolves when MuJoCo is given the
    # file path; we load from a string, so point it at the absolute assets directory.
    assets_dir = (xml_path.parent / "assets").resolve()
    xml_text = xml_text.replace('meshdir="assets"', f'meshdir="{assets_dir.as_posix()}"')

    mj_model = mujoco.MjModel.from_xml_string(xml_text)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = cfg["timestep"]

    site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, cfg["ee_site"])
    if site_id < 0:
        raise RuntimeError(f"EE site '{cfg['ee_site']}' not found in {xml_path}")

    jnt_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in cfg["joints"]]
    if any(j < 0 for j in jnt_ids):
        missing = [n for n, j in zip(cfg["joints"], jnt_ids) if j < 0]
        raise RuntimeError(f"Joints not found in {xml_path}: {missing}")

    qpos_ids = np.array([mj_model.jnt_qposadr[j] for j in jnt_ids])
    dof_ids = np.array([mj_model.jnt_dofadr[j] for j in jnt_ids])

    return mj_model, mj_data, site_id, qpos_ids, dof_ids, jnt_ids


def reset_to_home(mj_model, mj_data):
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    else:
        mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)


# ==========================================
# INVERSE KINEMATICS
# ==========================================
def _clamp_joints(mj_model, mj_data, qpos_ids, jnt_ids):
    for i, jid in enumerate(jnt_ids):
        lo, hi = mj_model.jnt_range[jid]
        mj_data.qpos[qpos_ids[i]] = np.clip(mj_data.qpos[qpos_ids[i]], lo, hi)


def solve_ik_position(mj_model, mj_data, site_id, qpos_ids, dof_ids, jnt_ids, target_pos, ik):
    """Position-only IK via damped least squares."""
    nv = mj_model.nv
    for _ in range(ik["max_iter"]):
        mujoco.mj_forward(mj_model, mj_data)
        dp = target_pos - mj_data.site_xpos[site_id]
        if np.linalg.norm(dp) < ik["tol_pos"]:
            break
        jacp = np.zeros((3, nv))
        mujoco.mj_jacSite(mj_model, mj_data, jacp, None, site_id)
        J = jacp[:, dof_ids]
        dq = J.T @ np.linalg.solve(J @ J.T + ik["lam"] * np.eye(3), dp)
        mj_data.qpos[qpos_ids] += ik["step"] * dq
        _clamp_joints(mj_model, mj_data, qpos_ids, jnt_ids)
    mujoco.mj_forward(mj_model, mj_data)
    return np.linalg.norm(target_pos - mj_data.site_xpos[site_id])


def solve_ik_full(mj_model, mj_data, site_id, qpos_ids, dof_ids, jnt_ids, target_pos, target_quat, ik):
    """6-DoF position + orientation IK via damped least squares."""
    nv = mj_model.nv
    ep, eo = 1.0, 1.0
    for _ in range(ik["max_iter"]):
        mujoco.mj_forward(mj_model, mj_data)
        cur_pos = mj_data.site_xpos[site_id].copy()
        cur_mat = mj_data.site_xmat[site_id].reshape(3, 3).copy()

        dp = target_pos - cur_pos
        ep = np.linalg.norm(dp)

        tm = np.zeros(9)
        mujoco.mju_quat2Mat(tm, target_quat)
        R_err = tm.reshape(3, 3) @ cur_mat.T
        angle = float(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0)))
        eo = abs(angle)
        if eo < 1e-9:
            do = np.zeros(3)
        else:
            s = np.sin(angle)
            do = angle / (2 * s) * np.array([
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1],
            ])

        if ep < ik["tol_pos"] and eo < ik["tol_ori"]:
            break

        err6 = np.concatenate([dp, do])
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacSite(mj_model, mj_data, jacp, jacr, site_id)
        J = np.vstack([jacp, jacr])[:, dof_ids]
        dq = J.T @ np.linalg.solve(J @ J.T + ik["lam"] * np.eye(6), err6)
        mj_data.qpos[qpos_ids] += ik["step"] * dq
        _clamp_joints(mj_model, mj_data, qpos_ids, jnt_ids)

    mujoco.mj_forward(mj_model, mj_data)
    ep = np.linalg.norm(target_pos - mj_data.site_xpos[site_id])
    return ep, eo


# ==========================================
# TRAJECTORY EXECUTION
# ==========================================
def execute_trajectory_in_mujoco(mj_model, mj_data, site_id, qpos_ids, dof_ids, jnt_ids,
                                 traj_pos, traj_quat, ik, substeps, viewer=None,
                                 settle_steps=0):
    """Drive the arm along a trajectory already expressed in workspace coordinates.

    Unlike toy_example.py's version this applies no scaling or offset of its own -- callers
    hand it world-frame targets via to_workspace() -- and it does not mutate its inputs.
    """
    traj_pos = np.atleast_2d(np.asarray(traj_pos, dtype=np.float64).squeeze())
    if traj_quat is not None:
        traj_quat = np.atleast_2d(np.asarray(traj_quat, dtype=np.float64).squeeze())

    ik_errors = []
    for i in range(traj_pos.shape[0]):
        qpos_start = mj_data.qpos[qpos_ids].copy()  # where the arm actually is right now

        if traj_quat is not None:
            ep, eo = solve_ik_full(mj_model, mj_data, site_id, qpos_ids, dof_ids, jnt_ids,
                                   traj_pos[i], traj_quat[i], ik)
        else:
            ep = solve_ik_position(mj_model, mj_data, site_id, qpos_ids, dof_ids, jnt_ids,
                                   traj_pos[i], ik)
            eo = 0.0
        ik_errors.append((ep, eo))

        qpos_target = mj_data.qpos[qpos_ids].copy()  # where the IK solver converged to

        # restore the real current pose before physically stepping toward the target
        mj_data.qpos[qpos_ids] = qpos_start
        mujoco.mj_forward(mj_model, mj_data)

        for sub in range(1, substeps + 1):
            alpha = sub / substeps
            mj_data.ctrl[:7] = (1 - alpha) * qpos_start + alpha * qpos_target
            mujoco.mj_step(mj_model, mj_data)
            if viewer is not None:
                viewer.sync()

    # The position actuators lag their setpoint, so after the last waypoint the arm is
    # still short of it. Hold the final command for a while to let it converge, otherwise
    # `achieved_pos` measures controller lag rather than where the trajectory ended.
    for _ in range(settle_steps):
        mujoco.mj_step(mj_model, mj_data)
        if viewer is not None:
            viewer.sync()

    achieved_pos = mj_data.site_xpos[site_id].copy()
    achieved_mat = mj_data.site_xmat[site_id].reshape(3, 3).copy()
    achieved_quat = np.zeros(4)
    mujoco.mju_mat2Quat(achieved_quat, achieved_mat.flatten())

    ep_mm = np.array([e[0] for e in ik_errors]) * 1000
    eo_deg = np.degrees([e[1] for e in ik_errors])
    print(f"  MuJoCo replay: pos_err mean={ep_mm.mean():.2f}mm max={ep_mm.max():.2f}mm  "
          f"ori_err mean={eo_deg.mean():.1f}deg max={eo_deg.max():.1f}deg")

    return ik_errors, achieved_pos, achieved_quat


# ==========================================
# VIEWER MARKERS
# ==========================================
def add_labeled_markers(mj_viewer, marker_points, labels=None, show_arrows=True):
    """Add A-E sphere markers, their text labels, and direction arrows to the viewer."""
    if mj_viewer is None:
        return

    if labels is None:
        labels = [chr(ord('A') + i) for i in range(marker_points.shape[0])]

    mj_viewer.user_scn.ngeom = 0  # avoid duplicates on re-run
    print(f"Adding {len(marker_points)} labeled markers to MuJoCo viewer...")

    for pt, label in zip(marker_points, labels):
        geom = mj_viewer.user_scn.geoms[mj_viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.012, 0, 0],
            pos=pt[:3],
            mat=np.eye(3).flatten(),
            rgba=[1.0, 0.2, 0.2, 1.0],
        )
        geom.label = label
        # Tag as a SITE so the label actually renders -- see the mjLABEL_SITE call below.
        geom.objtype = mujoco.mjtObj.mjOBJ_SITE
        mj_viewer.user_scn.ngeom += 1

    if show_arrows:
        for i in range(marker_points.shape[0] - 1):
            p0, p1 = marker_points[i, :3], marker_points[i + 1, :3]
            geom = mj_viewer.user_scn.geoms[mj_viewer.user_scn.ngeom]
            mujoco.mjv_initGeom(
                geom,
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=[0.004, 0.004, 0.0],
                pos=p0,
                mat=np.eye(3).flatten(),
                rgba=[0.2, 0.6, 1.0, 0.9],
            )
            geom.label = ""
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.004, p0, p1)
            mj_viewer.user_scn.ngeom += 1

    # Label SITES only, otherwise every robot geom picks up a text label too.
    mj_viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE
    mj_viewer.sync()
