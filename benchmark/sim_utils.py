"""MuJoCo replay helpers for simulation_benchmark.py.

Ports the Panda IK/replay code from GeodesicMotionSkills' toy_example.py and adds the
small runtime adapter that lets our VAE drive DiscretizedManifold and stochman's energy
minimisation.
"""
import types
from pathlib import Path

import mujoco
import numpy as np
import torch

from stochman.manifold import EmbeddedManifold
from GeodesicMotionSkills.Experiments.Utils.environment import Environment
from benchmark.data_utils import load_lasa_demos  # noqa: F401 (re-exported for simulation_benchmark.py)


# ==========================================
# VAE ADAPTER
# ==========================================
def as_embedded_manifold(vae):
    """Make sure the VAE carries the Manifold attributes the planners expect.

    vae_model.py's VAE now inherits EmbeddedManifold and sets env/time_step itself, so this
    normally changes nothing. It stays as a guard for VAE classes that subclass plain
    nn.Module (vae/evaluation/vae_test_with_graph.py imports it too): we bind what the graph
    and stochman planners actually read onto the instance, without touching the class.

        curve_length  weights every edge of the graph (discretized_manifold.py:77-98).
                      EmbeddedManifold's implementation only calls self.embed().
        curve_energy  the objective stochman's geodesic energy minimisation backpropagates
                      through; likewise only calls self.embed().
        env           read for obstacles/via-points (discretized_manifold.py:186-187).
        time_step     passed straight through as `graph_id`, which is only used by a
                      commented-out draw_graph() call -- carried so our call site reads
                      identically to benchmark/eval_graph.py.
    """
    if not hasattr(vae, "curve_length"):
        vae.curve_length = types.MethodType(EmbeddedManifold.curve_length, vae)
    if not hasattr(vae, "curve_energy"):
        vae.curve_energy = types.MethodType(EmbeddedManifold.curve_energy, vae)
    if not hasattr(vae, "env"):
        vae.env = Environment()
    if not hasattr(vae, "time_step"):
        vae.time_step = 0
    return vae


# ==========================================
# LASA DEMONSTRATIONS
# ==========================================
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

    # The Panda's position actuators have no gravity term, so without compensation the arm
    # sags ~0.0085 rad below every command. gravcomp must be set before compiling: writing
    # mj_model.body_gravcomp afterwards has no effect, because ngravcomp is fixed at compile.
    spec = mujoco.MjSpec.from_string(xml_text)
    if cfg.get("gravity_compensation", False):
        for body in spec.bodies:
            body.gravcomp = 1.0
    mj_model = spec.compile()
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


def joint_actuators(mj_model, jnt_ids):
    """The actuator driving each joint, in joint order (rather than assuming ctrl[:7])."""
    act_ids = []
    for jid in jnt_ids:
        matches = [a for a in range(mj_model.nu)
                   if mj_model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
                   and mj_model.actuator_trnid[a, 0] == jid]
        if not matches:
            raise RuntimeError(f"No actuator drives joint {mj_model.joint(jid).name}")
        act_ids.append(matches[0])
    return np.array(act_ids)


def reset_to_home(mj_model, mj_data):
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    else:
        mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)


def self_contact_count(mj_model, mj_data):
    """Contacts between two robot bodies (the world body, id 0, is excluded).

    Valid after mj_forward / mj_step, which run collision detection.
    """
    ncon = mj_data.ncon
    if ncon == 0:
        return 0
    b1 = mj_model.geom_bodyid[mj_data.contact.geom1[:ncon]]
    b2 = mj_model.geom_bodyid[mj_data.contact.geom2[:ncon]]
    return int(np.count_nonzero((b1 != 0) & (b2 != 0)))


def self_contact_pairs(mj_model, mj_data):
    """Names of the robot body pairs currently in contact, for warnings."""
    pairs = set()
    for c in mj_data.contact[:mj_data.ncon]:
        b1, b2 = mj_model.geom_bodyid[c.geom1], mj_model.geom_bodyid[c.geom2]
        if b1 != 0 and b2 != 0:
            pairs.add(tuple(sorted((mj_model.body(b1).name, mj_model.body(b2).name))))
    return pairs


# ==========================================
# INVERSE KINEMATICS
# ==========================================
def _clamp_joints(mj_model, mj_data, qpos_ids, jnt_ids):
    for i, jid in enumerate(jnt_ids):
        lo, hi = mj_model.jnt_range[jid]
        mj_data.qpos[qpos_ids[i]] = np.clip(mj_data.qpos[qpos_ids[i]], lo, hi)


def quat_to_mat(quat):
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)


def _orientation_error(target_mat, cur_mat):
    """Rotation vector taking cur_mat to target_mat (world frame), and its angle in rad."""
    R_err = target_mat @ cur_mat.T
    angle = float(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0)))
    if angle < 1e-9:
        return np.zeros(3), angle
    return angle / (2 * np.sin(angle)) * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ]), angle


def solve_ik_full(mj_model, mj_data, site_id, qpos_ids, dof_ids, jnt_ids, target_pos, target_quat, ik,
                  q_posture=None):
    """6-DoF position + orientation IK via damped least squares.

    With q_posture and ik['posture_gain'] > 0, a null-space term pulls the joints towards
    q_posture through the arm's redundant DoF only, so the end-effector target is unaffected.
    Without it the local solver drifts into wrist poses that self-collide (N-Shape: the hand
    hits link5). Returns the final position error (m) and orientation error (rad).
    """
    gain = ik.get("posture_gain", 0.0) if q_posture is not None else 0.0
    target_mat = quat_to_mat(target_quat)
    n = len(dof_ids)
    for _ in range(ik["max_iter"]):
        mujoco.mj_forward(mj_model, mj_data)
        dp = target_pos - mj_data.site_xpos[site_id]
        do, eo = _orientation_error(target_mat, mj_data.site_xmat[site_id].reshape(3, 3))
        if np.linalg.norm(dp) < ik["tol_pos"] and eo < ik["tol_ori"]:
            break

        jacp = np.zeros((3, mj_model.nv))
        jacr = np.zeros((3, mj_model.nv))
        mujoco.mj_jacSite(mj_model, mj_data, jacp, jacr, site_id)
        J = np.vstack([jacp, jacr])[:, dof_ids]
        J_pinv = J.T @ np.linalg.inv(J @ J.T + ik["lam"] * np.eye(6))
        dq = J_pinv @ np.concatenate([dp, do])
        if gain:
            dq += (np.eye(n) - J_pinv @ J) @ (gain * (q_posture - mj_data.qpos[qpos_ids]))
        mj_data.qpos[qpos_ids] += ik["step"] * dq
        _clamp_joints(mj_model, mj_data, qpos_ids, jnt_ids)

    mujoco.mj_forward(mj_model, mj_data)
    ep = np.linalg.norm(target_pos - mj_data.site_xpos[site_id])
    _, eo = _orientation_error(target_mat, mj_data.site_xmat[site_id].reshape(3, 3))
    return ep, eo


# ==========================================
# TRAJECTORY EXECUTION
# ==========================================
class PandaReplay:
    """Drive the simulated Panda through world-frame pose trajectories.

    IK runs on its own MjData (`ref`), a kinematic reference warm-started from the previous
    solution; the physical arm (`data`) only receives joint commands, interpolated from the
    previous command to the new one. toy_example.py instead solved IK from the lagging
    physical pose and restarted every command from it, so actuator lag fed back into the
    commands and the arm never caught up.

    Every reference IK solution is checked for self-collision. If it collides or misses
    `reseed_tol_pos`, IK is retried from other seeds and the collision-free solution closest
    to the previous command is kept (ik.reseed_on_collision).
    """

    SYNC_EVERY = 20  # physics steps between viewer syncs (50 Hz of simulated time at 1 ms)

    def __init__(self, mj_cfg, ik_cfg, seed=0):
        (self.model, self.data, self.site_id, self.qpos_ids,
         self.dof_ids, self.jnt_ids) = build_mujoco_model(mj_cfg)
        self.act_ids = joint_actuators(self.model, self.jnt_ids)
        reset_to_home(self.model, self.data)

        self.ref = mujoco.MjData(self.model)
        self.ref.qpos[:] = self.data.qpos
        mujoco.mj_forward(self.model, self.ref)

        self.q_home = self.data.qpos[self.qpos_ids].copy()
        self.cmd = self.q_home.copy()  # last joint command sent to the actuators
        self.lo, self.hi = self.model.jnt_range[self.jnt_ids].T

        self.ik = ik_cfg
        self.substeps = mj_cfg["waypoint_substeps"]
        self.settle_steps = mj_cfg.get("settle_steps", 0)
        self.approach_steps = mj_cfg.get("approach_steps", 0)
        self.approach_ik_max_iter = mj_cfg.get("approach_ik_max_iter", ik_cfg["max_iter"])
        self.rng = np.random.default_rng(seed)
        self.viewer = None
        self._steps = 0

    # ---- low level ----
    def _step(self, n):
        """Advance physics n steps; returns how many of them had a robot self-contact."""
        hits = 0
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            hits += self_contact_count(self.model, self.data) > 0
            self._steps += 1
            if self.viewer is not None and self._steps % self.SYNC_EVERY == 0:
                self.viewer.sync()
        return int(hits)

    def _ik(self, target_pos, target_quat, max_iter):
        ik = dict(self.ik, max_iter=max_iter)
        return solve_ik_full(self.model, self.ref, self.site_id, self.qpos_ids, self.dof_ids,
                             self.jnt_ids, target_pos, target_quat, ik, q_posture=self.q_home)

    def _solve(self, target_pos, target_quat, max_iter=None):
        """Reference IK for one pose. Returns (ep, eo, retried, in_collision)."""
        q_prev = self.ref.qpos[self.qpos_ids].copy()
        ep, eo = self._ik(target_pos, target_quat, max_iter or self.ik["max_iter"])
        in_collision = self_contact_count(self.model, self.ref) > 0
        tol = self.ik.get("reseed_tol_pos", np.inf)
        if not self.ik.get("reseed_on_collision", False) or (not in_collision and ep < tol):
            return ep, eo, False, in_collision

        q_first = self.ref.qpos[self.qpos_ids].copy()
        seeds = [self.q_home, 0.5 * (self.q_home + q_prev), q_prev + self.rng.normal(0.0, 0.3, len(q_prev))]
        best = None
        for seed in seeds:
            self.ref.qpos[self.qpos_ids] = np.clip(seed, self.lo, self.hi)
            e = self._ik(target_pos, target_quat, self.ik["reseed_max_iter"])
            if self_contact_count(self.model, self.ref) == 0 and e[0] < tol:
                dist = np.linalg.norm(self.ref.qpos[self.qpos_ids] - q_prev)
                if best is None or dist < best[0]:
                    best = (dist, self.ref.qpos[self.qpos_ids].copy(), e)

        self.ref.qpos[self.qpos_ids] = q_first if best is None else best[1]
        mujoco.mj_forward(self.model, self.ref)
        if best is None:
            return ep, eo, True, in_collision
        return best[2][0], best[2][1], True, False

    def _pose_error(self, target_pos, target_quat):
        ep = np.linalg.norm(self.data.site_xpos[self.site_id] - target_pos)
        _, eo = _orientation_error(quat_to_mat(target_quat), self.data.site_xmat[self.site_id].reshape(3, 3))
        return ep, eo

    # ---- public ----
    def approach(self, target_pos, target_quat):
        """Move from the current pose to the first pose of the replay, then settle.

        Without this, segment 0 started from the home pose ~0.29 m away and spent its first
        part catching up. Not part of any segment's metrics.
        """
        ep_ik, eo_ik, _, in_collision = self._solve(target_pos, target_quat, self.approach_ik_max_iter)
        start, goal = self.cmd.copy(), self.ref.qpos[self.qpos_ids].copy()
        collisions = 0
        for k in range(1, self.approach_steps + 1):
            a = k / self.approach_steps
            a = a * a * (3 - 2 * a)  # smoothstep: zero joint velocity at both ends
            self.data.ctrl[self.act_ids] = (1 - a) * start + a * goal
            collisions += self._step(1)
        collisions += self._step(self.settle_steps)
        self.cmd = goal

        ep, eo = self._pose_error(target_pos, target_quat)
        print(f"  Approach: pos_err={ep * 1000:.2f}mm ori_err={np.degrees(eo):.2f}deg")
        return {"pos_err_mm": float(ep * 1000), "ori_err_deg": float(np.degrees(eo)),
                "ik_residual_pos_mm": float(ep_ik * 1000), "ik_residual_ori_deg": float(np.degrees(eo_ik)),
                "ik_in_collision": bool(in_collision), "self_collision_steps": collisions}

    def execute(self, traj_pos, traj_quat):
        """Replay one trajectory already expressed in workspace coordinates.

        The command for each waypoint is ramped over `waypoint_substeps` physics steps, then
        the arm's actual pose is recorded against that waypoint. After the last waypoint the
        command is held for `settle_steps`. Inputs are not mutated.
        """
        traj_pos = np.atleast_2d(np.asarray(traj_pos, dtype=np.float64).squeeze())
        traj_quat = np.atleast_2d(np.asarray(traj_quat, dtype=np.float64).squeeze())
        n = traj_pos.shape[0]

        ik_residual = np.zeros((n, 2))
        exec_err = np.zeros((n, 2))
        achieved = np.zeros((n, 3))
        ik_collision_points, contact_pairs = [], set()
        retries = unresolved = collisions = 0
        max_jump = 0.0
        tol = self.ik.get("reseed_tol_pos", np.inf)

        for i in range(n):
            ep, eo, retried, in_collision = self._solve(traj_pos[i], traj_quat[i])
            ik_residual[i] = ep, eo
            retries += retried
            unresolved += retried and (in_collision or ep >= tol)
            if in_collision:
                ik_collision_points.append(i)

            q_target = self.ref.qpos[self.qpos_ids].copy()
            max_jump = max(max_jump, float(np.abs(q_target - self.cmd).max()))
            for sub in range(1, self.substeps + 1):
                a = sub / self.substeps
                self.data.ctrl[self.act_ids] = (1 - a) * self.cmd + a * q_target
                hit = self._step(1)
                if hit and len(contact_pairs) < 10:
                    contact_pairs |= self_contact_pairs(self.model, self.data)
                collisions += hit
            self.cmd = q_target

            achieved[i] = self.data.site_xpos[self.site_id]
            exec_err[i] = self._pose_error(traj_pos[i], traj_quat[i])

        # The actuators lag their setpoint; hold the last command so the end pose measures
        # where the trajectory ended rather than the lag.
        collisions += self._step(self.settle_steps)
        end_pos = self.data.site_xpos[self.site_id].copy()

        print(f"  MuJoCo replay: executed pos_err mean={exec_err[:, 0].mean() * 1000:.2f}mm "
              f"max={exec_err[:, 0].max() * 1000:.2f}mm  ori_err mean={np.degrees(exec_err[:, 1]).mean():.2f}deg "
              f"max={np.degrees(exec_err[:, 1]).max():.2f}deg  (IK residual max "
              f"{ik_residual[:, 0].max() * 1000:.2f}mm)")
        return {
            "ik_residual": ik_residual,          # [n, (m, rad)] reference IK solution vs target
            "exec_err": exec_err,                # [n, (m, rad)] actual pose after each waypoint
            "achieved_path": achieved,           # [n, 3] actual site positions
            "end_pos": end_pos,                  # after settling
            "self_collision_steps": collisions,
            "self_contact_pairs": sorted(contact_pairs),
            "ik_collision_points": ik_collision_points,
            "ik_retries": int(retries),
            "ik_unresolved": int(unresolved),
            "max_cmd_jump_rad": max_jump,
        }


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
