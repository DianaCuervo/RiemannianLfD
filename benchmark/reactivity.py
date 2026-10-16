"""Closed-loop benchmark: reactivity to mid-run perturbations, and noisy start/goal queries.

Every method runs in the same simulated control loop with step dt = 1/(T-1):

  * NODE is a policy: every control cycle it encodes the measured state, takes ONE RK4 step of
    its field f(z, g) and decodes the next state into a command. Nothing is ever replanned.
  * Graph and stochman are planners: they compute a whole path, the loop then follows it
    (velocity feed-forward along the plan), and every perturbation event triggers a full replan
    from the current state.

Suites (one condition per selected query, balanced, seeded, identical for every method):
  reactivity:  'push'        ambient in-plane displacement of the state at a fraction of the run
               'goal_switch' the goal is replaced by the goal of another query of the same demo
               'disturbance' ambient in-plane noise added to the state at every step
  query_noise: 'query_noise' ambient in-plane displacement (mm) and rotation (deg) of the start,
               the goal or both, before the run (no imitation metrics: there is no demonstration)

Each perturbed rollout is paired with an unperturbed rollout of the same method and query.
Dynamics are simulated in latent space; the timed block of a NODE cycle also includes the
encode/decode a robot would need, but the state is propagated from the latent point itself, so
the encoder/decoder round trip does not add drift to the simulation.
"""
import math

import numpy as np
import torch

from benchmark.metrics import decode_latente_paths, to_original_units
from benchmark.off_manifold import log10_det_M, ALPHAS
from benchmark.timing import QueryTimer


# ======================================================================================
# 1. PERTURBATION SET
# ======================================================================================
def _conditions(rcfg, suite):
    if suite == 'reactivity':
        conds = [('push', float(mm), float(f)) for mm in rcfg['push_mm'] for f in rcfg['at_fraction']]
        conds += [('goal_switch', 0.0, float(f)) for f in rcfg['at_fraction']]
        conds += [('disturbance', float(s), 0.0) for s in rcfg['disturbance_mm']]
        return [dict(type=t, magnitude=m, at_fraction=f, apply_to='', rot_deg=0.0) for t, m, f in conds]
    qn = rcfg['query_noise']
    return [dict(type='query_noise', magnitude=float(mm), at_fraction=0.0, apply_to=a, rot_deg=float(deg))
            for a in qn['apply_to'] for mm, deg in qn['levels']]


def build_perturbation_set(bench, rcfg, samples_per_path, suite='reactivity'):
    """Select queries (stratified by demonstration path and length tertile) and assign conditions.

    Returns a dict of plain lists/tensors, saved next to the benchmark data so every method and
    every later run sees exactly the same perturbations.
    """
    gt = bench['ground_truth'].detach().cpu()
    B = gt.shape[0]
    gen = torch.Generator().manual_seed(int(rcfg['seed']))
    path_id = torch.arange(B) // samples_per_path                      # 14 encoded paths x 50 windows
    length = (gt[:, 1:] - gt[:, :-1]).norm(dim=-1).sum(1)
    tertile = torch.bucketize(length, torch.quantile(length, torch.tensor([1 / 3, 2 / 3])))

    fraction = rcfg['fraction'] if suite == 'reactivity' else rcfg['query_noise']['fraction']
    n_sel = min(B, int(round(fraction * B)))
    cells = {}
    for i in torch.randperm(B, generator=gen).tolist():
        cells.setdefault((int(path_id[i]), int(tertile[i])), []).append(i)
    keys, chosen, depth = sorted(cells), [], 0
    while len(chosen) < n_sel:                                           # round-robin over strata
        for key in keys:
            if depth < len(cells[key]) and len(chosen) < n_sel:
                chosen.append(cells[key][depth])
        depth += 1
    idx = sorted(chosen)

    conds = _conditions(rcfg, suite)
    assign = [conds[j % len(conds)] for j in range(len(idx))]
    assign = [assign[j] for j in torch.randperm(len(idx), generator=gen).tolist()]

    new_goal_idx = []
    for i in idx:
        same = torch.nonzero(path_id == path_id[i]).flatten()
        same = same[same != i]
        new_goal_idx.append(int(same[torch.randint(len(same), (1,), generator=gen)]))

    return {
        'suite': suite,
        'idx': idx,
        'conditions': assign,
        'seed_per_query': [int(rcfg['seed']) * 100003 + i for i in idx],   # per-query RNG for directions/noise
        'new_goal_idx': new_goal_idx,
        'path_id': path_id[idx].tolist(),
        'length_tertile': tertile[idx].tolist(),
    }


# ======================================================================================
# 2. AMBIENT PERTURBATIONS
# ======================================================================================
class AmbientPerturber:
    """Applies a physical (ambient) perturbation to a latent state.

    z' = z + [enc(x + delta) - enc(x)],  x = dec(z)
    so a zero perturbation leaves z unchanged (the encoder/decoder round trip is not applied).
    Positions are pushed in the demonstration plane; with project_z the out-of-plane coordinate is
    set to that plane value before encoding (the LASA VAEs never saw z != 0 and extrapolate wildly).
    """

    def __init__(self, vae, xy_center, xy_scale, project_z=None):
        self.vae = vae
        self.c = torch.as_tensor(xy_center, dtype=torch.float32)
        self.s = float(xy_scale)
        self.project_z = project_z
        self.pos_dof = vae.pos_dof

    @torch.no_grad()
    def pose(self, z):
        pos, qua = self.vae.decode(z.reshape(-1, 2))[:2]
        p = pos.mean.reshape(z.shape[0], -1).clone()
        if self.project_z is not None and self.pos_dof > 2:
            p[:, 2:] = float(self.project_z)
        return torch.cat([p, qua.loc.reshape(z.shape[0], -1)], dim=-1)

    @torch.no_grad()
    def encode(self, x):
        return self.vae.encode(x, train_rbf=True)[1]

    @torch.no_grad()
    def perturb(self, z, dxy_mm=None, rot_deg=0.0, rng=None):
        x = self.pose(z)
        xp = x.clone()
        if dxy_mm is not None:
            xp[:, :2] = xp[:, :2] + torch.as_tensor(dxy_mm, dtype=torch.float32).reshape(1, 2) / self.s
        if rot_deg:
            xp[:, self.pos_dof:] = rotate_quaternion(xp[:, self.pos_dof:], rot_deg, rng)
        return z + (self.encode(xp) - self.encode(x))


def rotate_quaternion(q, deg, rng):
    """Rotate unit quaternions by exactly `deg` about a random axis, independent of component order:
    q' = cos(h) q + sin(h) u with u a random unit vector orthogonal to q, h = deg/2."""
    u = torch.randn(q.shape, generator=rng)
    u = u - (u * q).sum(-1, keepdim=True) * q
    u = u / u.norm(dim=-1, keepdim=True)
    h = math.radians(deg) / 2
    return math.cos(h) * q + math.sin(h) * u


def random_direction(rng):
    a = float(torch.rand(1, generator=rng)) * 2 * math.pi
    return torch.tensor([math.cos(a), math.sin(a)])


# ======================================================================================
# 3. CLOSED-LOOP ROLLOUTS
# ======================================================================================
def _rk4(f, z, dt):
    k1 = f(z)
    k2 = f(z + dt * k1 / 2)
    k3 = f(z + dt * k2 / 2)
    k4 = f(z + dt * k3)
    return z + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


class Scenario:
    """The perturbation events of one rollout, at control-step resolution."""

    def __init__(self, cond, n_nominal, rng_seed, new_goal=None):
        self.cond = cond
        self.type = cond['type'] if cond else 'none'
        self.at_fraction = cond['at_fraction'] if cond else None
        self.fired = False
        self.seed = rng_seed
        self.new_goal = new_goal
        rng = torch.Generator().manual_seed(rng_seed)
        self.push_dir = random_direction(rng)                       # same direction for every method

    def due(self, z, z0, g0):
        """Push / goal switch fire once the state has covered `at_fraction` of the start-goal distance.
        Progress-based (not step-based), so the event happens at the same point of the motion for every
        method regardless of its speed profile, and never after a method has already arrived."""
        if self.fired or self.type not in ('push', 'goal_switch'):
            return False
        d0 = float((z0 - g0).norm())
        progress = 1.0 - float((z - g0).norm()) / max(d0, 1e-12)
        if progress >= self.at_fraction:
            self.fired = True
            return True
        return False

    def pending(self):
        return self.type in ('push', 'goal_switch') and not self.fired

    def noise_rng(self):
        return torch.Generator().manual_seed(self.seed + 7)


def node_rollout(model, perturber, z0, g0, sc, dt, max_steps, tol, patience, stall, timing_device='cpu'):
    """NODE as a policy: one RK4 step of f(z, g) per control cycle; no replanning."""
    f_of = lambda g: (lambda x: model.ode_func(0, x, g))
    z, g = z0.clone(), g0.clone()
    states, step_ms, events = [z.clone()], [], []
    reaction_ms, event_step = None, None
    best, best_k, still = float('inf'), 0, 0
    rng = sc.noise_rng()
    with torch.no_grad():
        for k in range(max_steps):
            event = False
            if sc.due(z, z0, g0):
                if sc.type == 'push':
                    z = perturber.perturb(z, sc.cond['magnitude'] * sc.push_dir)
                    states.append(z.clone())
                else:
                    g = sc.new_goal.clone()
                event = True
            elif sc.type == 'disturbance' and k > 0:
                z = perturber.perturb(z, sc.cond['magnitude'] * torch.randn(2, generator=rng))
                states.append(z.clone()); event = True
            if event and event_step is None:
                event_step = len(states) - 1
            if event and sc.type in ('push', 'goal_switch'):
                best, best_k = float('inf'), len(states) - 1          # the closest approach must come after the event
            x_meas = perturber.pose(z)                               # simulated measurement (untimed)
            with QueryTimer(timing_device) as tm:                     # one control cycle
                perturber.encode(x_meas)                             # measured pose -> latent
                z_next = _rk4(f_of(g), z, dt)                        # one step of the field
                model_cmd = perturber.vae.decode(z_next)             # next latent -> pose command
            step_ms.append(tm.elapsed * 1e3)
            if event:
                events.append(tm.elapsed * 1e3)
                if reaction_ms is None:
                    reaction_ms = tm.elapsed * 1e3
            z = z_next
            states.append(z.clone())
            d = float((z - g).norm())
            if d < best:
                best, best_k = d, len(states) - 1
            if d < tol and not sc.pending():
                break
            # Stop when the state has stopped moving for `patience` steps (converged short of the goal).
            # Distance to the goal is NOT used here: on curved shapes it grows temporarily after a push.
            moved = float((states[-1] - states[-2]).norm())
            still = still + 1 if moved < stall else 0
            if still > patience:
                break
    path = torch.cat(states[:best_k + 1])                             # end at the closest approach (after the last push/goal switch)
    return dict(path=path, goal=g, step_ms=step_ms, event_ms=events, reaction_ms=reaction_ms,
                event_step=event_step, n_replans=0, planner_success=None)


def planner_rollout(plan, perturber, z0, g0, sc, dt, max_steps, tol, replan_every, plan_device,
                    timing_device):
    """Planner: plan once, follow the plan (velocity feed-forward), replan on every event.

    Replans are timed on plan_device (the device the planner runs on); normal cycles on timing_device."""
    z, g = z0.clone(), g0.clone()
    states, events, cycle_ms = [z.clone()], [], []
    reaction_ms, event_step, successes = None, None, []
    rng = sc.noise_rng()

    def replan(zc, gc):
        x_meas = perturber.pose(zc)
        with QueryTimer(plan_device) as tm:                      # synchronises if the planner runs on the GPU
            perturber.encode(x_meas)
            P, ok = plan(zc.to(plan_device), gc.to(plan_device))
            perturber.vae.decode(P[1:2].cpu())
        successes.append(ok)
        return P.detach().cpu(), tm.elapsed * 1e3

    P, t_plan = replan(z, g)
    initial_ms, i = t_plan, 0
    with torch.no_grad():
        for k in range(max_steps):
            event = False
            if sc.due(z, z0, g0):
                if sc.type == 'push':
                    z = perturber.perturb(z, sc.cond['magnitude'] * sc.push_dir)
                    states.append(z.clone())
                else:
                    g = sc.new_goal.clone()
                event = True
            elif sc.type == 'disturbance' and k > 0:
                z = perturber.perturb(z, sc.cond['magnitude'] * torch.randn(2, generator=rng))
                states.append(z.clone())
                event = (k % replan_every == 0)
            if event:
                if event_step is None:
                    event_step = len(states) - 1
                P, t = replan(z, g)
                i = 0
                events.append(t)
                if reaction_ms is None:
                    reaction_ms = t
            if i >= P.shape[0] - 1 and not sc.pending():
                break                                                 # end of the plan
            if i >= P.shape[0] - 1:
                z = z.clone(); states.append(z.clone()); continue     # at the plan's end, event still pending: hold
            with QueryTimer(timing_device) as tm:                     # normal cycle: next plan point
                perturber.vae.decode(P[i + 1:i + 2])
            cycle_ms.append(tm.elapsed * 1e3)
            z = z + (P[i + 1:i + 2] - P[i:i + 1])                     # follow the plan from the current state
            i += 1
            states.append(z.clone())
            if float((z - g).norm()) < tol and not sc.pending():
                break
    return dict(path=torch.cat(states), goal=g, step_ms=cycle_ms, event_ms=events, reaction_ms=reaction_ms,
                event_step=event_step, initial_plan_ms=initial_ms, n_replans=len(events),
                planner_success=float(np.mean(successes)) if successes else None)


def make_graph_planner(vae, discrete_model, T, latent_frame):
    """Graph planner. A plan counts as failed (planner_success) when an end point lies outside the
    grid [-latent_frame, latent_frame]^2: the graph snaps it to the border and the fitted curve is unreliable."""
    alpha = torch.linspace(0, 1, T).reshape(-1, 1)

    def plan(p0, p1):
        curve = discrete_model.connecting_geodesic(p0.reshape(-1), p1.reshape(-1), vae, vae.time_step)
        a = alpha.to(curve.device)
        inside = bool(torch.cat([p0, p1]).abs().max() <= latent_frame)
        return curve(a.transpose(1, 0)).detach().reshape(T, -1), inside
    return plan


def make_stochman_planner(vae, T, device):
    from stochman.manifold import Manifold
    from stochman.curves import CubicSpline
    t = torch.linspace(0, 1, T, device=device)

    def plan(p0, p1):
        p0, p1 = p0.reshape(1, -1), p1.reshape(1, -1)
        C, success = Manifold.connecting_geodesic(vae, p0, p1, init_curve=CubicSpline(p0, p1).to(device))
        return C(t).detach().reshape(T, -1), bool(success)
    return plan


# ======================================================================================
# 4. METRICS
# ======================================================================================
class RolloutScorer:
    """Ambient / latent / off-manifold metrics of variable-length rollouts (all on CPU)."""

    def __init__(self, vae_cpu, bench, off_ref, z_train, xy_center, xy_scale):
        self.vae = vae_cpu
        self.ref = off_ref
        self.tau = {a: off_ref['floor'] + a * (off_ref['wall'] - off_ref['floor']) for a in ALPHAS}
        bins, med = torch.as_tensor(off_ref['profile_bins']), torch.as_tensor(off_ref['profile_median_m'])
        self.wall_dist_pct = float(bins[med.argmax()])               # % of span where the radial profile peaks
        self.z_train = torch.as_tensor(z_train, dtype=torch.float32)
        self.span = float(off_ref['span'])
        self.c, self.s = xy_center, xy_scale
        self.pos_dof = vae_cpu.pos_dof

    def ambient(self, path):
        x = decode_latente_paths(self.vae, path.reshape(1, -1, 2))
        return np.asarray(to_original_units(x, self.c, self.s))[0]

    def off_data(self, pts):
        """Per point: above tau_alpha (dict) and beyond the wall (nearest training point too far)."""
        m = log10_det_M(self.vae, pts).float()
        nn = torch.cdist(pts.float(), self.z_train).min(1).values / self.span * 100
        return {a: (m > t) for a, t in self.tau.items()}, nn > self.wall_dist_pct

    def score(self, ro, ref_ro, rcfg, sc):
        path, g = ro['path'], ro['goal'].reshape(1, 2)
        x = self.ambient(path)
        xg = self.ambient(g)[0]
        pd = self.pos_dof
        end = x[-1]
        q1, q2 = end[pd:] / np.linalg.norm(end[pd:]), xg[pd:] / np.linalg.norm(xg[pd:])
        rot_err = float(np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(q1, q2)))))))
        pos_err = float(np.linalg.norm(end[:2] - xg[:2]))
        lat_err = float((path[-1:] - g).norm()) / self.span * 100
        length = float(np.linalg.norm(np.diff(x[:, :2], axis=0), axis=1).sum())
        tol = rcfg['tolerance']
        r = {
            'success': bool(pos_err < tol['pos_mm'] and rot_err < tol['rot_deg']),
            'success_latent': bool(lat_err < tol['latent_pct']),
            'goal_pos_mm': pos_err, 'goal_rot_deg': rot_err, 'goal_latent_pct': lat_err,
            'length_mm': length, 'steps': int(path.shape[0] - 1),
            'reaction_ms': ro['reaction_ms'], 'n_replans': ro['n_replans'],
            'cycle_ms_median': float(np.median(ro['step_ms'])) if ro['step_ms'] else None,
            'compute_ms_total': float(np.sum(ro['step_ms']) + np.sum(ro['event_ms']) + ro.get('initial_plan_ms', 0.0)),
            'planner_success': ro['planner_success'],
        }
        # Off-manifold: whole path and the part after the first event
        above, beyond = self.off_data(path)
        k0 = ro['event_step'] if ro['event_step'] is not None else 0
        for a in ALPHAS:
            r[f'off_alpha_{a}'] = bool(above[a].any())
            r[f'off_after_event_alpha_{a}'] = bool(above[a][k0:].any())
        r['beyond_wall'] = bool(beyond.any())
        if ro['event_step'] is not None and sc.type in ('push', 'disturbance'):
            r['event_state_off'] = bool(above[0.5][k0] or beyond[k0])   # stratum: where the push landed
        r['event_fired'] = ro['event_step'] is not None if sc.type != 'none' else None
        if ref_ro is not None and sc.type != 'goal_switch':
            r['extra_length_pct'] = 100.0 * (length / max(ref_ro['length_mm'], 1e-9) - 1.0)
            if sc.type == 'push' and ro['event_step'] is not None:  # steps until back near the unperturbed path
                band = rcfg['recover_band_pct'] / 100 * self.span
                d = torch.cdist(path[k0:], ref_ro['path']).min(1).values
                back = torch.nonzero(d < band).flatten()
                r['recovery_steps'] = int(back[0]) if len(back) else None
        for p in rcfg['control_periods_ms']:
            if ro['reaction_ms'] is not None:
                r[f'missed_cycles_{p}ms'] = int(ro['reaction_ms'] // p)
                r[f'deadline_miss_{p}ms'] = bool(ro['reaction_ms'] > p)
        return r


def _pct(vals):
    v = [x for x in vals if x is not None]
    return float(100.0 * np.mean(v)) if v else None


def _q(vals, q):
    v = [x for x in vals if x is not None]
    return float(np.percentile(v, q)) if v else None


def summarize(records, rcfg):
    """Group per condition (and per where the push landed) -> table rows."""
    groups = {}
    for rec in records:
        c = rec['condition']
        key = (c['type'], c['magnitude'], c['at_fraction'], c.get('apply_to', ''), c.get('rot_deg', 0.0))
        groups.setdefault(key, []).append(rec)
        if 'event_state_off' in rec['metrics']:
            stratum = 'pushed_off_data' if rec['metrics']['event_state_off'] else 'pushed_on_data'
            groups.setdefault(key + (stratum,), []).append(rec)
    rows = []
    for key, recs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        m = [r['metrics'] for r in recs]
        row = dict(type=key[0], magnitude=key[1], at_fraction=key[2], apply_to=key[3], rot_deg=key[4],
                   stratum=key[5] if len(key) > 5 else 'all', n=len(recs),
                   success_pct=_pct([x['success'] for x in m]),
                   success_latent_pct=_pct([x['success_latent'] for x in m]),
                   goal_pos_mm_median=_q([x['goal_pos_mm'] for x in m], 50), goal_pos_mm_p95=_q([x['goal_pos_mm'] for x in m], 95),
                   goal_rot_deg_median=_q([x['goal_rot_deg'] for x in m], 50),
                   extra_length_pct_median=_q([x.get('extra_length_pct') for x in m], 50),
                   recovery_steps_median=_q([x.get('recovery_steps') for x in m], 50),
                   recovered_pct=_pct([x.get('recovery_steps') is not None for x in m]) if key[0] == 'push' else None,
                   reaction_ms_median=_q([x['reaction_ms'] for x in m], 50), reaction_ms_p95=_q([x['reaction_ms'] for x in m], 95),
                   reaction_ms_max=_q([x['reaction_ms'] for x in m], 100),
                   replans_median=_q([x['n_replans'] for x in m], 50),
                   compute_ms_total_median=_q([x['compute_ms_total'] for x in m], 50),
                   planner_success_pct=_pct([x['planner_success'] for x in m if x['planner_success'] is not None]),
                   beyond_wall_pct=_pct([x['beyond_wall'] for x in m]),
                   event_fired_pct=_pct([x.get('event_fired') for x in m]))
        for a in ALPHAS:
            row[f'off_after_event_alpha_{a}_pct'] = _pct([x[f'off_after_event_alpha_{a}'] for x in m])
        for p in rcfg['control_periods_ms']:
            row[f'deadline_miss_{p}ms_pct'] = _pct([x.get(f'deadline_miss_{p}ms') for x in m])
            row[f'missed_cycles_{p}ms_median'] = _q([x.get(f'missed_cycles_{p}ms') for x in m], 50)
        rows.append(row)
    return rows
