"""Runs the closed-loop suites of benchmark/reactivity.py for NODE, graph and stochman.

Outputs in <results_dir>/<suite>/:
    perturbation_set.json                      the selected queries and their conditions
    <tag>_<suite>.json                         per-condition summary table + run settings
    <tag>_<suite>_rollouts.pt                  every rollout (paths, timings, metrics) for plots
<tag> gets a '_delay-<mode>' suffix for every replan-delay mode other than 'none'.
"""
import json
import os
import time

import numpy as np
import torch

from benchmark.data_utils import save_json
from benchmark.eval_node import node_model_name
from benchmark.off_manifold import encode_training_points
from benchmark.timing import WARMUP_QUERIES, device_label
from benchmark import reactivity as R
from node.evaluation.node_test import load_trained_node


def _set_settings(bench, rcfg, samples_per_path, suite):
    """Everything the perturbation set depends on; a cached set is reused only if these are unchanged."""
    keys = ('seed', 'fraction', 'push_mm', 'at_fraction', 'disturbance_mm') if suite == 'reactivity' else ('seed',)
    settings = {k: rcfg[k] for k in keys}
    if suite == 'query_noise':
        settings['query_noise'] = rcfg['query_noise']
    settings.update(suite=suite, samples_per_path=samples_per_path, n_queries=int(bench['ground_truth'].shape[0]))
    return json.loads(json.dumps(settings))                        # plain lists/dicts, comparable after a reload


def _set_or_load(bench, rcfg, samples_per_path, suite, cache_path):
    settings = _set_settings(bench, rcfg, samples_per_path, suite)
    if os.path.exists(cache_path):
        pset = torch.load(cache_path, weights_only=False)
        if pset.get('settings') == settings:
            print(f"✅ Found {suite} set at: {cache_path}")
            return pset
        print(f"♻️ {suite} set at {cache_path} was built with other settings: rebuilding it")
    pset = R.build_perturbation_set(bench, rcfg, samples_per_path, suite)
    pset['settings'] = settings
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(pset, cache_path)
    print(f"💾 {suite} set ({len(pset['idx'])} queries) saved to: {cache_path}")
    return pset


def _start_goal(bench, pset, j, perturber):
    """Start/goal of the j-th selected query (with the query noise applied for that suite)."""
    i = pset['idx'][j]
    z0 = bench['z1'][i:i + 1].detach().cpu().float()
    g0 = bench['z2'][i:i + 1].detach().cpu().float()
    cond = pset['conditions'][j]
    if cond['type'] == 'query_noise':
        rng = torch.Generator().manual_seed(pset['seed_per_query'][j] + 11)
        for which in ('start', 'goal'):
            if cond['apply_to'] in (which, 'both'):
                d = cond['magnitude'] * R.random_direction(rng)
                if which == 'start':
                    z0 = perturber.perturb(z0, d, cond['rot_deg'], rng)
                else:
                    g0 = perturber.perturb(g0, d, cond['rot_deg'], rng)
    return z0, g0


def run_suite(method_tag, rollout_fn, bench, pset, perturber, scorer, rcfg, out_dir, suite, extra=None):
    T = bench['ground_truth'].shape[1]
    dt = 1.0 / (T - 1)
    max_steps = int(rcfg['max_steps_factor'] * (T - 1))
    span = scorer.span
    tol = rcfg['tolerance']['latent_pct'] / 100 * span
    records, rollouts = [], []

    # Untimed warm-up (as in the standard suite): first-call costs must not land in p95 / max
    for j in range(min(WARMUP_QUERIES, len(pset['idx']))):
        z0, g0 = _start_goal(bench, pset, j, perturber)
        rollout_fn(z0, g0, R.Scenario(None, T - 1, pset['seed_per_query'][j]), dt, max_steps, tol)

    t0 = time.time()
    for j, i in enumerate(pset['idx']):
        cond = pset['conditions'][j]
        z0, g0 = _start_goal(bench, pset, j, perturber)
        new_goal = bench['z2'][pset['new_goal_idx'][j]:pset['new_goal_idx'][j] + 1].detach().cpu().float()
        sc = R.Scenario(cond if cond['type'] != 'query_noise' else None, T - 1, pset['seed_per_query'][j], new_goal)
        sc_ref = R.Scenario(None, T - 1, pset['seed_per_query'][j], new_goal)
        if cond['type'] == 'query_noise':                       # reference = the clean query
            zc, gc = bench['z1'][i:i + 1].cpu().float(), bench['z2'][i:i + 1].cpu().float()
            ref = rollout_fn(zc, gc, sc_ref, dt, max_steps, tol)
        else:
            ref = rollout_fn(z0, g0, sc_ref, dt, max_steps, tol)
        ro = rollout_fn(z0, g0, sc, dt, max_steps, tol)
        ref_m = scorer.score(ref, None, rcfg, sc_ref)
        ref_m['path'] = ref['path']
        m = scorer.score(ro, ref_m, rcfg, sc)
        if cond['type'] == 'query_noise':                       # stratum: did the noisy endpoints leave the data?
            above, beyond = scorer.off_data(torch.cat([z0, g0]))
            m['event_state_off'] = bool((above[0.5] | beyond).any())
        records.append(dict(query=i, condition=cond, metrics=m, step_ms=ro['step_ms']))
        rollouts.append(dict(query=i, condition=cond, path=ro['path'], ref_path=ref['path'], goal=ro['goal'],
                             step_ms=ro['step_ms'], event_ms=ro['event_ms'], replan_ms=ro['replan_ms'],
                             initial_plan_ms=ro['initial_plan_ms'], delay_steps=ro['delay_steps'],
                             event_step=ro['event_step'], diverged=ro['diverged'],
                             ref_diverged=ref['diverged'], ref_cycles=ref['n_cycles'], n_cycles=ro['n_cycles'],
                             metrics=m))
        if (j + 1) % 25 == 0:
            print(f"  [{method_tag}] {j + 1}/{len(pset['idx'])} rollouts ({time.time() - t0:.0f} s)", flush=True)

    rows = R.summarize(records, rcfg)
    report = dict(method=method_tag, suite=suite, n_queries=len(pset['idx']), T=T,
                  settings={k: v for k, v in rcfg.items() if k != 'query_noise'} | ({'query_noise': rcfg['query_noise']} if suite == 'query_noise' else {}),
                  **(extra or {}), conditions=rows)
    save_json(report, os.path.join(out_dir, f"{method_tag}_{suite}.json"))
    torch.save(rollouts, os.path.join(out_dir, f"{method_tag}_{suite}_rollouts.pt"))
    _print_table(method_tag, rows, rcfg)
    return report


def _print_table(tag, rows, rcfg):
    p = rcfg['control_periods_ms'][-1]
    print(f"\n{tag}: condition | n | success % | diverged % | goal mm (median) | cycles | extra length % | "
          f"off after event (a=0.5) % | reaction ms (median / mean / p95) | cycle ms (median / p95) | "
          f"missed cycles @{p} ms | delay steps | planner ok %")
    for r in rows:
        if r['stratum'] != 'all':
            continue
        name = f"{r['type']} {r['magnitude']:g}" + (f" @{r['at_fraction']:g}" if r['at_fraction'] else '') + \
               (f" {r['apply_to']} {r['rot_deg']:g}deg" if r['apply_to'] else '')
        f = lambda v, fmt: (fmt % v) if v is not None else '-'
        print(f"  {name:28s} {r['n']:4d} {f(r['success_pct'], '%6.1f')} {f(r['diverged_pct'], '%5.1f')} "
              f"{f(r['goal_pos_mm_median'], '%7.2f')} {f(r['cycles_median'], '%5.0f')} "
              f"{f(r['extra_length_pct_median'], '%7.1f')} {f(r['off_after_event_alpha_0.5_pct'], '%6.1f')} "
              f"{f(r['reaction_ms_median'], '%8.2f')} / {f(r['reaction_ms_mean'], '%8.2f')} / {f(r['reaction_ms_p95'], '%8.2f')} "
              f"{f(r['cycle_ms_median'], '%6.3f')} / {f(r['cycle_ms_p95'], '%6.3f')} {f(r[f'missed_cycles_{p}ms_median'], '%5.0f')} "
              f"{f(r['reaction_delay_steps_median'], '%5.0f')} {f(r['planner_success_pct'], '%6.1f')}")


def run_reactivity_benchmark(dataset_cfg, bench, results_dir, device, vae_model, vae_cpu, frameworks,
                             node_variants, suite='reactivity'):
    """Entry point used by run_benchmark.py --suite reactivity|query_noise."""
    rcfg = dataset_cfg['reactivity']
    out_dir = os.path.join(results_dir, suite)
    os.makedirs(out_dir, exist_ok=True)
    timing_device = rcfg.get('timing_device', 'cpu')
    if rcfg.get('cpu_threads'):
        torch.set_num_threads(int(rcfg['cpu_threads']))

    cache = os.path.join(dataset_cfg['benchmark_dir'], f"{suite}_set.pt")
    pset = _set_or_load(bench, rcfg, dataset_cfg['samples_per_path'], suite, cache)
    save_json({k: v for k, v in pset.items()}, os.path.join(out_dir, "perturbation_set.json"))

    perturber = R.AmbientPerturber(vae_cpu, bench['xy_center'], bench['xy_scale'], rcfg.get('project_z'))
    z_train = encode_training_points(vae_cpu, dataset_cfg['type'], bench['shape'])
    scorer = R.RolloutScorer(vae_cpu, bench, bench['off_ref'], z_train, bench['xy_center'], bench['xy_scale'])
    # A run stops as diverged once its state is this far out (same bound for every method)
    bound = float(rcfg.get('diverge_factor', 2.0)) * float(scorer.z_train.abs().max())
    modes = rcfg.get('replan_delay') or ['none']
    unknown = set(modes) - {'none', 'continue'}
    if unknown:
        raise ValueError(f"reactivity.replan_delay: unknown mode(s) {sorted(unknown)} (use 'none' and/or 'continue')")
    T = bench['ground_truth'].shape[1]

    # (tag, rollout_fn(period_ms), extra report fields) per method
    methods = []
    if 'node' in frameworks:
        fw = dataset_cfg['frameworks']['node'][dataset_cfg['type']]
        stall = rcfg['stall_pct'] / 100 * scorer.span
        for variant in node_variants:
            name = node_model_name(fw.get('model_name'), dataset_cfg['type'], bench['shape'], beta=bench['beta'],
                                   run_suffix=variant)
            path = os.path.join(fw.get('model_dir', f"./models/node/{dataset_cfg['type']}"), f"{name}.pth")
            arch = fw[bench['shape']] if dataset_cfg['type'] == 'lasa' else fw
            model = load_trained_node(path, latent_dim=arch['latent_dim'], hidden_dim=arch['hidden_dim'], device='cpu')
            make = lambda period, m=model: (lambda z0, g0, sc, dt, ms, tol: R.node_rollout(
                m, perturber, z0, g0, sc, dt, ms, tol, rcfg['patience'], stall, timing_device, period, bound))
            methods.append((f"node_{variant}", make, {}))

    if 'graph' in frameworks:
        from GeodesicMotionSkills.Experiments.Utils import discretized_manifold
        from benchmark.timing import QueryTimer
        from benchmark.eval_graph import graph_grid
        gfw = dataset_cfg['frameworks']['graph']
        lm, graph_size = graph_grid(gfw, dataset_cfg['type'], bench['ground_truth'].detach().cpu())
        ran = torch.linspace(-lm, lm, graph_size, device=device)
        x, y = torch.meshgrid(ran, ran, indexing='ij')
        with QueryTimer(device) as tm:
            dm = discretized_manifold.DiscretizedManifold(vae_model, torch.cat((x.unsqueeze(0), y.unsqueeze(0))))
        print(f"\n🕸️ graph {graph_size} x {graph_size} on [-{lm:.2f}, {lm:.2f}]^2, construction {tm.elapsed:.1f} s")
        graph_plan = R.make_graph_planner(vae_model, dm, T, lm)
        make = lambda period: (lambda z0, g0, sc, dt, ms, tol: R.planner_rollout(
            graph_plan, perturber, z0, g0, sc, dt, ms, tol, rcfg['replan_every'], device, timing_device, period, bound))
        methods.append(('graph', make, {'graph_size': graph_size, 'latent_frame': lm, 'graph_construction_s': tm.elapsed}))

    if 'stochman' in frameworks:
        stochman_plan = R.make_stochman_planner(vae_model, T, device)
        make = lambda period: (lambda z0, g0, sc, dt, ms, tol: R.planner_rollout(
            stochman_plan, perturber, z0, g0, sc, dt, ms, tol, rcfg['replan_every'], device, timing_device, period, bound))
        methods.append(('stochman', make, {}))

    reports = {}
    for mode in modes:
        period = None if mode == 'none' else float(rcfg['sim_period_ms'])
        for base_tag, make, extra in methods:
            tag = base_tag if mode == 'none' else f"{base_tag}_delay-{mode}"
            run_info = dict(replan_delay=mode, sim_period_ms=period, timing_device=device_label(timing_device),
                            planner_device=device_label(device), cpu_threads=torch.get_num_threads(),
                            diverge_bound=bound, latent_span=scorer.span, warmup_rollouts=WARMUP_QUERIES)
            print(f"\n🔁 {suite}: {tag}")
            reports[tag] = run_suite(tag, make(period), bench, pset, perturber, scorer, rcfg, out_dir, suite,
                                     extra={**extra, 'run_info': run_info})

    return reports
