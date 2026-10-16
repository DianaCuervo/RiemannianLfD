"""Collect every benchmark result into one CSV: one row per (shape, beta, method, variant, device).

Run from the repo root: python benchmark/collect_results.py [--results_dir ./benchmark/results]
"""
import argparse
import csv
import glob
import json
import os


def flatten(d, prefix=''):
    """Nested dict -> {'a.b.c': value}; lists (e.g. the radial profile) are skipped."""
    out = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{name}."))
        elif not isinstance(value, list):
            out[name] = value
    return out


def load(path):
    with open(path, 'r') as f:
        return json.load(f)


def collect(results_dir):
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(results_dir, '*'))):
        if not os.path.isdir(run_dir):
            continue
        run = os.path.basename(run_dir)

        # Method-independent rows: reconstruction floor and ground-truth off-manifold sanity row
        recon_path = os.path.join(run_dir, 'vae_reconstruction_metrics.json')
        if os.path.exists(recon_path):
            recon = load(recon_path)
            rows.append({'run': run, 'shape': recon.get('shape'), 'beta_scale': recon.get('beta_scale'),
                         'method': 'VAE reconstruction', 'variant': '', 'device': '',
                         **flatten(recon.get('ambient_metrics', {}), 'ambient_metrics.')})
        ref_path = os.path.join(run_dir, 'off_manifold_reference.json')
        if os.path.exists(ref_path):
            ref = load(ref_path)
            rows.append({'run': run, 'shape': ref.get('shape'), 'beta_scale': ref.get('beta_scale'),
                         'method': 'Ground truth', 'variant': '', 'device': '',
                         'off_manifold.floor': ref.get('floor'), 'off_manifold.wall': ref.get('wall'),
                         **flatten(ref.get('ground_truth_rates', {}), 'off_manifold.')})

        # One row per method metrics file and device it was timed on
        for metrics_path in sorted(glob.glob(os.path.join(run_dir, '*_metrics.json'))):
            if os.path.basename(metrics_path) == 'vae_reconstruction_metrics.json':
                continue
            metrics = load(metrics_path)
            tag = os.path.basename(metrics_path)[:-len('_metrics.json')]
            base = {'run': run, 'shape': metrics.get('shape'), 'beta_scale': metrics.get('beta_scale'),
                    'method': metrics.get('framework'), 'variant': metrics.get('variant', '')}
            flat = flatten({k: v for k, v in metrics.items()
                            if k not in ('framework', 'variant', 'shape', 'beta_scale', 'dataset')})

            timing_paths = sorted(glob.glob(os.path.join(run_dir, f"{glob.escape(tag)}_timing_*.json")))
            if not timing_paths:
                rows.append({**base, 'device': '', **flat})
            for timing_path in timing_paths:
                timing = load(timing_path)
                timing_flat = flatten({k: v for k, v in timing.items()
                                       if k not in ('framework', 'variant', 'device')}, 'timing.')
                rows.append({**base, 'device': timing.get('device', ''), **flat, **timing_flat})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Collect benchmark results into one CSV")
    parser.add_argument('--results_dir', type=str, default='./benchmark/results')
    parser.add_argument('--out', type=str, default=None, help="Default: <results_dir>/summary.csv")
    args = parser.parse_args()

    rows = collect(args.results_dir)
    if not rows:
        print(f"No results found under {args.results_dir}")
        return

    leading = ['run', 'shape', 'beta_scale', 'method', 'variant', 'device']
    columns = leading + sorted({k for row in rows for k in row} - set(leading))
    out_path = args.out or os.path.join(args.results_dir, 'summary.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ {len(rows)} rows written to {out_path}")


if __name__ == "__main__":
    main()
