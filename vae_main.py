"""Train, test and visualize the VAE, separately from main.py's NODE pipeline.

The raw-data location comes from node_config.yaml's 'dataset' section; everything else
from vae_config.yaml, with training/test settings overridable from the command line.

The toy beta_scale is selected in vae_config.yaml only; lasa's comes from --beta_scale.

Usage:
    python vae_main.py --mode train --dataset toy
    python vae_main.py --mode train --dataset lasa --shape N --beta_scale 5
    python vae_main.py --mode test --dataset lasa --shape N --beta_scale 5 --test_id 3
    python vae_main.py --mode visualize --dataset lerobot --task pick
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

from node.utils.logger import ConsoleLogger
from vae.data.vae_dataset import load_vae_trajectories, select_test_demo
from vae.evaluation.vae_test_with_graph import test_vae_with_graph
from vae.training.vae_train import train_vae
from vae.utils.config import load_vae_config, load_dataset_config
from vae.utils.plots import plot_metric_with_latent_points
from vae.vae_model import load_pretrained_vae

LOGS_DIR = './vae/training/training_logs'
CURVES_DIR = './vae/training/training_curves'
RESULTS_DIR = './vae/evaluation/results'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)  # also seeds the KMeans in VAE.init_std
    torch.manual_seed(seed)


def load_trained_vae(vae_cfg, trajectories):
    """Load the VAE exactly as main.py does, so the metric matches the NODE pipeline's."""
    points = torch.from_numpy(np.vstack(trajectories)).float()
    return load_pretrained_vae(vae_cfg, points, device='cpu')


def main():
    parser = argparse.ArgumentParser(description="Run the RiemannianLfD VAE pipeline")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test', 'visualize'])
    parser.add_argument('--dataset', type=str, required=True, choices=['toy', 'lasa', 'lerobot'])
    parser.add_argument('--shape', type=str, default=None, help="LASA shape (e.g. N, Angle)")
    parser.add_argument('--task', type=str, default=None, help="LEROBOT task (e.g. pick, place)")
    parser.add_argument('--beta_scale', type=str, default=None,
                        help="RBF beta scale for the lasa VAE (e.g. 1, 5, 10); one VAE per value. "
                             "The toy one is selected in vae_config.yaml")
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--artifacts_dir', type=str, default=None,
                        help="Read/write the model and KMeans clusters in this folder instead of "
                             "the configured paths, e.g. for smoke runs that must not touch the "
                             "shipped weights")
    # train
    parser.add_argument('--epochs', type=int, default=None, help="Override stage 1/2 epochs")
    parser.add_argument('--epochs_rbf', type=int, default=None, help="Override stage 3 epochs")
    parser.add_argument('--val_ratio', type=float, default=0.3)
    parser.add_argument('--overwrite', action='store_true', help="Replace an existing checkpoint")
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'],
                        help="Training device; defaults to cuda when available")
    # test
    parser.add_argument('--test_id', type=int, default=None,
                        help="Demo whose points the geodesics connect; defaults to the dataset "
                             "config's test_id, else 0")
    parser.add_argument('--graph_size', type=int, default=100, help="Graph nodes per latent axis")
    parser.add_argument('--num_segments', type=int, default=5,
                        help="Random demo sub-segments to plan, besides the full demo")
    parser.add_argument('--time_steps', type=int, default=500, help="Points along each geodesic")
    args = parser.parse_args()

    # Initialize the VAE config and dataset config, with placeholders resolved
    vae_cfg = load_vae_config(args.dataset, args.shape, args.task, beta=args.beta_scale)
    dataset_cfg = load_dataset_config(args.dataset, args.shape, args.task)

    artifacts = vae_cfg['training_artifacts']
    if args.artifacts_dir:
        for key in ('model_path', 'cluster_path'):
            artifacts[key] = os.path.join(args.artifacts_dir, os.path.basename(artifacts[key]))
    vae_name = os.path.splitext(os.path.basename(artifacts['model_path']))[0]  # e.g. VAE_toy_beta_scale_10

    sys.stdout = ConsoleLogger()
    sys.stdout.set_log_file(save_dir=LOGS_DIR, model_name=f"{vae_name}_{args.mode}")
    print(f"Starting VAE {args.mode} for Dataset: {args.dataset.upper()} | {vae_name} | "
          f"beta_scale: {vae_cfg['architecture'].get('beta_scale')} | Seed: {args.seed}")
    set_seed(args.seed)

    if args.mode == 'train':
        training = vae_cfg.setdefault('training', {})
        if args.epochs is not None:
            training['epochs'] = args.epochs
        if args.epochs_rbf is not None:
            training['epochs_rbf'] = args.epochs_rbf
        device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"\n🚀 Launching VAE Training on {device.upper()}...")
        train_vae(vae_cfg, dataset_cfg, device=device, seed=args.seed, val_ratio=args.val_ratio,
                  overwrite=args.overwrite, curves_dir=CURVES_DIR)
        return

    latent_frame = vae_cfg['visualization']['latent_frame']
    results_dir = os.path.join(RESULTS_DIR, vae_name)
    trajectories = load_vae_trajectories(dataset_cfg)
    vae = load_trained_vae(vae_cfg, trajectories)

    if args.mode == 'test':
        test_id = args.test_id if args.test_id is not None else dataset_cfg.get('test_id', 0)
        print(f"Test demo: {test_id}")
        test_vae_with_graph(vae, trajectories, select_test_demo(trajectories, test_id), results_dir,
                            space_name=vae_name, latent_frame=latent_frame,
                            graph_size=args.graph_size, num_random_segments=args.num_segments,
                            time_steps=args.time_steps, seed=args.seed)

    elif args.mode == 'visualize':
        with torch.no_grad():
            z_train = vae.encode(torch.from_numpy(np.vstack(trajectories)).float(), train_rbf=True)[1]
        plot_metric_with_latent_points(vae, z_train.numpy(), vae_name, latent_frame, results_dir,
                                       f"{vae_name} Manifold.svg")


if __name__ == "__main__":
    main()
