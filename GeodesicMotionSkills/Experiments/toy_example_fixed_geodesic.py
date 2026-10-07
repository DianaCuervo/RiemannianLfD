"""Graph-based geodesic for the toy example, with two fixes applied on top of
toy_example.py / Utils/discretized_manifold.py (neither file is modified):

  1. RBF beta: nnj.RBF stores beta as a plain attribute, so it is not part of
     the checkpoint. toy_example.test_model() calls init_std() *before*
     load_state_dict(), so beta is computed from a randomly initialised encoder
     and differs on every run. Here beta is recomputed from the *trained*
     encoder after the checkpoint is loaded, the same way train_model() did it
     (init_std after encoder training, BatchNorm in training mode).

  2. Spline nodes: DiscretizedManifold.connecting_geodesic() fits a
     CubicSpline(p1, p2) with the default num_nodes=5 (4 cubic segments) to a
     graph path with ~200 nodes, so the fitted curve cuts corners and leaves the
     data support. Here a spline with more nodes is passed via the existing
     `curve` argument.

Usage (from GeodesicMotionSkills/Experiments):
    python toy_example_fixed_geodesic.py --beta fixed --num-nodes 5 20 0 --out results_fixed
    python toy_example_fixed_geodesic.py --beta original --num-nodes 5 20 0 --out results_original
num-nodes 0 means "one spline node per graph-path node".
"""

import argparse
import copy
import glob
import json
import os
import pickle
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")))
from GeodesicMotionSkills.Experiments import toy_example as te
from GeodesicMotionSkills.Experiments.Utils.discretized_manifold import DiscretizedManifold
from stochman.curves import CubicSpline

# Same settings as the __main__ block of toy_example.py (they are module globals there)
SETTINGS = dict(trajectory_number=7, test_id=0, encoder_scales=[1.0], graph_size=100, dof=5, pos_dof=2,
                qua_dof=3, latent_max=10, batch_size=128, r2_letter="J", s2_letter="C",
                model_path="../models", device="cpu")


def load_data():
    """Replicates the data loading of toy_example.py's __main__ block."""
    s = SETTINGS
    trajectory_flatten, test_traj = None, None
    for i in range(s["trajectory_number"]):
        r2_file = "../Dataset/letter_" + s["r2_letter"] + "_R2_" + str(i) + ".p"
        s2_file_test = "../Dataset/letter_" + s["s2_letter"] + "_S2_" + str(s["test_id"]) + ".p"
        s2_file = "../Dataset/letter_" + s["s2_letter"] + "_S2_" + str(i) + ".p"

        test_traj = pickle.load(open(r2_file, "rb"), encoding="latin1").transpose()
        trajectory = pickle.load(open(r2_file, "rb"), encoding="latin1").transpose()
        test_trajectory_qua = -pickle.load(open(s2_file_test, "rb"), encoding="latin1")
        trajectory_qua = pickle.load(open(s2_file, "rb"), encoding="latin1")

        trajectory_n = copy.deepcopy(trajectory)
        trajectory = np.append(trajectory, trajectory_qua, 1)
        trajectory_n = np.append(trajectory_n, -trajectory_qua, 1)
        test_traj = np.append(test_traj, test_trajectory_qua, 1)

        block = np.vstack([trajectory, trajectory_n])
        trajectory_flatten = block if trajectory_flatten is None else np.vstack([trajectory_flatten, block])
    input_data = trajectory_flatten[:, 0:s["dof"]]
    return torch.from_numpy(input_data).float(), test_traj


def trained_beta(model, x):
    """beta as computed in init_std() during train_model(): encoder already trained,
    model still in training mode (BatchNorm uses batch statistics). A copy of the
    encoder is used so the loaded BatchNorm running statistics are not touched."""
    enc = copy.deepcopy(model.encoder_loc)
    enc.train()
    with torch.no_grad():
        z = enc(x)
    return 10.0 / z.std(dim=0).mean()


def set_beta(model, beta):
    rbf_beta = beta * torch.ones(1, model.num_clusters)
    model.dec_std_pos[0].beta = rbf_beta.view(1, -1)
    model.dec_std_qua[0].beta = rbf_beta.view(1, -1)


def load_model(x, fix_beta):
    """Same loading sequence as toy_example.test_model(), optionally followed by the beta fix."""
    s = SETTINGS
    fn = sorted(glob.glob(s["model_path"] + "/*.pt"))[0]
    model = te.VAE(layers=[s["dof"], 200, 100, 2], batch_size=s["batch_size"], sigma_z=s["encoder_scales"][0])
    model.obstacle_input_space = None
    model.init_std(x, load_clusters=True)  # beta computed here from the untrained encoder
    checkpoint = torch.load(fn)
    model.load_state_dict(checkpoint["model_state_dict"])  # beta is not in the checkpoint
    if fix_beta:
        set_beta(model, trained_beta(model, x))
    model.disable_training()
    model.eval()
    return model, fn


def curve_metrics(eval_model, curve, demo_pos, n=500):
    """Energy/length under eval_model's metric, max position std along the curve, and
    the distance of the decoded trajectory to the demonstration (metric independent)."""
    alpha = torch.linspace(0, 1, n)
    with torch.no_grad():
        pts = curve(alpha)
        energy = eval_model.curve_energy(pts.unsqueeze(0).clone()).item()
        length = eval_model.curve_length(pts.unsqueeze(0).clone()).item()
        emb = eval_model.embed(pts.unsqueeze(0))[0]
        decoded = eval_model.decode(pts, train_rbf=True)[0].mean.numpy()
    dist = np.linalg.norm(decoded[:, None, :] - demo_pos[None, :, :], axis=2).min(axis=1)
    return dict(energy=energy, length=length, max_std_pos=emb[:, 5:7].max().item(),
                decoded_to_demo_mean=float(dist.mean()), decoded_to_demo_max=float(dist.max())), pts.numpy(), decoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", choices=["original", "fixed"], default="fixed")
    parser.add_argument("--num-nodes", type=int, nargs="+", default=[5, 20, 0])
    parser.add_argument("--out", default="geodesic_fix_results")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for k, v in SETTINGS.items():  # toy_example's functions read these as module globals
        setattr(te, k, v)
    os.makedirs(args.out, exist_ok=True)

    x, test_traj = load_data()
    model, fn = load_model(x, fix_beta=(args.beta == "fixed"))
    # All curves are also scored under the corrected metric so runs are comparable
    eval_model = model if args.beta == "fixed" else load_model(x, fix_beta=True)[0]
    beta = model.dec_std_pos[0].beta.flatten()[0].item()
    print("checkpoint %s | beta used for graph: %.4f | corrected beta: %.4f"
          % (fn, beta, eval_model.dec_std_pos[0].beta.flatten()[0].item()))

    s = SETTINGS
    ran = torch.linspace(-s["latent_max"], s["latent_max"], s["graph_size"])
    gx, gy = torch.meshgrid(ran, ran, indexing="ij")
    grid = torch.cat((gx.unsqueeze(0), gy.unsqueeze(0)))
    t0 = time.time()
    dm = DiscretizedManifold(model, grid)
    print("graph built in %.0fs" % (time.time() - t0))

    with torch.no_grad():
        p0 = model.encode(torch.tensor(test_traj[0]).float(), train_rbf=True)[1].view(1, -1)
        p1 = model.encode(torch.tensor(test_traj[-1]).float(), train_rbf=True)[1].view(1, -1)

    results = dict(beta_mode=args.beta, beta=beta, curves={})
    plt.figure(figsize=(12, 6))
    ax_lat, ax_dec = plt.subplot(1, 2, 1), plt.subplot(1, 2, 2)
    with torch.no_grad():
        _, z_data = eval_model.encode(x, train_rbf=True)
    ax_lat.scatter(z_data[:, 0], z_data[:, 1], s=2, c="lightgray", label="training data (latent)")
    ax_dec.plot(test_traj[:, 0], test_traj[:, 1], "k", lw=3, alpha=0.3, label="demonstration")

    for k in args.num_nodes:
        dm.connecting_geodesic(p0, p1, model, 0)  # sets dm.coordinates; path length needed for k=0
        n_nodes = k if k > 0 else dm.coordinates.shape[0] + 2
        if n_nodes == 5:
            curve = dm.connecting_geodesic(p0, p1, model, 0)  # exactly the original call
        else:
            curve = dm.connecting_geodesic(p0, p1, model, 0, curve=CubicSpline(p0, p1, num_nodes=n_nodes))
        m, latent, decoded = curve_metrics(eval_model, curve, test_traj[:, :2])
        m["num_nodes"] = n_nodes
        results["curves"][str(n_nodes)] = m
        print("num_nodes=%3d  energy=%.4g  length=%.4g  max_std_pos=%.4g  decoded->demo mean=%.4g max=%.4g"
              % (n_nodes, m["energy"], m["length"], m["max_std_pos"], m["decoded_to_demo_mean"],
                 m["decoded_to_demo_max"]))
        ax_lat.plot(latent[:, 0], latent[:, 1], label="spline, %d nodes" % n_nodes)
        ax_dec.plot(decoded[:, 0], decoded[:, 1], label="spline, %d nodes" % n_nodes)
    graph_nodes = dm.coordinates.numpy()
    ax_lat.plot(graph_nodes[:, 0], graph_nodes[:, 1], "k.", ms=2, label="graph path")
    ax_lat.set_title("latent space (beta=%.3g)" % beta)
    ax_dec.set_title("decoded position")
    ax_lat.legend(fontsize=7)
    ax_dec.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "geodesic_beta-%s.png" % args.beta))
    with open(os.path.join(args.out, "metrics_beta-%s.json" % args.beta), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
