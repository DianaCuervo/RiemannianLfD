import os

import numpy as np
import torch
from tqdm import tqdm

from vae.data.vae_dataset import load_vae_trajectories, build_point_dataset
from vae.utils.plots import plot_vae_training_curves
from vae.vae_model import VAE


def log_prob(model, x, positional_dist, quaternion_dist, quaternion_dist_n):
    """Compute the log probability of quaternion vmf and position Gaussian distributions.

    Ported from GeodesicMotionSkills/Experiments/toy_example.py::log_prob, using the
    model's own pos_dof/qua_dof instead of module-level globals.
    """
    pos_dof = model.pos_dof
    log_p = torch.mean(positional_dist.log_prob(x[:, :pos_dof]), dim=1)
    log_q = torch.mean(quaternion_dist.log_prob(x[:, pos_dof:]), dim=0)
    log_q_n = torch.mean(quaternion_dist_n.log_prob(x[:, pos_dof:]), dim=0)
    log_q = torch.log(torch.clamp((torch.exp(log_q) + torch.exp(log_q_n)) / 2, 1e-10, np.inf))
    log_likelihood = log_p + (model.quaternion_log_scale * log_q)
    return log_likelihood


def loss_function_elbo(x, model, train_rbf, n_samples):
    """Evidence lower bound loss, ported from toy_example.py::loss_function_elbo."""
    q, _ = model.encode(x, train_rbf=train_rbf)

    z = q.rsample(torch.Size([n_samples]))  # (n_samples)x(batch size)x(latent dim)
    px_z, px_qua_z, px_qua_z_n = model.decode(z, train_rbf=train_rbf, negative=True)  # p(x|z)

    log_p_negative = log_prob(model, x, px_z, px_qua_z, px_qua_z_n)  # vMF(x|mu(z), k(z))
    log_p_negative = torch.mean(log_p_negative) * 1

    log_p = log_p_negative
    kl = torch.tensor([0.0])
    if model.activate_KL:
        log_p = log_p_negative
        kl = -0.5 * torch.sum(1 + q.variance.log() - q.mean.pow(2) - q.variance) * model.kl_coeff
        kl = kl * 400000000
        elbo = torch.mean(log_p - kl, dim=0)
    else:
        elbo = torch.mean(log_p, dim=0)
    log_mean = torch.mean(log_p, dim=0)
    return -elbo, kl, log_mean


def train(model, optimizer, loss_function, data_loader, epoch, epochs, device, train_rbf):
    """One training epoch, ported from toy_example.py::train. Returns the mean batch loss."""
    model.train()
    total_loss, num_batches = 0.0, 0
    for batch_idx, (data,) in enumerate(data_loader):
        data = data.to(device)
        # prevent crashing when the leftover training data is not enough for an epoch
        if data.shape[0] != model.batch_size:
            break
        optimizer.zero_grad()
        batch_loss, loss_kl, loss_log = loss_function(data, train_rbf)
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()
        num_batches += 1
    return total_loss / num_batches if num_batches else float('nan')


def evaluate_elbo(model, data_loader, device, n_samples=1):
    """Mean negative ELBO of the fully trained model (RBF variances included) over a loader.

    The evaluation counterpart of toy_example.py::validate_std.
    """
    was_training = model.training
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data, in data_loader:
            loss, _, _ = loss_function_elbo(data.to(device), model, train_rbf=True, n_samples=n_samples)
            total_loss += loss.item()
    model.train(was_training)
    return total_loss / max(len(data_loader), 1)


def train_vae(vae_cfg, dataset_cfg, device='cpu', seed=None, val_ratio=0.3, overwrite=False,
              curves_dir=None):
    """
    Config-driven replacement for toy_example.py's train_model(): trains a single VAE
    (no encoder_scale sweep, no repetitions loop) in the same 3 stages as the original.
    """
    arch = vae_cfg['architecture']
    artifacts = vae_cfg['training_artifacts']
    training = vae_cfg.get('training', {})

    epochs = int(training.get('epochs', 1000))
    epochs_rbf = int(training.get('epochs_rbf', 1000))
    learning_rate = float(training.get('learning_rate', 1e-3))
    learning_rate_rbf = float(training.get('learning_rate_rbf', 1e-4))
    n_samples = int(training.get('n_samples', 1))
    batch_size = artifacts['batch_size']

    # The shipped checkpoints are tracked in git; check before spending hours training
    model_path = artifacts['model_path']
    if os.path.exists(model_path) and not overwrite:
        raise FileExistsError(f"{model_path} already exists. Pass --overwrite to replace it, "
                              f"or --artifacts_dir to train into another folder.")

    print("\n--- Loading raw demonstrations for VAE training ---")
    trajectories = load_vae_trajectories(dataset_cfg)

    train_loader, val_loader, train_tensor = build_point_dataset(
        trajectories, batch_size=batch_size, test_size=val_ratio, seed=seed)
    print(f"VAE dataset | Train: {train_tensor.shape[0]} points | Val: {len(val_loader.dataset)} points")

    model = VAE(
        layers=arch['layers'],
        batch_size=batch_size,
        pos_dof=arch['pos_dof'],
        qua_dof=arch['qua_dof'],
        sigma=float(arch.get('sigma', 1e-6)),
        sigma_z=float(arch['sigma_z']),
    ).to(device)

    history = {"stage1_kl": [], "stage2_reconstruction": []}

    # --- Stage 1: regularization-focused training (KL active) ---
    model.activate_KL = True
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = lambda data, train_rbf: loss_function_elbo(data, model, train_rbf, n_samples=n_samples)
    for epoch in tqdm(range(epochs), desc="Stage 1/3: KL regularization"):
        history["stage1_kl"].append(
            train(model, optimizer, loss_function, train_loader, epoch, epochs, device, train_rbf=False))

    # --- Stage 2: reconstruction-focused training (decoder only) ---
    model.activate_KL = False
    model.kl_coeff = 0.1
    params = list(model.decoder_loc.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    for epoch in tqdm(range(epochs), desc="Stage 2/3: reconstruction"):
        if epoch == int(epochs / 2):
            model.empowered_quaternions = True
        history["stage2_reconstruction"].append(
            train(model, optimizer, loss_function, train_loader, epoch, epochs, device, train_rbf=False))
    model.empowered_quaternions = False

    # --- Stage 3: train RBF/variance networks ---
    # KMeans inside init_std draws from numpy's global RNG, which the caller seeds
    os.makedirs(os.path.dirname(artifacts['cluster_path']) or '.', exist_ok=True)
    model.init_std(
        train_tensor.to(device),
        load_clusters=False,
        cluster_path=artifacts['cluster_path'],
        beta_scale=arch.get('beta_scale', 1.0),
    )
    # fit_std takes one optimizer step per epoch, on gradients accumulated over every
    # batch (as toy_example.py did), so it is effectively full-batch gradient descent
    model.fit_std(train_loader, epochs_rbf, model, n_samples=n_samples, learning_rate=learning_rate_rbf)

    history["final_train_elbo_loss"] = evaluate_elbo(model, train_loader, device, n_samples)
    history["final_val_elbo_loss"] = evaluate_elbo(model, val_loader, device, n_samples)
    print(f"Final -ELBO | Train: {history['final_train_elbo_loss']:.4e} | "
          f"Val: {history['final_val_elbo_loss']:.4e}")

    # --- Save the trained model ---
    print(f"Saving VAE model: {model_path}")
    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.to('cpu').state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'encoder_scale': arch['sigma_z'],
        # nnj.RBF keeps beta outside the state dict; load_pretrained_vae restores it from here
        'rbf_beta': float(model.dec_std_qua[0].beta.flatten()[0]),
        'history': history,
        'seed': seed,
    }, model_path)

    if curves_dir:
        name = os.path.splitext(os.path.basename(model_path))[0]
        plot_vae_training_curves(history, save_dir=curves_dir, filename=f"{name}_Training-Curves.svg")

    return model, history
