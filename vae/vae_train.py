import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from node.data.preprocessing import (
    get_demonstrations_paths,
    get_demonstrations_paths_newVAE,
    get_demonstrations_paths_lerobot,
)
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
    """One training epoch, ported from toy_example.py::train."""
    model.train()
    for batch_idx, (data,) in enumerate(data_loader):
        data = data.to(device)
        # prevent crashing when the leftover training data is not enough for an epoch
        if data.shape[0] != model.batch_size:
            break
        optimizer.zero_grad()
        batch_loss, loss_kl, loss_log = loss_function(data, train_rbf)
        batch_loss.backward()
        optimizer.step()


def build_point_dataset(trajectories, batch_size, test_size=0.3):
    """
    Flattens a list of raw demo trajectories (each [T, dof]) into a single point-cloud
    dataset, matching the flatten+TensorDataset+train_test_split done inline in
    toy_example.py's __main__ block.

    Note: sklearn's train_test_split on a TensorDataset returns a plain list of
    (tensor,)-tuples, not a TensorDataset, so we also hand back the raw stacked
    training tensor for callers (e.g. init_std) that need a batch tensor directly.
    """
    input_data = np.vstack(trajectories)
    dataset = TensorDataset(torch.from_numpy(input_data).float())
    train_data, test_data = train_test_split(dataset, test_size=test_size)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    train_tensor = torch.stack([item[0] for item in train_data])
    return train_loader, test_loader, train_tensor


def train_vae(vae_cfg, dataset_cfg, device='cpu'):
    """
    Config-driven replacement for toy_example.py's train_model(): trains a single VAE
    (no encoder_scale sweep, no repetitions loop) in the same 3 stages as the original.
    """
    arch = vae_cfg['architecture']
    artifacts = vae_cfg['training_artifacts']
    training = vae_cfg.get('training', {})

    epochs = training.get('epochs', 1000)
    epochs_rbf = training.get('epochs_rbf', 1000)
    learning_rate = training.get('learning_rate', 1e-3)
    n_samples = training.get('n_samples', 1)
    batch_size = artifacts['batch_size']

    print("\n--- Loading raw demonstrations for VAE training ---")
    dataset_type = dataset_cfg.get('type', 'toy')
    if dataset_type == 'toy':
        trajectories = get_demonstrations_paths(
            origin_dir=dataset_cfg['origin_dir'],
            trajectory_number=dataset_cfg['trajectory_number'],
            test_id=dataset_cfg['test_id'],
            s2_letter=dataset_cfg['s2_letter'],
            r2_letter=dataset_cfg['r2_letter'],
        )
    elif dataset_type == 'lasa':
        trajectories = get_demonstrations_paths_newVAE(
            origin_dir=dataset_cfg['origin_dir'],
            origin_file=dataset_cfg['origin_file'],
            trajectory_number=dataset_cfg['trajectory_number'],
        )
    elif dataset_type == 'lerobot':
        trajectories = get_demonstrations_paths_lerobot(
            repo_id=dataset_cfg['repo_id'],
            euler_order=dataset_cfg['euler_order'],
            n_points=dataset_cfg['n_points'],
        )
    else:
        raise ValueError(f"Unknown dataset type for VAE training: '{dataset_type}'")

    train_loader, test_loader, train_tensor = build_point_dataset(trajectories, batch_size=batch_size)

    model = VAE(
        layers=arch['layers'],
        batch_size=batch_size,
        pos_dof=arch['pos_dof'],
        qua_dof=arch['qua_dof'],
        sigma=float(arch.get('sigma', 1e-6)),
        sigma_z=float(arch['sigma_z']),
    ).to(device)

    # --- Stage 1: regularization-focused training (KL active) ---
    model.activate_KL = True
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = lambda data, train_rbf: loss_function_elbo(data, model, train_rbf, n_samples=n_samples)
    for epoch in tqdm(range(int(epochs)), desc="Stage 1/3: KL regularization"):
        train(model, optimizer, loss_function, train_loader, epoch, epochs, device, train_rbf=False)

    # --- Stage 2: reconstruction-focused training (decoder only) ---
    model.activate_KL = False
    model.kl_coeff = 0.1
    params = list(model.decoder_loc.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    for epoch in tqdm(range(int(epochs)), desc="Stage 2/3: reconstruction"):
        if epoch == int(epochs / 2):
            model.empowered_quaternions = True
        train(model, optimizer, loss_function, train_loader, epoch, epochs, device, train_rbf=False)
    model.empowered_quaternions = False

    # --- Stage 3: train RBF/variance networks ---
    model.init_std(
        train_tensor.to(device),
        load_clusters=False,
        cluster_path=artifacts['cluster_path'],
        beta_scale=arch.get('beta_scale', 1.0),
    )
    model.fit_std(train_loader, epochs_rbf, model, n_samples=n_samples)

    # --- Save the trained model ---
    model_path = artifacts['model_path']
    print(f"Saving VAE model: {model_path}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.to('cpu').state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'encoder_scale': arch['sigma_z'],
    }, model_path)

    return model
