import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from node.data.preprocessing import (
    get_demonstrations_paths,
    get_demonstrations_paths_newVAE,
    get_demonstrations_paths_lerobot,
)


def load_vae_trajectories(dataset_cfg):
    """Load the raw demonstrations (each [T, dof]) the VAE is trained on.

    Every loader returns two copies per demo, [+q, -q] (the quaternion double-cover
    augmentation), so demo i sits at index 2*i.
    """
    dataset_type = dataset_cfg.get('type', 'toy')
    if dataset_type == 'toy':
        return get_demonstrations_paths(
            origin_dir=dataset_cfg['origin_dir'],
            trajectory_number=dataset_cfg['trajectory_number'],
            test_id=dataset_cfg['test_id'],
            s2_letter=dataset_cfg['s2_letter'],
            r2_letter=dataset_cfg['r2_letter'],
        )
    elif dataset_type == 'lasa':
        return get_demonstrations_paths_newVAE(
            origin_dir=dataset_cfg['origin_dir'],
            origin_file=dataset_cfg['origin_file'],
            trajectory_number=dataset_cfg['trajectory_number'],
        )
    elif dataset_type == 'lerobot':
        return get_demonstrations_paths_lerobot(
            repo_id=dataset_cfg['repo_id'],
            euler_order=dataset_cfg['euler_order'],
            n_points=dataset_cfg['n_points'],
        )
    raise ValueError(f"Unknown dataset type for VAE training: '{dataset_type}'")


def load_vae_training_points(dataset_cfg):
    """All VAE training points as one [N, dof] tensor, for load_pretrained_vae.

    Training computed the RBF beta on a random 70% split of these points
    (build_point_dataset); using all of them changes the latent std by well under 1%.
    """
    return torch.from_numpy(np.vstack(load_vae_trajectories(dataset_cfg))).float()


def select_test_demo(trajectories, test_id):
    """The +q copy of demo `test_id` from load_vae_trajectories' [+q, -q] pairs."""
    num_demos = len(trajectories) // 2
    if not 0 <= test_id < num_demos:
        raise ValueError(f"test_id {test_id} out of range: only {num_demos} demos loaded")
    return np.asarray(trajectories[2 * test_id])


def build_point_dataset(trajectories, batch_size, test_size=0.3, seed=None):
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
    train_data, test_data = train_test_split(dataset, test_size=test_size, random_state=seed)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    train_tensor = torch.stack([item[0] for item in train_data])
    return train_loader, test_loader, train_tensor
