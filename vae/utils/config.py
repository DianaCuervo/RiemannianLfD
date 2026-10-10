import copy

import yaml

VAE_CONFIG_PATH = 'config_files/vae_config.yaml'
NODE_CONFIG_PATH = 'config_files/node_config.yaml'


def _placeholders(dataset, shape=None, task=None, beta=None):
    """The {shape}/{task}/{beta} substitutions main_new.py applies, after validating the CLI input."""
    placeholders = {}
    if dataset == 'lasa':
        if not shape:
            raise ValueError("--shape is required for the lasa dataset (e.g. --shape N)")
        placeholders['{shape}'] = f"{shape}-Shape"
    if dataset == 'lerobot':
        if not task:
            raise ValueError("--task is required for the lerobot dataset (e.g. --task pick)")
        placeholders['{task}'] = task
    if beta is not None:
        placeholders['{beta}'] = str(beta)
    return placeholders


def _substitute(value, placeholders):
    """Replace every placeholder in all strings of a nested config."""
    if isinstance(value, dict):
        return {k: _substitute(v, placeholders) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, placeholders) for v in value]
    if isinstance(value, str):
        for key, replacement in placeholders.items():
            value = value.replace(key, replacement)
    return value


def _select(config, dataset, task, file_path):
    selected = config.get(dataset)
    if selected is None:
        raise ValueError(f"Dataset '{dataset}' not found in {file_path}")
    if dataset == 'lerobot':
        if task not in selected:
            raise ValueError(f"Task '{task}' not found under '{dataset}' in {file_path}")
        selected = selected[task]
    return copy.deepcopy(selected)


def load_vae_config(dataset, shape=None, task=None, beta=None):
    """The VAE config for one dataset (and lerobot task), with placeholders resolved.

    beta: the RBF beta_scale (e.g. '1', '5', '10') for configs with one VAE per beta_scale
    (toy, lasa), which fills their '{beta}' paths and architecture.beta_scale.
    """
    with open(VAE_CONFIG_PATH, 'r') as file:
        full_config = yaml.safe_load(file)
    vae_cfg = _select(full_config, dataset, task, VAE_CONFIG_PATH)
    vae_cfg = _substitute(vae_cfg, _placeholders(dataset, shape, task, beta))

    unresolved = [key for key, value in vae_cfg['training_artifacts'].items()
                  if isinstance(value, str) and '{beta}' in value]
    if '{beta}' in str(vae_cfg['architecture'].get('beta_scale', '')):
        unresolved.append('beta_scale')
    if unresolved:
        raise ValueError(f"--beta_scale is required for the {dataset} dataset ('{{beta}}' in {unresolved})")
    if 'beta_scale' in vae_cfg['architecture']:
        vae_cfg['architecture']['beta_scale'] = float(vae_cfg['architecture']['beta_scale'])
    return vae_cfg


def load_dataset_config(dataset, shape=None, task=None):
    """Where the raw demonstrations live: the 'dataset' section of node_config.yaml only.

    Keeps the raw-data location in one place for both pipelines; nothing NODE-specific
    from that file is used.
    """
    with open(NODE_CONFIG_PATH, 'r') as file:
        full_config = yaml.safe_load(file)
    dataset_cfg = _select(full_config, dataset, task, NODE_CONFIG_PATH).get('dataset')
    if dataset_cfg is None:
        raise ValueError(f"No 'dataset' section for '{dataset}' in {NODE_CONFIG_PATH}")
    return _substitute(dataset_cfg, _placeholders(dataset, shape, task))
