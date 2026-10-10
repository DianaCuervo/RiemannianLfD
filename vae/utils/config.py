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


def beta_label(beta_scale):
    """The {beta} of the file names for a beta_scale: 10, 10.0 and '10' all give '10'."""
    return f"{float(beta_scale):g}"


def load_vae_config(dataset, shape=None, task=None, beta=None):
    """The VAE config for one dataset (and lerobot task), with placeholders resolved.

    The '{beta}' of the file names comes from architecture.beta_scale when the config sets
    it to a number (toy: select it in vae_config.yaml only). A config that leaves it as
    '{beta}' (lasa) takes it from `beta` instead (the --beta_scale CLI argument).
    """
    with open(VAE_CONFIG_PATH, 'r') as file:
        full_config = yaml.safe_load(file)
    vae_cfg = _select(full_config, dataset, task, VAE_CONFIG_PATH)

    config_beta = vae_cfg['architecture'].get('beta_scale')
    if config_beta is not None and '{beta}' not in str(config_beta):
        if beta is not None and beta_label(beta) != beta_label(config_beta):
            raise ValueError(f"The {dataset} beta_scale is selected in {VAE_CONFIG_PATH} ({config_beta}), "
                             f"not on the command line: drop --beta_scale {beta} or edit the config")
        beta = config_beta
    vae_cfg = _substitute(vae_cfg, _placeholders(dataset, shape, task,
                                                 beta_label(beta) if beta is not None else None))

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
