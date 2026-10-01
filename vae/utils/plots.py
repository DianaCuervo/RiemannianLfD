import os

import matplotlib.pyplot as plt
import numpy as np

from node.utils.plots import visualize_metric, save_test_plot


def _save_figure(fig, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"📸 Plot saved as SVG to: {save_path}")


### Plot fns for vae training
def plot_vae_training_curves(history, save_dir, filename="VAE_Training-Curves.svg"):
    """Per-epoch training loss of stage 1 (KL) and stage 2 (reconstruction).

    Stage 3 runs inside VAE.fit_std, which reports no per-epoch loss; its result shows up
    as the final -ELBO in the title. symlog because the -ELBO is usually negative.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for ax, key, title in [(ax1, "stage1_kl", "Stage 1: KL regularization"),
                           (ax2, "stage2_reconstruction", "Stage 2: reconstruction")]:
        ax.plot(range(1, len(history[key]) + 1), history[key], color='dodgerblue', linewidth=2)
        ax.set_yscale('symlog')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('-ELBO (train)')
        ax.set_title(title)
        ax.grid(True, which="both", ls="-", alpha=0.2)

    fig.suptitle(f"Final -ELBO | Train: {history['final_train_elbo_loss']:.4e} | "
                 f"Val: {history['final_val_elbo_loss']:.4e}")
    fig.tight_layout()
    _save_figure(fig, save_dir, filename)


### Plot fns for vae testing
def plot_metric_with_latent_points(vae, z_train, space_name, latent_frame, save_dir, filename,
                                   z_demo=None, z_paths=None):
    """node's visualize_metric, overlaid with the encoded training points and, optionally,
    the encoded test demo and planned latent paths."""
    visualize_metric(vae, space_name, latent_frame)
    plt.scatter(z_train[:, 0], z_train[:, 1], s=2, color='black', alpha=0.15, label='Encoded demonstrations')
    if z_demo is not None:
        plt.plot(z_demo[:, 0], z_demo[:, 1], 'k--', linewidth=1.5, label='Test demo')
    for i, z_path in enumerate(z_paths if z_paths is not None else []):
        plt.plot(z_path[:, 0], z_path[:, 1], color='red', linewidth=2, label='Graph geodesic' if i == 0 else None)
    plt.legend(loc='upper right', fontsize='small')
    save_test_plot(save_dir=save_dir, filename=filename)
    plt.close()


def plot_measure_and_mf(measure, mf, latent_frame, z_train, z_paths, save_dir,
                        filename="Measure and Magnification Factor.svg"):
    """Variance measure and magnification factor side by side, with the encoded
    demonstrations and the graph geodesics. Port of toy_example.py::plot.

    measure/mf are [x, y]-indexed grids, hence the rot90 before imshow.
    """
    extent = [-latent_frame, latent_frame, -latent_frame, latent_frame]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, field, title in [(axes[0], measure, 'Variance measure (log)'),
                             (axes[1], mf, 'Magnification factor (log)')]:
        image = ax.imshow(np.rot90(field), interpolation='bicubic', extent=extent)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.scatter(z_train[:, 0], z_train[:, 1], marker=".", color="white", s=200, alpha=0.1,
                   label='Encoded Demonstrations')
        for i, z_path in enumerate(z_paths):
            ax.plot(z_path[:, 0], z_path[:, 1], color="red", label='Graph geodesic' if i == 0 else None)
        ax.set_xlim(-latent_frame, latent_frame)
        ax.set_ylim(-latent_frame, latent_frame)
        ax.set_title(title)
        ax.set_xlabel('z1')
        ax.set_ylabel('z2')
    axes[0].legend(loc='upper right')
    fig.tight_layout()
    _save_figure(fig, save_dir, filename)


def plot_task_space(demo_positions, decoded_paths, segments, save_dir, filename="Task Space Geodesics.svg"):
    """Decoded graph geodesics against the test demo, in the (normalized) task-space xy plane."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(demo_positions[:, 0], demo_positions[:, 1], 'k--', alpha=0.6, label='Test demo')
    for (start, end), positions in zip(segments, decoded_paths):
        line, = ax.plot(positions[:, 0], positions[:, 1], linewidth=2, label=f'Geodesic {start}->{end}')
        ax.scatter(demo_positions[[start, end], 0], demo_positions[[start, end], 1],
                   color=line.get_color(), edgecolors='black', s=60, zorder=5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Decoded graph geodesics vs. test demo')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axis('equal')
    ax.legend(fontsize='small')
    _save_figure(fig, save_dir, filename)
