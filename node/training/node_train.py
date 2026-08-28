import torch
import torch.optim as optim
import time
import os
import numpy as np

from vae.vae_model import get_M


# --- SCHEDULER based on Epochs ---
def get_loss_weights_riemannianmse(epoch, base_scale=1e-4, ramp_start=500, ramp_length=2500,
                                   w_imit_target=1.0, w_goal_start=1.0, w_goal_target=50.0, w_ener_target=1.0):
    """
    base_scale: The normalization factor to counteract the massive size
    of the metric tensor G. If your clamped G peaks around 1e4, base_scale
    should be around 1e-4 so the total loss stays close to 1.0.
    """

    # Stage 1: Local Flow
    # The model learns the raw paths with no goal pressure or energy tension.
    if epoch < ramp_start:
        return {
            "w_imit": w_imit_target * base_scale,
            "w_goal": 0.0,
            "w_ener": 0.0
        }

    # Stage 2: Combined Geometric Awareness & Goal Precision
    else:
        # Ramp
        # alpha = (epoch - ramp_start) / rest_epochs
        alpha = min((epoch - ramp_start) / float(ramp_length), 1.0)

        # Calculate the dynamic goal multiplier (e.g., starts at 1, goes to 50)
        current_goal_mult = w_goal_start + ((w_goal_target - w_goal_start) * alpha)

        return {
            # Keep imitation constant at the base scale
            "w_imit": w_imit_target * base_scale,
            # Goal ramps from 1x to 5x of the imitation weight
            "w_goal": current_goal_mult * base_scale,
            # Energy requires aggressive downscaling.
            "w_ener": w_ener_target * alpha
       }

# --- Training Loop with the Riemannian MSE (Matching the Target Sequence) Strategy
def train_node_energy_goal_imitation_riemannianmse(model, vae, train_loader, val_loader, node_cfg, device='cpu'):
    # --- 1. Dynamic Config Extraction ---
    METRIC_SCALE_ENERGY = node_cfg['training']['metric_scale_energy']
    SAFE_BASELINE = node_cfg['training']['safe_baseline']
    lr = node_cfg['training'].get('learning_rate', 1e-4)
    lr_gamma = node_cfg['training'].get('lr_gamma', 1e-1)
    drop_fraction = node_cfg['training'].get('lr_drop_fraction', 0.35)
    epochs = node_cfg['training'].get('epochs', 4000)
    # Extract countermeasures parameters against exploding energy
    clamp_min = node_cfg["training"]["metric_clamp_min"]
    clamp_max = node_cfg["training"]["metric_clamp_max"]
    clip_norm = node_cfg["training"]["grad_clip_norm"]
    # Extract scheduler variables
    sched_cfg = node_cfg['training'].get('scheduler', {})
    base_scale = sched_cfg.get('base_scale', 1e-4)
    ramp_start = sched_cfg.get('ramp_start', 500)
    ramp_length = sched_cfg.get('ramp_length', 2500)
    # Extract the target weights!
    w_imit_target = sched_cfg.get('w_imit_target', 1.0)
    w_goal_start = sched_cfg.get('w_goal_start', 1.0)
    w_goal_target = sched_cfg.get('w_goal_target', 50.0)
    w_ener_target = sched_cfg.get('w_ener_target', 1.0)
    # Generate dynamic names based on the shape
    model_name = node_cfg['dataset']['model_name']

    # Save the models next to where the processed data lives
    save_dir = node_cfg['dataset'].get('model_dir', './node/data').replace('_processed', '_models')
    os.makedirs(save_dir, exist_ok=True)

    # --- 2. Setup Optimizer & Schedulers ---
    optimizer = optim.Adam(model.parameters(), lr=lr)
    vae.to(device).eval()
    model.to(device)

    # Dynamic milestone: Drop LR at 35% of total epochs
    drop_epoch = int(epochs * drop_fraction)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[drop_epoch], gamma=lr_gamma)

    start_time = time.time()
    print(f"\n{'=' * 140}")
    print(f"Starting Training: {model_name} on {device.upper()}")
    print(f"Epochs: {epochs} | Initial LR: {lr} | LR Drop at: {drop_epoch}")
    print(f"{'=' * 140}\n")

    history = {
        "train_imit": [], "train_ener": [], "train_goal": [], "train_loss": [],
        "val_imit": [], "val_ener": [], "val_goal": [], "val_loss": []
    }

    avg_val_imit, avg_val_ener, avg_val_goal, avg_val_total = 0.0, 0.0, 0.0, 0.0

    # --- 3. Main Training Loop ---
    for epoch in range(epochs):
        model.train()
        epoch_imitation_loss, epoch_energy_loss, epoch_goal_loss, epoch_total_train_loss = 0, 0, 0, 0
        epoch_euclidean_imit = 0

        w = get_loss_weights_riemannianmse(
            epoch, base_scale, ramp_start, ramp_length,
            w_imit_target, w_goal_start, w_goal_target, w_ener_target
        )
        w_imitation, w_goal, w_energy = w['w_imit'], w['w_goal'], w['w_ener']

        if epoch == drop_epoch:
            print(f"📉 Epoch {drop_epoch}: Learning Rate dropped to {lr * lr_gamma}")

        for p1, p2, z_target in train_loader:
            p1, p2, z_target = p1.to(device), p2.to(device), z_target.to(device)
            B, T_steps, D = z_target.shape

            optimizer.zero_grad()

            # PREDICTION
            t_steps = torch.linspace(0, 1, T_steps).to(device)
            pred_traj = model(p1, p2, t_steps)
            z_pred = pred_traj[:, :, 0, :]
            v_pred = pred_traj[:, :, 1, :]
            y_pred = z_pred.permute(1, 0, 2)

            # LOSS CALCULATION
            with torch.no_grad():
                G_raw_true = get_M(vae, z_target.reshape(-1, D).clone())[2]
                # FIX: Removed torch.tensor() to prevent warnings, using clone().detach() instead
                if not isinstance(G_raw_true, torch.Tensor):
                    G_raw_true = torch.tensor(G_raw_true, dtype=torch.float32)
                G_true_clamped = torch.clamp(G_raw_true.clone().detach(), min=clamp_min, max=clamp_max).to(device)
                G_true = G_true_clamped.reshape(B, T_steps, D, D)

            delta_imit = y_pred - z_target
            squared_riem_dist = torch.einsum('bni,bnij,bnj->bn', delta_imit, G_true, delta_imit)
            loss_imitation = torch.mean(squared_riem_dist)

            euclidean_imit = torch.mean((y_pred.detach() - z_target) ** 2).item()

            if w_energy > 0:
                G_raw = get_M(vae, (z_pred.reshape(-1, D)).clone())[2]
                if not isinstance(G_raw, torch.Tensor):
                    G_raw = torch.tensor(G_raw, dtype=torch.float32)
                G_clamped = torch.clamp(G_raw.clone().detach(), min=clamp_min, max=clamp_max).to(device)
                G = G_clamped.reshape(T_steps, B, D, D)

                energy_per_step = torch.einsum('tb i, tb ij, tb j -> tb', v_pred, G, v_pred)
                loss_energy = energy_per_step.mean() * METRIC_SCALE_ENERGY
            else:
                loss_energy = torch.tensor(0.0).to(device)

            pred_goal = z_pred[-1]
            with torch.no_grad():
                G_raw_goal = get_M(vae, p2.clone())[2]
                if not isinstance(G_raw_goal, torch.Tensor):
                    G_raw_goal = torch.tensor(G_raw_goal, dtype=torch.float32)
                G_goal_clamped = torch.clamp(G_raw_goal.clone().detach(), min=clamp_min, max=clamp_max).to(device)
                G_goal = G_goal_clamped.reshape(B, D, D)

            delta_goal = pred_goal - p2
            loss_goal = torch.mean(torch.einsum('bi,bij,bj->b', delta_goal, G_goal, delta_goal))

            total_loss = (w_imitation * loss_imitation) + (w_goal * loss_goal) + (w_energy * loss_energy)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()

            epoch_imitation_loss += loss_imitation.item()
            epoch_energy_loss += loss_energy.item()
            epoch_goal_loss += loss_goal.item()
            epoch_total_train_loss += total_loss.item()
            epoch_euclidean_imit += euclidean_imit

        # --- Validation & Dashboard (Every 50 Epochs) ---
        if (epoch == 0 or (epoch + 1) % 50 == 0 or epoch == epochs - 1):
            model.eval()
            val_imit, val_ener, val_goal, val_euclidean_imit = 0.0, 0.0, 0.0, 0.0

            with torch.no_grad():
                for p1, p2, z_target in val_loader:
                    p1, p2, z_target = p1.to(device), p2.to(device), z_target.to(device)
                    B_val, T_val, D_val = z_target.shape

                    t_steps = torch.linspace(0, 1, T_val).to(device)
                    pred_traj = model(p1, p2, t_steps)
                    z_pred, v_pred = pred_traj[:, :, 0, :], pred_traj[:, :, 1, :]
                    y_pred_val = z_pred.permute(1, 0, 2)

                    # Validation Imitation
                    G_raw_val_true = get_M(vae, z_target.reshape(-1, D_val).clone())[2]
                    if not isinstance(G_raw_val_true, torch.Tensor):
                        G_raw_val_true = torch.tensor(G_raw_val_true, dtype=torch.float32)
                    G_val_true = torch.clamp(G_raw_val_true.clone().detach(), min=clamp_min, max=clamp_max).to(device).reshape(B_val, T_val,
                                                                                                          D_val, D_val)
                    del_val_imit = y_pred_val - z_target
                    val_imit += torch.mean(
                        torch.einsum('bni,bnij,bnj->bn', del_val_imit, G_val_true, del_val_imit)).item()
                    val_euclidean_imit += torch.mean((y_pred_val - z_target) ** 2).item()

                    # Validation Energy
                    G_raw_val_pred = get_M(vae, z_pred.reshape(-1, D_val).clone())[2]
                    if not isinstance(G_raw_val_pred, torch.Tensor):
                        G_raw_val_pred = torch.tensor(G_raw_val_pred, dtype=torch.float32)
                    G_val_pred = torch.clamp(G_raw_val_pred.clone().detach(), min=clamp_min, max=clamp_max).to(
                        device).reshape(B_val, T_val, D_val, D_val)

                    val_ener += (torch.einsum('tb i, tb ij, tb j -> tb', v_pred, G_val_pred,
                                              v_pred).mean().item()) * METRIC_SCALE_ENERGY

                    # Validation Goal
                    G_raw_val_goal = get_M(vae, p2.clone())[2]
                    if not isinstance(G_raw_val_goal, torch.Tensor):
                        G_raw_val_goal = torch.tensor(G_raw_val_goal, dtype=torch.float32)
                    G_val_goal = torch.clamp(G_raw_val_goal.clone().detach(), min=clamp_min, max=clamp_max).to(device).reshape(B_val, D_val,
                                                                                                          D_val)
                    del_val_goal = z_pred[-1] - p2
                    val_goal += torch.mean(torch.einsum('bi,bij,bj->b', del_val_goal, G_val_goal, del_val_goal)).item()

                n_val = max(len(val_loader), 1)
                avg_val_imit = val_imit / n_val
                avg_val_goal = val_goal / n_val
                avg_val_ener = val_ener / n_val
                avg_val_total = (w_imitation * avg_val_imit) + (w_goal * avg_val_goal) + (w_energy * avg_val_ener)

                avg_train_euclid = epoch_euclidean_imit / len(train_loader)
                avg_val_euclid = val_euclidean_imit / n_val

            print(f"Epoch {epoch + 1} |")
            print(
                f"| Train | Tot: {epoch_total_train_loss / len(train_loader):.5f} | Imit: {epoch_imitation_loss / len(train_loader):.5f} | Ener: {epoch_energy_loss / len(train_loader):.5f} | Goal: {epoch_goal_loss / len(train_loader):.5f} |")
            print(
                f"| Valid | Tot: {avg_val_total:.5f} | Imit: {avg_val_imit:.5f} | Ener: {avg_val_ener:.5f} | Goal: {avg_val_goal:.5f} |")

            # Generalization Gaps
            gap = abs((epoch_imitation_loss / len(train_loader)) - avg_val_imit)
            print(f"{'✅' if gap < 0.01 else '🟡' if gap < 0.05 else '⚠️'} Riem. Gap: {gap:.5f}")

            euclid_gap = abs(avg_train_euclid - avg_val_euclid)
            print(f"{'✅' if euclid_gap < 0.10 else '🟡' if euclid_gap < 0.30 else '⚠️'} Euclid Gap: {euclid_gap:.5f}")

            if w_energy > 0:
                with torch.no_grad():
                    mean_density = torch.diagonal(G, dim1=-2, dim2=-1).sum(-1).mean().item() / 2.0
            else:
                mean_density = 0.0

            relative_cost = mean_density / SAFE_BASELINE

            print(
                f"🚩 DASHBOARD: Weights (Imit: {w_imitation:.4e}, Goal: {w_goal:.4e}, Ener: {w_energy:.2e}) | Density: {mean_density:,.2f} ({relative_cost:.2f}x base)\n{'-' * 140}")

        scheduler.step()

        history["train_imit"].append(epoch_imitation_loss / len(train_loader))
        history["train_ener"].append(epoch_energy_loss / len(train_loader))
        history["train_goal"].append(epoch_goal_loss / len(train_loader))
        history["train_loss"].append(epoch_total_train_loss / len(train_loader))
        history["val_imit"].append(avg_val_imit)
        history["val_ener"].append(avg_val_ener)
        history["val_goal"].append(avg_val_goal)
        history["val_loss"].append(avg_val_total)

    end_time = time.time()
    print(f"Training Complete in {(end_time - start_time) / 60:.2f} minutes.")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved dynamically to {save_path}")

    return model, history
