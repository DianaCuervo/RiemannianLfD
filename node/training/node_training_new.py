import torch
import torch.optim as optim
import time
import os

from vae.vae_model import get_M

# --- SCHEDULER based on Epochs ---
def get_loss_weights_riemannianmse(epoch, base_scale=1e-4, ramp_start=500, ramp_length=2500,
                                   w_imit_target=1.0, w_goal_start=1.0, w_goal_target=50.0, w_ener_target=1.0,
                                   enabled=True):
    """
    base_scale: The normalization factor to counteract the massive size
    of the metric tensor G. If your clamped G peaks around 1e4, base_scale
    should be around 1e-4 so the total loss stays close to 1.0.
    enabled: If False, skip the ramp and use the target weights from epoch 0.
    """

    # Constant weights (no scheduling): ramp_start, ramp_length and w_goal_start are ignored
    if not enabled:
        return {
            "w_imit": w_imit_target * base_scale,
            "w_goal": w_goal_target * base_scale,
            "w_ener": w_ener_target
        }

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

# --- ENERGY in the embedding space (stochman's EmbeddedManifold.curve_energy strategy) ---
def embedded_curve_energy(vae, z_traj):
    """
    Discrete curve energy measured after decoding the curve:
        E = (T-1) * sum_t ||f(z_{t+1}) - f(z_t)||^2  ~=  mean_t v_t^T M(z_t) v_t

    Unlike ||J(z) v||^2 with J computed under no_grad, the gradient flows through the decoder f,
    so it contains dM/dz and pulls the curve towards low-metric regions (true geodesic energy).
    No Jacobian or second derivatives are needed: one decoder forward and backward pass.

    z_traj: [T, B, D] latent trajectory. Returns the energy per curve, shape [B].
    """
    T_steps, B, D = z_traj.shape
    # vae.embed only supports a leading batch of 1 (as in get_M): embed all B*T points at once, then split curves
    points = z_traj.permute(1, 0, 2).reshape(1, B * T_steps, D)
    emb = vae.embed(points).reshape(B, T_steps, -1)  # BxTxK
    delta = emb[:, 1:, :] - emb[:, :-1, :]  # Bx(T-1)xK
    return (delta ** 2).sum(dim=(1, 2)) * (T_steps - 1)

# --- Training Loop with the Riemannian MSE + Embedded (decoder) Energy Strategy
def train_node_energy_goal_imitation_embedded(model, vae, train_loader, val_loader, node_cfg, device='cpu'):
    # --- 1. Dynamic Config Extraction ---
    METRIC_SCALE_ENERGY = node_cfg['training']['metric_scale_energy']
    SAFE_BASELINE = node_cfg['training']['safe_baseline']
    lr = node_cfg['training'].get('learning_rate', 1e-4)
    lr_gamma = node_cfg['training'].get('lr_gamma', 1e-1)
    drop_fraction = node_cfg['training'].get('lr_drop_fraction', 0.35)
    epochs = node_cfg['training'].get('epochs', 4000)
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
    sched_enabled = sched_cfg.get('enabled', True)
    # Generate dynamic names based on the shape
    model_name = node_cfg['dataset']['model_name']

    # Save the models next to where the processed data lives
    save_dir = node_cfg['dataset'].get('model_dir', './node/data').replace('_processed', '_models')
    os.makedirs(save_dir, exist_ok=True)

    # --- 2. Setup Optimizer & Schedulers ---
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Frozen VAE: eval mode (BatchNorm running stats) and no gradients accumulated in its weights,
    # but gradients still flow through it to the NODE trajectory for the energy term
    vae.to(device).eval()
    vae.requires_grad_(False)
    model.to(device)

    # Dynamic milestone: Drop LR at drop_fraction of total epochs
    drop_epoch = int(epochs * drop_fraction)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[drop_epoch], gamma=lr_gamma)

    start_time = time.time()
    print(f"\n{'=' * 140}")
    print(f"Starting Training (embedded energy): {model_name} on {device.upper()}")
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
            w_imit_target, w_goal_start, w_goal_target, w_ener_target, sched_enabled
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
            y_pred = z_pred.permute(1, 0, 2)

            # LOSS CALCULATION
            # --- IMITATION LOSS --- (metric at the fixed targets: no gradient through J needed)
            with torch.no_grad():
                _, J_raw_true, _ = get_M(vae, z_target.reshape(-1, D).clone())
                J_raw_true = torch.as_tensor(J_raw_true, dtype=torch.float32, device=device)
                J_true = J_raw_true.reshape(B, T_steps, -1, D)

            delta_imit = y_pred - z_target
            J_delta_imit = torch.einsum('bnkd,bnd->bnk', J_true, delta_imit)
            loss_imitation = torch.mean(torch.sum(J_delta_imit ** 2, dim=-1))

            euclidean_imit = torch.mean((y_pred.detach() - z_target) ** 2).item()

            # --- ENERGY LOSS --- (gradient flows through the decoder, i.e. through M(z))
            if w_energy > 0:
                loss_energy = embedded_curve_energy(vae, z_pred).mean() * METRIC_SCALE_ENERGY
            else:
                loss_energy = torch.tensor(0.0).to(device)

            # --- GOAL LOSS ---
            pred_goal = z_pred[-1]
            with torch.no_grad():
                _, J_raw_goal, _ = get_M(vae, p2.clone())
                J_raw_goal = torch.as_tensor(J_raw_goal, dtype=torch.float32, device=device)
                J_goal = J_raw_goal.reshape(B, -1, D)

            delta_goal = pred_goal - p2
            J_delta_goal = torch.einsum('bkd,bd->bk', J_goal, delta_goal)
            loss_goal = torch.mean(torch.sum(J_delta_goal ** 2, dim=-1))

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
            val_density, n_val_points = 0.0, 0

            with torch.no_grad():
                for p1, p2, z_target in val_loader:
                    p1, p2, z_target = p1.to(device), p2.to(device), z_target.to(device)
                    B_val, T_val, D_val = z_target.shape

                    t_steps = torch.linspace(0, 1, T_val).to(device)
                    pred_traj = model(p1, p2, t_steps)
                    z_pred = pred_traj[:, :, 0, :]
                    y_pred_val = z_pred.permute(1, 0, 2)

                    # Validation Imitation
                    _, J_raw_val_true, _ = get_M(vae, z_target.reshape(-1, D_val).clone())
                    J_raw_val_true = torch.as_tensor(J_raw_val_true, dtype=torch.float32, device=device)
                    J_val_true = J_raw_val_true.reshape(B_val, T_val, -1, D_val)

                    del_val_imit = y_pred_val - z_target
                    J_del_val_imit = torch.einsum('bnkd,bnd->bnk', J_val_true, del_val_imit)
                    val_imit += torch.mean(torch.sum(J_del_val_imit ** 2, dim=-1)).item()

                    val_euclidean_imit += torch.mean((y_pred_val - z_target) ** 2).item()

                    # Validation Energy (same embedded energy as training)
                    val_ener += embedded_curve_energy(vae, z_pred).mean().item() * METRIC_SCALE_ENERGY

                    # Density along the predicted paths (trace(M)/2), averaged over the whole validation set
                    _, J_raw_val_pred, _ = get_M(vae, z_pred.reshape(-1, D_val).clone())
                    J_raw_val_pred = torch.as_tensor(J_raw_val_pred, dtype=torch.float32, device=device)
                    val_density += (J_raw_val_pred ** 2).sum(dim=(-2, -1)).sum().item() / 2.0
                    n_val_points += J_raw_val_pred.shape[0]

                    # Validation Goal
                    _, J_raw_val_goal, _ = get_M(vae, p2.clone())
                    J_raw_val_goal = torch.as_tensor(J_raw_val_goal, dtype=torch.float32, device=device)
                    J_val_goal = J_raw_val_goal.reshape(B_val, -1, D_val)

                    del_val_goal = z_pred[-1] - p2
                    J_del_val_goal = torch.einsum('bkd,bd->bk', J_val_goal, del_val_goal)
                    val_goal += torch.mean(torch.sum(J_del_val_goal ** 2, dim=-1)).item()

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

            mean_density = val_density / max(n_val_points, 1)
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

    save_path = os.path.join(save_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved dynamically to {save_path}")

    return model, history
