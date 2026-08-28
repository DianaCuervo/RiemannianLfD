import torch
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Forces the visual popup
import matplotlib.pyplot as plt
import os
import random

from vae.vae_model import get_M


### Basic plot fns for the toy_example dataset
#Plot to check datasets
def plot_trajectories(trajectory_list):
    """
    Plots multiple trajectories.
    Expects a list of tensors, each with shape [N, 2]
    """
    plt.figure(figsize=(10, 8))

    # Loop through each trajectory in the list
    for i, traj in enumerate(trajectory_list):
        # Convert to numpy and move to CPU if necessary
        if torch.is_tensor(traj):
            data = traj.detach().numpy()
        else:
            data = traj

        #print("shape data: " + str(len(data)))
        # Extract x (col 0) and y (col 1)
        x = data[:, 0]
        y = data[:, 1]

        # Plot the curve
        plt.plot(x, y, label=f'Trajectory {i + 1}', linewidth=2)

        # Optional: Mark the start and end points
        plt.scatter(x[0], y[0], color='green', s=30, label="Start")  # Start
        plt.scatter(x[-1], y[-1], color='red', s=30, label="End")  # End

        if i ==0:
            plt.legend()

    plt.title(f'Visualization of {len(trajectory_list)} Trajectories')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')  # Keeps the scale 1:1 so circles look like circles
    plt.show()
#To check trajectories in the dataloaders
def plot_latent_dataloader(dataloader):

    # Get a batch of data
    p1_batch, p2_batch, z_target_batch = next(iter(dataloader))
    num_paths = z_target_batch.shape[0]
    p1, p2, z_target = p1_batch[:num_paths], p2_batch[:num_paths], z_target_batch[:num_paths]

    # Create plot with predetermined size
    plt.figure(figsize=(12, 12))

    for i in range(z_target.shape[0]):

        # 2. Plot Teacher (Robot Demo)
        plt.plot(z_target[i, :, 0].detach().numpy(), z_target[i, :, 1].detach().numpy(), 'k--', label='Robot Demo', alpha=0.6, color='black')

        # 3. Plot Student (NODE Prediction)
        #plt.plot(z_pred[:, i, 0], z_pred[:, i, 1], 'b-', linewidth=2, label='NODE Path', color='red')

        # 4. Markers for Start and Goal
        plt.scatter(p1[i, 0].detach().cpu().numpy(), p1[i, 1].detach().cpu().numpy(), c='green', s=50, label='Start (p1)')
        plt.scatter(p2[i, 0].detach().cpu().numpy(), p2[i, 1].detach().cpu().numpy(), c='red', marker='*', s=200, label='Goal (p2)')

        if i==0:
            plt.legend(fontsize='small')

    plt.title(f"Samples vs Predictions for " + str(num_paths) + " samples")
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')# Keeps the scale 1:1 so circles look like circles
    plt.show()

## Plot fns after interpolation, if required
#Plot to check position interpolation
def verify_pos_plot(original_p, interpolated_p):
    plt.figure(figsize=(8, 8))

    # Plot 200 points as blue circles
    plt.scatter(original_p[:, 0], original_p[:, 1], c='blue', label='Original (200)', alpha=0.3, s=10)

    # Plot 500 points as a red line
    plt.plot(interpolated_p[:, 0], interpolated_p[:, 1], c='red', label='Interpolated (500)', linewidth=1)

    plt.legend()
    plt.title("Position Interpolation Check")
    plt.axis('equal')  # Keep it square!
    plt.show()
#Plot to check rotations interpolation
def verify_rot_plot(original_rot, interpolated_rot):
    """
    original_p: [200, 3] tensor (Original)
    interpolated_p: [500, 3] tensor (Interpolated)
    """

    # Convert to numpy for matplotlib
    rOri = original_rot#.detach().cpu().numpy()
    rInt = interpolated_rot#.detach().cpu().numpy()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Plot the original 200 points (as dots to see spacing)
    ax.scatter(rOri[:, 0], rOri[:, 1], rOri[:, 2],
               color='blue', label='Original (200 pts)', s=20, alpha=0.5)

    # 2. Plot the interpolated 500 points (as a solid line to see smoothness)
    ax.plot(rInt[:, 0], rInt[:, 1], rInt[:, 2],
            color='red', label='Interpolated (500 pts)', linewidth=2)

    # 3. Add a wireframe sphere of radius 1 (the boundary of the unit ball)
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="grey", alpha=0.1)

    # Labels
    ax.set_xlabel('qx')
    ax.set_ylabel('qy')
    ax.set_zlabel('qz')
    ax.set_title('Rotation Space Trajectory (Imaginary Quaternion Components)')
    ax.legend()

    # Keep it square
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    plt.show()

### Plot fns for the lasa dataset
def plot_trajectories_newVAE(trajectory_list):
    """
    Plots multiple trajectories in 3D (or 2D fallback).
    Expects a list of tensors or arrays, each with shape [N, 3] or [N, 2].
    """
    if trajectory_list is None or len(trajectory_list) == 0:
        print("No trajectories to plot.")
        return

    # Inspect the first trajectory to determine spatial dimension
    first_traj = trajectory_list[0]
    dim = first_traj.shape[-1] if hasattr(first_traj, 'shape') else len(first_traj[0])

    fig = plt.figure(figsize=(10, 8))

    # Configure 3D or 2D axes
    if dim >= 3:
        ax = fig.add_subplot(111, projection='3d')
        is_3d = True
    else:
        ax = fig.add_subplot(111)
        is_3d = False

    # Helper flag to ensure "Start" and "End" appear only once in the legend
    legend_labeled = False

    for i, traj in enumerate(trajectory_list):
        # 1. Safe conversion: moves CUDA tensors to CPU before NumPy conversion
        if torch.is_tensor(traj):
            data = traj.detach().cpu().numpy()
        else:
            data = traj

        # 2. Extract X and Y coordinates
        x = data[:, 0]
        y = data[:, 1]

        # Handle Start/End labels for legend
        start_label = "Start" if not legend_labeled else ""
        end_label = "End" if not legend_labeled else ""

        if is_3d:
            # Extract Z coordinate only when dim >= 3
            z = data[:, 2]
            ax.plot(x, y, z, label=f'Trajectory {i + 1}', linewidth=2)
            ax.scatter(x[0], y[0], z[0], color='green', s=40, label=start_label)
            ax.scatter(x[-1], y[-1], z[-1], color='red', s=40, label=end_label)
        else:
            # Fallback 2D plot
            ax.plot(x, y, label=f'Trajectory {i + 1}', linewidth=2)
            ax.scatter(x[0], y[0], color='green', s=40, label=start_label)
            ax.scatter(x[-1], y[-1], color='red', s=40, label=end_label)

        legend_labeled = True

    # 3. Titles and Axis Labels
    mode_str = "3D" if is_3d else "2D"
    ax.set_title(f'Visualization of {len(trajectory_list)} Trajectories ({mode_str})')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')

    if is_3d:
        ax.set_zlabel('Z Position')
        # Safe check: only call set_box_aspect if installed Matplotlib version supports it
        if hasattr(ax, 'set_box_aspect'):
            ax.set_box_aspect([1, 1, 1])
    else:
        ax.axis('equal')

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='best')
    plt.show()

### Plot fns for the dataloaders
def plot_trajectories_on_manifold(vae, dataloader, space_name="Latent Space", latent_max=15):
    """
    Plots a batch of trajectories from the dataloader layered directly
    on top of the VAE's Riemannian Manifold.
    """
    vae.eval()
    print(f"Plotting dataloader trajectories over the {space_name} manifold...")

    plt.figure(figsize=(12, 12))

    # ==========================================
    # 1. RENDER THE BACKGROUND MANIFOLD
    # ==========================================
    x_range = np.linspace(-latent_max, latent_max, 100)
    y_range = np.linspace(-latent_max, latent_max, 100)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_pts = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    # Use the warning-free tensor extraction we fixed earlier
    with torch.no_grad():
        G = get_M(vae, grid_pts.clone())[2].clone().detach()
        det_G = torch.det(G).cpu().numpy().reshape(xx.shape)

    safe_det = np.clip(det_G, 1e-10, None)

    # Plot the contour map
    cp = plt.contourf(xx, yy, np.log10(safe_det), cmap='YlOrRd', alpha=0.4)
    plt.colorbar(cp, label='Magnification Factor (log10 Det M)')

    # ==========================================
    # 2. OVERLAY THE DATALOADER TRAJECTORIES
    # ==========================================
    # Get a single batch of data
    p1_batch, p2_batch, z_target_batch = next(iter(dataloader))
    num_paths = z_target_batch.shape[0]

    for i in range(num_paths):
        # We only want to label the very first line so the legend doesn't repeat!
        demo_label = 'Robot Demo' if i == 0 else ""
        start_label = 'Start (p1)' if i == 0 else ""
        goal_label = 'Goal (p2)' if i == 0 else ""

        # Move tensors to CPU and Numpy for Matplotlib
        z_np = z_target_batch[i].detach().cpu().numpy()
        p1_np = p1_batch[i].detach().cpu().numpy()
        p2_np = p2_batch[i].detach().cpu().numpy()

        # Plot Robot Demo path
        plt.plot(z_np[:, 0], z_np[:, 1], 'k--', alpha=0.6, label=demo_label)

        # Plot Start (p1)
        plt.scatter(p1_np[0], p1_np[1], c='green', s=50, zorder=5, label=start_label)

        # Plot Goal (p2)
        plt.scatter(p2_np[0], p2_np[1], c='red', marker='*', s=200, zorder=5, label=goal_label)

    # ==========================================
    # 3. FORMATTING
    # ==========================================
    plt.title(f"{space_name} Manifold with {num_paths} Dataloader Samples")
    plt.xlabel('z1')
    plt.ylabel('z2')

    # Restrict the view slightly so the plot bounds match the latent_max exactly
    plt.xlim(-latent_max, latent_max)
    plt.ylim(-latent_max, latent_max)

    plt.legend(loc='upper right', fontsize='medium')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')

    plt.show()

### Plot fns for vae testing
def visualize_metric(vae, space_name, latent_frame):
    #model.eval()
    vae.eval()

    print("Printing Riemannian Manifold...")

    # Plot the points
    frame = latent_frame+5
    plt.figure(figsize=(frame, frame))

    # 1. Plot Background Energy (Rough approximation using metric determinant)
    x_range = np.linspace(-frame, frame, 100)
    y_range = np.linspace(-frame, frame, 100)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_pts = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    with torch.no_grad():
        G = get_M(vae, grid_pts.clone())[2].clone().detach()  #Use determinant as a proxy for "Cost/Uncertainty"
        det_G = torch.det(G).cpu().numpy().reshape(xx.shape)
    #print("shape G: " + str(G.shape))
    #print("shape det G: " + str(det_G.shape))
    # Clip values to be at least 1e-10
    safe_det = np.clip(det_G, 1e-10, None)
    cp = plt.contourf(xx, yy, np.log10(safe_det), cmap='YlOrRd', alpha=0.3)
    plt.colorbar(cp, label='Magnification Factor')

    plt.title(f'Visualization of the {space_name} Latent Riemannian Manifold')
    plt.xlabel('z1')
    plt.ylabel('z2')
    #plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')#, adjustable='box')  # Keeps the scale 1:1 so circles look like circles

    #plt.show()

### Plot fns for node training
# Plotting Train Curves
def plot_training_results(history):
    """
    history: dict containing 'train_loss', 'val_imit', 'val_ener'
    """
    epochs = range(1, len(history['train_loss']) + 1)

    # Use a clean style
    #plt.style.use('seaborn-v0_8-muted')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- PLOT 1: TOTAL TRAINING LOSS ---
    ax1.plot(epochs[1:], history['train_loss'][1:], color='red', linewidth=2, label='Total Train Loss')
    ax1.plot(epochs[1:], history['train_ener'][1:], color='orange', linewidth=2, linestyle='--', label='Energy Loss')
    ax1.plot(epochs[1:], history['train_goal'][1:], color='green', linewidth=2, linestyle='-.', label='Goal Loss')
    ax1.plot(epochs[1:], history['train_imit'][1:], color='dodgerblue', linewidth=2, linestyle=':', label='Imitation Loss')

    ax1.set_yscale('log')  # Vital for seeing early vs late convergence
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (Log Scale)')
    ax1.set_title('Overall Model Convergence')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend()

    # --- PLOT 2: IMITATION VS ENERGY (Validation) ---
    # We plot these on the same graph to see the "Tug-of-War"
    color_val = 'blue'
    color_train = 'red'

    #ax2.plot(epochs[1:], history['val_loss'][1:], color=color_val, linewidth=2, linestyle='--', label='Validation Loss')
    #ax2.plot(epochs[1:], history['train_loss'][1:], color=color_train, linewidth=2, label='Training Loss')
    ax2.plot(epochs[1:], history['val_imit'][1:], color=color_val, linewidth=2, linestyle='--', label='Validation Imitation Loss')
    ax2.plot(epochs[1:], history['train_imit'][1:], color=color_train, linewidth=2, label='Training Imitation Loss')

    ax2.set_yscale('log')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Scaled Metrics (Log Scale)')
    ax2.set_title('Training vs Validation Metrics')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend()

    plt.tight_layout()
    plt.show()

# Saving Train Curves plots
def plot_save_training_results(history, save_dir=None, filename="training_results.svg"):
    """
    history: dict containing 'train_loss', 'val_imit', 'val_ener', etc.
    save_dir: str, path to the directory where the plot should be saved.
    filename: str, name of the file (should end in .svg)
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- PLOT 1: TOTAL TRAINING LOSS ---
    ax1.plot(epochs[1:], history['train_loss'][1:], color='red', linewidth=2, label='Total Train Loss')
    ax1.plot(epochs[1:], history['train_ener'][1:], color='orange', linewidth=2, linestyle='--', label='Energy Loss')
    ax1.plot(epochs[1:], history['train_goal'][1:], color='green', linewidth=2, linestyle='-.', label='Goal Loss')
    ax1.plot(epochs[1:], history['train_imit'][1:], color='dodgerblue', linewidth=2, linestyle=':',
             label='Imitation Loss')

    ax1.set_yscale('log')  # Vital for seeing early vs late convergence
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (Log Scale)')
    ax1.set_title('Overall Model Convergence')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend()

    # --- PLOT 2: IMITATION VS ENERGY (Validation) ---
    color_val = 'blue'
    color_train = 'red'

    ax2.plot(epochs[1:], history['val_imit'][1:], color=color_val, linewidth=2, linestyle='--',
             label='Validation Imitation Loss')
    ax2.plot(epochs[1:], history['train_imit'][1:], color=color_train, linewidth=2, label='Training Imitation Loss')

    ax2.set_yscale('log')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Scaled Metrics (Log Scale)')
    ax2.set_title('Training vs Validation Metrics')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend()

    plt.tight_layout()

    # --- NEW SAVING LOGIC ---
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        # Save as SVG (bbox_inches='tight' prevents labels from getting cut off)
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        print(f"📊 Training plot saved as SVG to: {save_path}")

    # You can keep plt.show() if you want it to pop up on your screen.
    # If you only want it to save silently in the background, replace this with plt.close()
    plt.close()

### Plot fns for node testing
#Visualization functions for proxy-geodesics
def visualize_path_comparison(model, vae, test_loader, space_name, latent_frame, num_samples=10):
    model.eval()
    vae.eval()

    # Get a batch of data
    p1_batch, p2_batch, z_target_batch = next(iter(test_loader))
    #print("number of testing paths: " + str(p1_batch.shape[0]-1))
    p1, p2, z_target = p1_batch[:num_samples], p2_batch[:num_samples], z_target_batch[:num_samples]

    # Predict with NODE
    t_steps = torch.linspace(0, 1, z_target.shape[1]).to(p1.device) #*5
    #t_steps = torch.linspace(0, 10, 50).to(p1.device)
    #t_steps = torch.linspace(0, 50, 1).to(p1.device)

    #print("z_target shape: " + str( z_target.shape))

    with torch.no_grad():
        pred_traj = model(p1, p2, t_steps)
        z_pred = pred_traj[:, :, 0, :].cpu().numpy()  # [T, num_samples, 2]

    #print("z_pred shape: " + str( z_pred.shape))
    #print("z_target shape: " + str( z_target.shape))
    z_target = z_target.detach().numpy()  # [num_samples, T, 2]
    #print("new z_target shape: " + str( z_target.shape))

    # Create plot with predetermined size
    frame = latent_frame+5
    plt.figure(figsize=(frame, frame))

    # 1. Plot Background Energy (Rough approximation using metric determinant)
    x_range = np.linspace(-frame, frame, 100)
    y_range = np.linspace(-frame, frame, 100)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_pts = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(p1.device)

    with torch.no_grad():
        G = get_M(vae, grid_pts)[2].clone().detach()
        # Use determinant as a proxy for "Cost/Uncertainty"
        det_G = torch.det(G).cpu().numpy().reshape(xx.shape)
    #print("shape G: " + str(G.shape))
    #print("shape det G: " + str(det_G.shape))
    # Clip values to be at least 1e-10
    safe_det = np.clip(det_G, 1e-10, None)
    cp = plt.contourf(xx, yy, np.log10(safe_det), cmap='YlOrRd', alpha=0.3)
    plt.colorbar(cp, label='Magnification Factor')

    for i in range(num_samples):

        # 2. Plot Teacher (Robot Demo)
        plt.plot(z_target[i, :, 0], z_target[i, :, 1], 'k--', label='Robot Demo', alpha=0.6, color='black')

        # 3. Plot Student (NODE Prediction)
        plt.plot(z_pred[:, i, 0], z_pred[:, i, 1], 'b-', linewidth=2, label='NODE Path', color='black')

        # 4. Markers for Start and Goal
        plt.scatter(p1[i, 0].detach().cpu().numpy(), p1[i, 1].detach().cpu().numpy(), c='green', s=50, label='Start (p1)')
        plt.scatter(p2[i, 0].detach().cpu().numpy(), p2[i, 1].detach().cpu().numpy(), c='red', marker='*', s=200, label='Goal (p2)')

        if i==0:
            plt.legend(fontsize='small')

    plt.title(f"Samples vs Predictions for {num_samples} samples of the {space_name}")

    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')# Keeps the scale 1:1 so circles look like circles
    #plt.show()

def visualize_random_points_comparison(model, vae, test_loader, space_name, latent_frame):
    model.eval()
    vae.eval()

    print("Testing with random point in test set...")
    # Get a batch of data
    p1_batch, p2_batch, z_target_batch = next(iter(test_loader))
    random_demo = random.randint(0, p1_batch.shape[0]-1)
    print("Random index: " + str(random_demo))

    p1, p2, z_target = p1_batch[random_demo], p2_batch[random_demo], z_target_batch[random_demo]

    print("Start point: " + str(p1))
    print("Goal point: " + str(p2))
    print("Goal path shape: " + str(z_target.shape))

    # 1. Define a new start and goal (in latent space)
    p1_test = p1
    p2_test = p2
    t_steps = torch.linspace(0, 1, z_target.shape[0]).to(p1.device) #*5
    #t_steps = torch.linspace(0, 10, 50).to(p1.device)
    #t_steps = torch.linspace(0, 500, 1).to(p1.device)


    # 2. Run the NODE
    with torch.no_grad():
        prediction = model(p1_test, p2_test, t_steps)
        z_path = prediction[:, :, 0].cpu().numpy()  # Shape [500, 1, 2]

    # 3. Convert to numpy for plotting
    #print("Predicted path shape: " + str(z_path.shape))
    #z_path_np = z_path.squeeze().cpu().numpy()
    #plot_trajectories(z_path)

    x = z_path[:, 0]
    y = z_path[:, 1]
    print("Start X point pred: " + str(x[0]))
    print("Start Y point pred: " + str(y[0]))
    print("Predicted path goal: (" + str(x[-1]) + ", " + str(y[-1]) +  ")")

    # Plot the curve
    frame = latent_frame+5
    plt.figure(figsize=(frame, frame))
    plt.plot(x, y, label=f'Test trajectory', linewidth=2, color='black')
    plt.plot(z_target[:, 0].detach().numpy(), z_target[:, 1].detach().numpy(), 'k--', label='Robot Demo', alpha=0.6, color='black')

    #plt.plot(ref_trajectory[:,0].detach().numpy(), ref_trajectory[:,1].detach().numpy(), 'k--', label=f'Reference trajectory', linewidth=2, color='black')

    # Optional: Mark the start and end points
    plt.scatter(p1[0].detach().numpy(), p1[1].detach().numpy(), marker='o',color='yellow', edgecolors='black', s=150, label="Expected Start")  # Start exp
    plt.scatter(x[0], y[0], color='green', s=50, label="Start")  # Start pred
    plt.scatter(p2[0].detach().numpy(), p2[1].detach().numpy(), marker='*',color='yellow', edgecolors='black', s=200, label="Expected End")  # End exp
    plt.scatter(x[-1], y[-1], color='red', s=50, label="End")  # End pred

    plt.text(x[0], y[0], '({:.2f}, {:.2f})'.format(x[0], y[0]))  # Add text pred Start
    plt.text(x[-1], y[-1], '({:.2f}, {:.2f})'.format(x[-1], y[-1]))  # Add text pred Goal
    plt.text(p1[0].detach().numpy(), p1[1].detach().numpy(), '({:.2f}, {:.2f})'.format(p1[0].detach().numpy(), p1[1].detach().numpy())) # Add text
    plt.text(p2[0].detach().numpy(), p2[1].detach().numpy(), '({:.2f}, {:.2f})'.format(p2[0].detach().numpy(), p2[1].detach().numpy()))  # Add text goal

    # 1. Plot Background Energy (Rough approximation using metric determinant)
    x_range = np.linspace(-frame, frame, 100)
    y_range = np.linspace(-frame, frame, 100)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_pts = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(p1_test.device)

    with torch.no_grad():
        G = get_M(vae, grid_pts)[2].clone().detach()
        # Use determinant as a proxy for "Cost/Uncertainty"
        det_G = torch.det(G).cpu().numpy().reshape(xx.shape)
    #print("shape G: " + str(G.shape))
    #print("shape det G: " + str(det_G.shape))
    # Clip values to be at least 1e-10
    safe_det = np.clip(det_G, 1e-10, None)
    cp = plt.contourf(xx, yy, np.log10(safe_det), cmap='YlOrRd', alpha=0.3)
    plt.colorbar(cp, label='Magnification Factor')

    plt.title(f'Visualization of Test Trajectory with random start/end point of the {space_name}')

    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')  # Keeps the scale 1:1 so circles look like circles
    # Forces the x and y axes to have the exact same scale
    #plt.gca().set_aspect('equal', adjustable='box')
    #plt.tight_layout(pad=3)
    #plt.show()
    print("Successful.")

def save_test_plot(save_dir, filename):
    """
    Saves the currently active matplotlib figure to the specified directory.

    save_dir: str, path to the directory where the plot should be saved.
    filename: str, name of the file (e.g., 'path_comparison.svg')
    """
    if save_dir:
        # 1. Ensure the folder exists
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        # 2. Save the plot
        # bbox_inches='tight' acts exactly like your tight_layout(pad=2.0)
        # to ensure titles and labels never get cut off!
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        print(f"📸 Test plot saved as SVG to: {save_path}")

    # 3. Close the figure silently in the background
    # This is CRUCIAL so your automated testing overnight loop doesn't
    # freeze waiting for you to manually close a pop-up window!
    #plt.close()