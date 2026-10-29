import torch
import numpy as np
import matplotlib
try:
    matplotlib.use('Qt5Agg', force=True)  # Forces the visual popup
    import matplotlib.pyplot as plt
except ImportError:
    # No Qt binding / no display (e.g. headless CI, SSH) - fall back to non-interactive.
    # matplotlib.use() alone doesn't eagerly import the backend module, so the failure
    # only surfaces once pyplot (or switch_backend) is actually imported - hence the
    # import lives inside this try block too.
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
import os
import random
from vae.vae_model import get_M


### Basic plot fns for the sphere case
#Plot to check sphere trajectories
def plot_sphere_trajectories(trajectory_list, draw_sphere=True):
    """
    Plots multiple 3D trajectories.
    Expects a list of tensors or numpy arrays, each with shape [N, 3]
    (where N is the number of time steps).
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Optional: Draw a faint sphere in the background for visual context
    if draw_sphere:
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 60)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(x_sphere, y_sphere, z_sphere,
                        color='lightgray', alpha=0.1, edgecolor='none')
        # Draw the equator
        ax.plot(np.cos(u), np.sin(u), np.zeros_like(u),
                color='gray', linestyle=':', alpha=0.5)

    # Loop through each trajectory in the list
    for i, traj in enumerate(trajectory_list):
        # Convert to numpy and move to CPU if necessary
        if torch.is_tensor(traj):
            data = traj.detach().cpu().numpy()
        else:
            data = traj

        # Extract x (col 0), y (col 1), and z (col 2)
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]

        # Only label the first few trajectories and the first start/end
        # to avoid massive legend spam if there are many paths
        traj_label = f'Trajectory {i + 1}' if i < 5 else ""
        start_label = "Start" if i == 0 else ""
        end_label = "End" if i == 0 else ""

        # Plot the curve
        ax.plot(x, y, z, linewidth=2, label=traj_label)

        # Mark the start and end points
        ax.scatter(x[0], y[0], z[0], color='green', s=50, zorder=5, label=start_label)
        ax.scatter(x[-1], y[-1], z[-1], color='red', marker='*', s=150, zorder=5, label=end_label)

    ax.set_title(f'Visualization of {len(trajectory_list)} 3D Trajectories', pad=20, fontsize=14)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')

    # 3D equivalent of plt.axis('equal') to keep the sphere perfectly round
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])

    # Clean up the grid panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linestyle='--', alpha=0.5)

    # Compile the legend, ignoring empty labels
    handles, labels = ax.get_legend_handles_labels()
    valid_handles = [h for h, l in zip(handles, labels) if l != ""]
    valid_labels = [l for l in labels if l != ""]
    if valid_handles:
        ax.legend(valid_handles, valid_labels, loc='upper right')

    plt.show()
#Plot to check sphere datasets
def plot_sphere_trajectories_from_dataloader(dataloader, space_name="Analytic Sphere", num_paths_to_plot=15):
    """
    Plots a batch of 3D trajectories from the dataloader layered directly
    on top of the 3D unit sphere manifold.
    """
    print(f"Plotting dataloader trajectories over the {space_name} manifold...")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # ==========================================
    # 1. RENDER THE BACKGROUND MANIFOLD (3D SPHERE)
    # ==========================================
    # Create a spherical mesh grid
    u = np.linspace(0, 2 * np.pi, 100)  # Longitude
    v = np.linspace(0, np.pi, 100)  # Colatitude
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface as translucent blue
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='orange', alpha=0.2,
                    edgecolor='gray', linewidth=0.2)

    # Plot the equator for spatial reference
    ax.plot(np.cos(u), np.sin(u), np.zeros_like(u), color='gray', linestyle='--', alpha=0.5)

    # ==========================================
    # 2. OVERLAY THE DATALOADER TRAJECTORIES
    # ==========================================
    batch = next(iter(dataloader))

    # --- THE FIX: Smart Data Extraction ---
    # If your dataloader returns (p1, p2, target_traj), the path is at index 2
    if isinstance(batch, (list, tuple)) and len(batch) >= 3:
        gt_traj_batch = batch[2]
    else:
        # Fallback for simple TensorDatasets
        gt_traj_batch = batch[0]

        # Shape of gt_traj_batch should now correctly be [Batch, Time, 3]

    # Restrict the number of paths so the plot doesn't become a messy hairball
    actual_paths_to_plot = min(num_paths_to_plot, gt_traj_batch.shape[0])

    for i in range(actual_paths_to_plot):
        # Only label the first line to avoid repeating in the legend
        demo_label = 'Robot Demo Path' if i == 0 else ""
        start_label = 'Start' if i == 0 else ""
        goal_label = 'Goal' if i == 0 else ""

        # Move tensor to CPU and Numpy for Matplotlib
        path_np = gt_traj_batch[i].detach().cpu().numpy()  # Shape: [Time, 3]

        # Extract start and end points
        start_pt = path_np[0]
        goal_pt = path_np[-1]

        # Plot Robot Demo path
        ax.plot(path_np[:, 0], path_np[:, 1], path_np[:, 2],
                color='black', linestyle='--', alpha=0.8, linewidth=2, label=demo_label)

        # Plot Start
        ax.scatter(start_pt[0], start_pt[1], start_pt[2],
                   color='green', s=60, zorder=5, label=start_label)

        # Plot Goal
        ax.scatter(goal_pt[0], goal_pt[1], goal_pt[2],
                   color='red', marker='*', s=200, zorder=5, label=goal_label)

    # ==========================================
    # 3. FORMATTING
    # ==========================================
    ax.set_title(f"{space_name}: {actual_paths_to_plot} Dataloader Samples", pad=20, fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Force the axes to have the same scale, keeping the sphere perfectly round
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])

    ax.legend(loc='upper right', fontsize='medium')

    # Clean up the background grid
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linestyle='--', alpha=0.5)

    # Set a nice viewing angle
    ax.view_init(elev=20, azim=45)

    plt.show()
## Testing fns
# Plot fns for manifold testing
def visualize_sphere_metric(manifold, space_name="Analytic Sphere"):
    """
    Renders the 3D physical surface of the manifold.
    Replaces the 2D metric contour map.
    """
    print(f"Printing {space_name} Manifold...")

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Create a spherical mesh grid
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface as translucent blue
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='orange', alpha=0.3,
                    edgecolor='gray', linewidth=0.2)
    # Plot the equator for spatial reference
    ax.plot(np.cos(u), np.sin(u), np.zeros_like(u), color='gray', linestyle='--', alpha=0.8)

    ax.set_title(f'Visualization of the {space_name}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.grid(True, linestyle='--', alpha=0.6)
    # plt.show()
# Plot fns for node testing
def visualize_sphere_path_comparison(model, manifold, test_loader, space_name="Analytic Sphere", num_samples=10):
    model.eval()

    # Smart unpacking to handle the dataloader format
    batch = next(iter(test_loader))
    if isinstance(batch, (list, tuple)) and len(batch) >= 3:
        p1_batch, p2_batch, z_target_batch = batch[0], batch[1], batch[2]
    else:
        z_target_batch = batch[0]
        p1_batch, p2_batch = z_target_batch[:, 0, :], z_target_batch[:, -1, :]

    p1, p2, z_target = p1_batch[:num_samples], p2_batch[:num_samples], z_target_batch[:num_samples]

    # Predict with NODE
    t_steps = torch.linspace(0, 1, z_target.shape[1]).to(p1.device)
    model_device = next(model.parameters()).device

    with torch.no_grad():
        pred_traj = model(p1.to(model_device), p2.to(model_device), t_steps.to(model_device))
        z_pred = pred_traj[:, :, 0, :].cpu().numpy()  # Shape: [T, num_samples, 3]

    z_target = z_target.detach().cpu().numpy()  # Shape: [num_samples, T, 3]

    # Create plot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Render Sphere Background
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='lightgray', alpha=0.15, edgecolor='none')
    ax.plot(np.cos(u), np.sin(u), np.zeros_like(u), color='gray', linestyle=':', alpha=0.3)

    for i in range(num_samples):
        # Only label the first iteration to avoid legend spam
        demo_label = 'Ideal Geodesic (Demo)' if i == 0 else ""
        node_label = 'NODE Prediction' if i == 0 else ""
        start_label = 'Start (p1)' if i == 0 else ""
        goal_label = 'Goal (p2)' if i == 0 else ""

        # Plot Teacher (Robot Demo)
        ax.plot(z_target[i, :, 0], z_target[i, :, 1], z_target[i, :, 2],
                'k--', label=demo_label, alpha=0.6, linewidth=2)

        # Plot Student (NODE Prediction)
        ax.plot(z_pred[:, i, 0], z_pred[:, i, 1], z_pred[:, i, 2],
                'b-', linewidth=2, label=node_label, alpha=0.8)

        # Markers for Start and Goal
        p1_np = p1[i].detach().cpu().numpy()
        p2_np = p2[i].detach().cpu().numpy()
        ax.scatter(p1_np[0], p1_np[1], p1_np[2], marker='o', color='yellow', edgecolors='black', s=150,
                   label="Expected Start")
        ax.scatter(z_pred[0, i, 0], z_pred[0, i, 1], z_pred[0, i, 2], color='green', s=50, zorder=5,
                   label="Predicted Start")
        ax.scatter(p2_np[0], p2_np[1], p2_np[2], marker='*', color='yellow', edgecolors='black', s=250,
                   label="Expected End")
        ax.scatter(z_pred[-1, i, 0], z_pred[-1, i, 1], z_pred[-1, i, 2], color='red', s=50, zorder=5,
                   label="Predicted End")

        if i==0:
            ax.legend(fontsize='small', loc='upper right')

    ax.set_title(f"3D Predictions vs Ground Truth ({num_samples} samples)", pad=20, fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    # plt.show()
def visualize_sphere_random_points_comparison(model, manifold, test_loader, space_name="Analytic Sphere"):
    model.eval()

    print("Testing with random point in test set...")
    batch = next(iter(test_loader))
    if isinstance(batch, (list, tuple)) and len(batch) >= 3:
        p1_batch, p2_batch, z_target_batch = batch[0], batch[1], batch[2]
    else:
        z_target_batch = batch[0]
        p1_batch, p2_batch = z_target_batch[:, 0, :], z_target_batch[:, -1, :]

    random_demo = random.randint(0, p1_batch.shape[0] - 1)
    print("Random index: " + str(random_demo))

    p1, p2, z_target = p1_batch[random_demo], p2_batch[random_demo], z_target_batch[random_demo]

    # Run the NODE
    p1_test, p2_test = p1, p2
    t_steps = torch.linspace(0, 1, z_target.shape[0]).to(p1.device)
    model_device = next(model.parameters()).device

    with torch.no_grad():
        prediction = model(p1_test.to(model_device), p2_test.to(model_device), t_steps.to(model_device))
        z_path = prediction[:, :, 0].cpu().numpy()  # Shape [T, 1, 3]

    z_path = z_path.squeeze()  # Shape [T, 3]
    x, y, z = z_path[:, 0], z_path[:, 1], z_path[:, 2]

    print(f"Start point pred: ({x[0]:.2f}, {y[0]:.2f}, {z[0]:.2f})")
    print(f"Goal point pred: ({x[-1]:.2f}, {y[-1]:.2f}, {z[-1]:.2f})")

    # Setup 3D Plot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Render Sphere Background
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='lightgray', alpha=0.15, edgecolor='none')
    ax.plot(np.cos(u), np.sin(u), np.zeros_like(u), color='gray', linestyle=':', alpha=0.3)

    # Plot True and Predicted Paths
    z_target_np = z_target.detach().cpu().numpy()
    ax.plot(x, y, z, label='Test Trajectory (Pred)', linewidth=3, color='blue', alpha=0.8)
    ax.plot(z_target_np[:, 0], z_target_np[:, 1], z_target_np[:, 2],
            'k--', label='Ideal Geodesic (Demo)', linewidth=2, color='black')

    # Markers for Start and End points
    p1_np = p1.detach().cpu().numpy()
    p2_np = p2.detach().cpu().numpy()

    ax.scatter(p1_np[0], p1_np[1], p1_np[2], marker='o', color='yellow', edgecolors='black', s=150,
               label="Expected Start")
    ax.scatter(x[0], y[0], z[0], color='green', s=50, zorder=5, label="Predicted Start")

    ax.scatter(p2_np[0], p2_np[1], p2_np[2], marker='*', color='yellow', edgecolors='black', s=250,
               label="Expected End")
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=50, zorder=5, label="Predicted End")

    ax.set_title(f"Random Test Sample: {space_name}", pad=20, fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.legend(loc='upper right')

    # Optional: Clean up grid panes for better visibility
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    print("Successful.")
    # plt.show()

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

def plot_trajectories_on_manifold_list(vae, trajectories, space_name="Latent Space", latent_max=15):
    """
    Plots a list of trajectories layered directly on top of the VAE's Riemannian Manifold.
    """
    vae.eval()
    print(f"Plotting {len(trajectories)} trajectories over the {space_name} manifold...")

    plt.figure(figsize=(latent_max+3, latent_max+3))

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
    # 2. OVERLAY THE TRAJECTORIES
    # ==========================================
    num_paths = len(trajectories)

    for i, traj in enumerate(trajectories):
        # We only want to label the very first line so the legend doesn't repeat!
        demo_label = 'Trajectory' if i == 0 else ""
        start_label = 'Start' if i == 0 else ""
        goal_label = 'Goal' if i == 0 else ""

        # Convert to numpy if it's a PyTorch tensor
        if torch.is_tensor(traj):
            z_np = traj.detach().cpu().numpy()
        else:
            z_np = np.asarray(traj)

        # Automatically extract the start and goal points from the trajectory ends
        p1_np = z_np[0]   # First point
        p2_np = z_np[-1]  # Last point

        # Plot the continuous path
        #plt.plot(z_np[:, 0], z_np[:, 1], 'k--', alpha=0.6, label=demo_label)

        # Plot Start (p1)
        plt.scatter(p1_np[0], p1_np[1], c='green', s=50, zorder=5, label=start_label)

        # Plot Goal (p2)
        plt.scatter(p2_np[0], p2_np[1], c='red', marker='*', s=200, zorder=5, label=goal_label)

    # ==========================================
    # 3. FORMATTING
    # ==========================================
    plt.title(f"{space_name} Manifold with {num_paths} Trajectories")
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
        #pred_traj = model(p1, p2, t_steps)
        model_device = next(model.parameters()).device
        pred_traj = model(p1.to(model_device), p2.to(model_device), t_steps.to(model_device))
        z_pred = pred_traj[:, :, 0, :].cpu().numpy()  # [T, num_samples, 2]

    #print("z_pred shape: " + str( z_pred.shape))
    #print("z_target shape: " + str( z_target.shape))
    z_target = z_target.detach().numpy()  # [num_samples, T, 2]
    #print("new z_target shape: " + str( z_target.shape))

    # Create plot with predetermined size
    frame = latent_frame+5
    plt.figure(figsize=(frame, frame))

    # 1. Plot Background Energy (Rough approximation using metric determinant)
    point_dim = 100
    x_axis = np.linspace(-frame, frame, point_dim)
    y_axis = np.linspace(-frame, frame, point_dim)
    x_range = np.linspace(-frame, frame, point_dim)
    y_range = np.linspace(-frame, frame, point_dim)
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
        #prediction = model(p1_test, p2_test, t_steps)
        model_device = next(model.parameters()).device
        prediction = model(p1_test.to(model_device), p2_test.to(model_device), t_steps.to(model_device))
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

### Plot fns for benchmark testing
def visualize_gtvspred_comparison_multiple(proxy_geo, z_real, vae_model, device, space_name, axis_point, latent_frame, num_samples=10, model_name = " "):
    """
        Standardized Inputs:
        z_real:    [Batch, Time, 2] -> e.g., [7, 500, 2]
        proxy_geo: [Batch, Time, 2] -> e.g., [7, 500, 2]
    """
    # Create plot with predetermined size
    frame = latent_frame+5
    plt.figure(figsize=(frame, frame))

    # 1. Background: Energy Manifold
    vae_model.eval()
    torch.manual_seed(42)

    point_dim = axis_point
    x_axis = np.linspace(-frame, frame, point_dim)
    y_axis = np.linspace(-frame, frame, point_dim)
    xx, yy = np.meshgrid(x_axis, y_axis, indexing='ij')
    grid_pts = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32).to(device)

    with torch.no_grad():
        # Using your get_M function
        _, _, G = get_M(vae_model, grid_pts.clone())
        G_torch = G.clone().detach() #torch.tensor(G)
        # Determinant calculation for the background heatmap
        det_G = torch.det(G_torch).cpu().numpy().reshape(xx.shape)

    safe_det = np.clip(det_G, 1e-10, None) #mf_flat = torch.log(torch.sqrt(torch.abs(det_G) + 1e-10))
    # Plot Background
    cp = plt.contourf(xx, yy, np.log10(safe_det), cmap='YlOrRd', alpha=0.3)
    plt.colorbar(cp, label='Magnification Factor')

    # 2. Iterate through the Batch (The 7 Paths)
    #num_paths = z_real.shape[0]
    num_paths = num_samples
    #path_idx = np.random.randint(0, proxy_geo.shape[0]-1, size=num_paths)

    # Plot Trajectories
    for i in range(num_paths):
    #for ix in range(num_paths):
        #idx = path_idx[ix]
        #i=idx
        # Extract specific path data
        z_target_np = z_real[i].detach().cpu().numpy()  # [500, 2]
        z_pred_np = proxy_geo[i].detach().cpu().numpy()  # [500, 2]

        plt.plot(z_target_np[:, 0], z_target_np[:, 1], 'k--', color='black', label='Ground Truth (Demo)', alpha=0.8, linewidth=1)
        plt.plot(z_pred_np[:, 0], z_pred_np[:, 1], color='black', label=f'{model_name} Prediction', linewidth=2)

        # 3. Markers
        # Expected (Yellow Stars/Circles)
        plt.scatter(z_target_np[0, 0], z_target_np[0, 1], marker='o', color='yellow', edgecolors='black', s=150, label="GT Start")
        plt.scatter(z_target_np[-1, 0], z_target_np[-1, 1], marker='*', color='yellow', edgecolors='black', s=200, label="GT Goal")

        # Predicted (Green/Red dots)
        plt.scatter(z_pred_np[0, 0], z_pred_np[0, 1], marker='o', color='green', s=50, label="Pred Start")
        plt.scatter(z_pred_np[-1, 0], z_pred_np[-1, 1], marker='o', color='red', s=30, label="Pred End")

        if i == 0:
        #if ix == 0:
            plt.legend(loc = "lower center")

    # Clean up and show
    plt.title(f'Benchmark Comparison for {space_name}: Ground Truth vs {model_name} Model ({num_paths} Predicted Trajectories)')
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    #plt.show()

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