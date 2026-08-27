import os
import torch
import pickle
import copy
import numpy as np
import torch.nn.functional as F
import random
from scipy.io import loadmat
import json

from node.utils.hyperparameter_calculator import calculate_dataset_baselines
from node.utils.plots import plot_trajectories

### SHARED UTILITIES
#Dataset creation
def create_dataset(encoded_paths, save_dir):
    """
    Args:
        encoded_paths (list of tensors): List of encoded trajectories, each shape [Steps, Latent_Dim]
        save_dir (str): Folder to save .pt files.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_count = 0
    for path_idx, path in enumerate(encoded_paths):
        sample = {
            'p1': path[0, :].clone(),
            'p2': path[-1, :].clone(),
            'z': path.clone(),
            'path_id': path_idx
        }
        torch.save(sample, f"{save_dir}/path_{file_count}.pt")
        file_count += 1
    print(f"✅ Successfully created {file_count} segments in {save_dir}")
#Encode demonstrations
def encode_demonstrations_paths(vae_model, demo_paths, device):
    latent_paths = []
    for i in range(len(demo_paths)):
        # 1. Safely convert the individual path to a PyTorch float tensor regardless of its original type
        path_tensor = torch.as_tensor(demo_paths[i], dtype=torch.float32, device=device)

        # 2. Isolate it from the computational graph cleanly using PyTorch commands
        path_tensor = path_tensor.clone().detach()

        # 3. Pass the safe tensor into the VAE encoder
        encoded_path = vae_model.encode(path_tensor, train_rbf=True)[1]

        latent_paths.append(encoded_path)
    return latent_paths
#Segmentation of encoded demonstrations
def create_universal_segmented_dataset(encoded_paths, min_window=20, max_window=150, samples_per_path=499):
    """
    Creates a randomized dataset to teach 'Local Flow' for Many-to-Many navigation.

    Args:
        encoded_paths: List of [Steps, Latent_Dim] tensors.
        min_window: Shortest trip the robot learns.
        max_window: Longest trip the robot learns.
        samples_per_path: How many random sub-segments to pull from each demo.
    """
    # 1. Setup and Seed for consistency
    torch.manual_seed(42)
    np.random.seed(42)
    segmented_data = []
    for path in encoded_paths:
        num_steps = path.shape[0]
        print("num_steps: ", num_steps)

        #segmented_data.append(path)

        for _ in range(samples_per_path):
            # 1. Randomize the "Trip Duration" (Window Size) --> original line
            #win_size = np.random.randint(min_window, max_window)

            # 1. Choose a "Tier" based on 30/50/20 distribution
            roll = random.random()
            if roll < 0.35:  # SHORT (Precision)
                win_size = random.randint(min_window, (num_steps//2)-100)
            elif roll < 0.70:  # MEDIUM (Flow)
                win_size = random.randint(((num_steps//2)-100) + 1, (num_steps//2)+100)
            else:  # LONG (Global context)
                win_size = random.randint(((num_steps//2)+100)+1, max_window)

            # 2. Pick a random start point
            if num_steps <= win_size:
                continue
            start = np.random.randint(0, num_steps - win_size)
            end = start + win_size

            # 3. Extract Segment
            z_segment = path[start:end, :].clone()

            # We store it as a tuple: (Start_Point, Goal_Point, Entire_Path)
            segmented_data.append(z_segment)
    print(f"✅ Created {len(segmented_data)} segments.")
    return segmented_data
#Time-alignment over indexes
def unify_time_steps(latent_paths, time_steps=50):
    unified_paths = []
    for path in latent_paths:
        indices = torch.linspace(0, len(path) - 1, steps=time_steps).round().long()
        u_path = path[indices]
        unified_paths.append(u_path)
    return unified_paths

### Dataset Pre-Processing
## TOY EXAMPLE LOGIC
#Get example demonstrations
def get_demonstrations_paths(origin_dir, trajectory_number, test_id, s2_letter, r2_letter):

    trajectories = []
    for i in range(trajectory_number):
        r2_file_test = origin_dir+"/letter_" + r2_letter + "_R2_" + str(test_id) + ".p"
        r2_file = origin_dir+"/letter_" + r2_letter + "_R2_" + str(i) + ".p"
        s2_file_test = origin_dir+"/letter_" + s2_letter + "_S2_" + str(test_id) + ".p"
        s2_file = origin_dir+"/letter_" + s2_letter + "_S2_" + str(i) + ".p"

        test_traj = pickle.load(open(r2_file_test, "rb"), encoding="latin1").transpose()
        trajectory = pickle.load(open(r2_file, "rb"), encoding="latin1").transpose()
        test_trajectory_qua = -pickle.load(open(s2_file_test, "rb"), encoding="latin1")
        trajectory_qua = pickle.load(open(s2_file, "rb"), encoding="latin1")

        trajectory_n = copy.deepcopy(trajectory)
        trajectory = np.append(trajectory, trajectory_qua, 1) #--> Small horsefeet
        trajectory_n = np.append(trajectory_n, -trajectory_qua, 1) #--> Big horsefeet
        test_traj_n = copy.deepcopy(test_traj)
        test_traj = np.append(test_traj, test_trajectory_qua, 1)
        test_traj_n = np.append(test_traj_n, -test_trajectory_qua, 1)

        #For Experiments
        trajectories.append(trajectory)
        trajectories.append(trajectory_n)
        #For training split
        # if i != test_id:
        #    trajectories.append(trajectory)
        #    trajectories.append(trajectory_n)
        # if i == test_id:
        #     trajectories.append(test_traj)
        #     trajectories.append(test_traj_n)
        # #Testing trajectories extraction
        # print("Testing traj list after" + str(i))
        # print(len(trajectories))
    return trajectories
#Trajectory Interpolation
def interpolate_trajectories(original_pos, interpolation_points):
    interpolated_paths = []

    for demo in original_pos:
        pos_original = demo[:,:2]
        new_pos = interpolate_position(pos_original, interpolation_points)
        ori_original = demo[:,2:]
        new_ori = interpolate_orientation(ori_original, interpolation_points)
        combined = torch.cat([new_pos, new_ori], dim=1)
        interpolated_paths.append(combined)

    return interpolated_paths
#Position Interpolation
def interpolate_position(original_pos, new_sampling_points):
    """
    original_pos: Tensor of shape [200, 2] (x, y)
    Returns: Tensor of shape [new_sampling_points, 2]
    """
    # 0. If it's a numpy array, make it a tensor
    if isinstance(original_pos, np.ndarray):
        original_pos = torch.from_numpy(original_pos).float()

    # 1. Reshape for PyTorch interpolate: [Batch, Channels, Length]
    # Current: [200, 2] -> Needed: [1, 2, 200]
    path = original_pos.T.unsqueeze(0)

    # 2. Perform Linear Interpolation
    # 'align_corners=True' ensures the first and last points stay exactly the same
    new_path = F.interpolate(path, size=new_sampling_points, mode='linear', align_corners=True)

    # 3. Reshape back to [new_sampling_points, 2]
    new_pos = new_path.squeeze(0).T

    return new_pos
#Orientation Interpolation
def interpolate_orientation(original_pos, new_sampling_points):
    """
    original_pos: Tensor of shape [200, 3] (qx, qy, qz)
    new_sampling_points: # of points to interpolate
    Returns: Tensor of shape [500, 3]
    """
    # 1. Convert to tensor if it's a list or numpy
    if not torch.is_tensor(original_pos):
        original_pos = torch.tensor(original_pos, dtype=torch.float32)

    # 2. Reshape for interpolation: [1, 3, 200]
    x = original_pos.T.unsqueeze(0)

    # 3. Linear Interpolation to 500 points
    x_new = F.interpolate(x, size=new_sampling_points, mode='linear', align_corners=True)

    # 4. Reshape back to [500, 3]
    rot_new = x_new.squeeze(0).T

    # 5. RENORMALIZATION (The "N" in NLERP)
    # This ensures the 500 points stay on the same "sphere surface" as the original points
    # We calculate the norm of the original points to know our "target radius"
    target_radius = torch.norm(original_pos, dim=1).mean()

    current_norms = torch.norm(rot_new, dim=1, keepdim=True)
    rot_new = (rot_new / current_norms) * target_radius

    return rot_new

## lasa LOGIC
#Get lasa-demonstrations
def get_demonstrations_paths_newVAE(origin_dir, origin_file, trajectory_number):

    # Our VAE dataset
    mat_file = f"{origin_dir}/{origin_file}"
    demoUQ = loadmat(mat_file)["demoUQ"]

    # --- STEP 1: Load all 7 raw trajectories first (no normalization yet) ---
    raw_trajectories = []
    raw_quats = []

    for i in range(trajectory_number):
        pos = demoUQ[0, i]["tsPos"][0, 0].T
        quat = demoUQ[0, i]["quat"][0, 0].T
        # if i == test_id:
        #     quat = -quat  # keep your existing sign convention for the test trajectory
        raw_trajectories.append(pos)
        raw_quats.append(quat)

    # --- STEP 2 & 3: Normalize the positions BEFORE concatenating ---
    norm_trajectories = normalize_newVAE(raw_trajectories)

    # --- STEP 4: Concatenate positions with Quaternions & handle test_id ---
    trajectories = []

    for i in range(trajectory_number):
        pos_norm = norm_trajectories[i]
        quat = raw_quats[i]

        trajectory = pos_norm
        trajectory_qua = quat
        trajectory_n = copy.deepcopy(trajectory)
        trajectory = np.append(trajectory, trajectory_qua, 1)  # --> Small horsefeet
        trajectory_n = np.append(trajectory_n, -trajectory_qua, 1)  # --> Big horsefeet

        trajectories.append(trajectory)
        trajectories.append(trajectory_n)

    print(f"Length Trajectory: {len(trajectory)}")
    print(f"Length Trajectories: {len(trajectories)}")
    print(f"Trajectories euclidian dimension: {len(trajectories[0][1])}")
    print(f"Trajectory 0: {trajectories[0][0]}")
    print(f"Trajectory: {trajectories[1][0]}")
    print(f"Trajectory: {trajectories[2][0]}")
    print(f"Trajectory: {trajectories[3][0]}")
    print(f"Trajectory: {trajectories[4][0]}")
    print(f"Trajectory: {trajectories[5][0]}")
    print(f"Trajectory 6: {trajectories[6][0]}")
    print(f"Trajectory 7: {trajectories[7][0]}")
    print(f"Trajectory: {trajectories[8][0]}")
    print(f"Trajectory: {trajectories[9][0]}")
    print(f"Trajectory: {trajectories[10][0]}")

    print(f"Total # of Trajectories: {len(trajectories)}")

    return trajectories
#Normalize lasa-demonstrations
def normalize_newVAE(raw_trajectories):
    # --- STEP 2: Compute ONE shared normalization from all 7 trajectories combined ---
    all_xy = np.vstack([traj[:, 0:2] for traj in raw_trajectories])

    xy_min = all_xy.min(axis=0)
    xy_max = all_xy.max(axis=0)
    xy_center = (xy_min + xy_max) / 2
    xy_scale = (xy_max - xy_min).max() / 2  # single scalar -> isotropic scaling


    def normalize_xy(xy_data):
        return (xy_data - xy_center) / xy_scale


    # --- STEP 3: Apply the same normalization to every trajectory ---
    norm_trajectories = []
    for pos in raw_trajectories:
        xy_norm = normalize_xy(pos[:, 0:2])
        z = pos[:, 2:3]
        norm_trajectories.append(np.hstack([xy_norm, z]))

    return norm_trajectories

## ROBOT EXPERIMENT LOGIC
#Get Robot-demonstrations
def get_demonstrations_paths_robotexp(model_task="Angle"):
    trajectories = []
    return trajectories

### THE MASTER PREPROCESSOR
def build_dataset_offline(config, vae_model, device='cpu'):
    """
    Reads the config, looks at the requested dataset, and processes it
    if the .pt files don't already exist.
    """
    dataset_type = config['dataset']['type']  # e.g., 'lasa' or 'toy'
    shape_name = config['dataset']['shape_name']  # e.g., 'NShape'
    save_dir = config['dataset']['save_dir']  # e.g., './data/NShape_50seg'

    if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
        print(f"Dataset already exists at {save_dir}. Skipping preprocessing.")
        return save_dir

    print(f"Building dataset for {dataset_type.upper()} - {shape_name}...")
    os.makedirs(save_dir, exist_ok=True)

    if dataset_type == 'toy':
        real_paths = get_demonstrations_paths(
            origin_dir = config['dataset']['origin_dir'],
            trajectory_number=config['dataset']['trajectory_number'], # the number of total demonstration files
            test_id = config['dataset']['test_id'],  # the index of the demonstration used for testing
            r2_letter = config['dataset']['r2_letter'],
            s2_letter = config['dataset']['s2_letter']
        )
        interpolated_paths = interpolate_trajectories(
            real_paths,
            interpolation_points=config['dataset']['interpolation_points'],
        )
        latent_paths = encode_demonstrations_paths(vae_model, interpolated_paths, device)

        # --- RUN THE BASELINE CALCULATOR ---
        calc_baseline, calc_scale = calculate_dataset_baselines(vae_model, latent_paths, device=device)
        metadata_path = os.path.join(save_dir, "dataset_baselines.json")
        with open(metadata_path, 'w') as f:
            json.dump({"safe_baseline": calc_baseline, "metric_scale_energy": calc_scale}, f, indent=4)
        print(f"💾 Saved baselines permanently to: {metadata_path}")

        segmented_paths = create_universal_segmented_dataset(
            latent_paths,
            min_window=config['dataset']['min_window'],
            max_window=config['dataset']['max_window'],
            samples_per_path=config['dataset']['samples_per_path']
        )
        final_paths = unify_time_steps(segmented_paths, time_steps=config['dataset']['time_steps'])
        create_dataset(final_paths, save_dir)
    elif dataset_type == 'lasa':
        real_paths = get_demonstrations_paths_newVAE(
            origin_dir = config['dataset']['origin_dir'],
            origin_file = config['dataset']['origin_file'],
            trajectory_number=config['dataset']['trajectory_number'] # the number of total demonstration files
        )
        latent_paths = encode_demonstrations_paths(vae_model, real_paths, device)

        # --- RUN THE BASELINE CALCULATOR ---
        calc_baseline, calc_scale = calculate_dataset_baselines(vae_model, latent_paths, device=device)
        metadata_path = os.path.join(save_dir, "dataset_baselines.json")
        with open(metadata_path, 'w') as f:
            json.dump({"safe_baseline": calc_baseline, "metric_scale_energy": calc_scale}, f, indent=4)
        print(f"💾 Saved baselines permanently to: {metadata_path}")

        segmented_paths = create_universal_segmented_dataset(
            latent_paths,
            min_window=config['dataset']['min_window'],
            max_window=config['dataset']['max_window'],
            samples_per_path=config['dataset']['samples_per_path']
        )
        final_paths = unify_time_steps(segmented_paths, time_steps=config['dataset']['time_steps'])
        create_dataset(final_paths, save_dir)
    #elif dataset_type == 'robot':
        ### To Be Added

    return save_dir

#Datasets preparation
def prepare_data(vae_model):
    print("Testing real demos...")
    real_space_paths = get_demonstrations_paths(vae_model)
    print("# of Paths: " + str(len(real_space_paths)))
    print("Paths shape: " + str(real_space_paths[0].shape))
    print("Points values: " + str(real_space_paths[0][0]))
    print("Points values: " + str(real_space_paths[0][-1]))

    print("Testing interpolation...")
    print("Real paths shape: " + str(real_space_paths[0].shape))
    interpolated_paths = interpolate_trajectories(real_space_paths, 500)
    print("# of paths: " + str(len(interpolated_paths)))
    print("Interpolated Paths shape: " + str(interpolated_paths[0].shape))

    print("Testing latent demos...")
    print("Interpolated Paths shape: " + str(interpolated_paths[0].shape))
    latent_space_paths = encode_demonstrations_paths(vae_model, interpolated_paths)
    print("# of paths: " + str(len(latent_space_paths)))
    print("Latent Paths shape: " + str(latent_space_paths[0].shape))

    plot_trajectories(real_space_paths)
    plot_trajectories(interpolated_paths)
    plot_trajectories(latent_space_paths)
    print("Encoding")
    print("Testing latent demos unified time steps...")
    latent_space_paths1 = encode_demonstrations_paths(vae_model, torch.tensor(real_space_paths))
    latent_space_paths2 = encode_demonstrations_paths(vae_model, interpolated_paths)
    print("# of paths: " + str(len(latent_space_paths1)))
    print("Latent Paths shape without Interpolation: " + str(latent_space_paths1[0].shape))
    print("# of paths: " + str(len(latent_space_paths2)))
    print("Latent Paths shape after Interpolation: " + str(latent_space_paths2[0].shape))

    # Saving robot demos
    #save_robot_trajectories(interpolated_paths, data_name = "/robot_demos_txt", file_format="txt")
    #save_robot_trajectories(interpolated_paths, data_name = "/robot_demos_csv", file_format="csv")

    print("Segmentation")
    segmented_paths = create_universal_segmented_dataset(latent_space_paths, min_window=80, max_window=480, samples_per_path=50)
    plot_trajectories(segmented_paths)
    print("# of paths: " + str(len(segmented_paths)))
    print("Latent Paths shape after Segmentation: " + str(segmented_paths[0].shape))
    print("DownSampling")
    test = unify_time_steps(segmented_paths, time_steps=50)
    test1 = unify_time_steps(latent_space_paths, time_steps=50)

    plot_trajectories(test)
    print("# of paths: " + str(len(test)))
    print("Latent Paths shape after Unify: " + str(test[0].shape))
    # ############ CREATING DATASETs PATHS --> DONE
    # create_dataset(test, save_dir=ROOT_DIR + "/NODE_Datasets/14demos_random_50segpaths_50ts") #--> Create train + val dataset
    #create_dataset(test, save_dir=ROOT_DIR + "/test_datasets/14demos_random_50segpaths_50ts_tsd") #--> Create test dataset
    #create_dataset(test1, save_dir=ROOT_DIR + "/test_datasets/original_demos_50ts_tsd") #--> Create test dataset with full paths
