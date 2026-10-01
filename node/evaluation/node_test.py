import torch
from torch.utils.data import Dataset, DataLoader
import os
from node.node_model import GoalConditionedNODE
from node.utils.plots import visualize_metric, visualize_path_comparison, visualize_random_points_comparison, \
    save_test_plot


################################################### Testing My RiemMSE NODE Playground
#Loading the trained RNODE
def load_trained_node(model_path, latent_dim=2, hidden_dim=256, device='cpu'):
    # 1. Recreate the architecture
    model = GoalConditionedNODE(latent_dim=latent_dim, hidden_dim=hidden_dim)

    # 2. Load the state dictionary (the weights)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find saved model at: {model_path}")
    # Use weights_only=True for security as we discussed
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    # 3. Load weights into the model
    model.load_state_dict(state_dict)

    # 4. Set to evaluation mode
    model.to(device)
    model.eval()

    print(f"Successfully loaded model from {model_path}")
    return model

def test_node_or(vae_model):
    print("Testing...")
    #Use your existing Dataset class pointed at the NEW folder
    test_data_final = LatentTrajectoryDataset(ROOT_DIR + "/test_datasets/14demos_random_50segpaths_50ts_tsd") #original_demos_50ts_tsd  random_50segpaths_50ts_tds 14demos_random_50segpaths_50ts_tsd
    #print("dataset type:" + str(test_data_final1.type))
    #test_data_final = unify_time_steps(test_data_final1, time_steps=50)
    test_loader_final = DataLoader(test_data_final, batch_size=32, shuffle=False)
    my_model = load_trained_node(ROOT_DIR + "/NODE_models/NODE_RiemMSE_m2m_14demos_samemixeddata_wg49_we1_finetuning_4000ep_EGI_Exp1.pth")
    #NODE_RiemMSE_m2m_14demos_mixed_data_2500ep_EGI
    #NODE_m2m_14demos_mixed_data_4000ep_EGI_Exp1
    visualize_metric(vae_model)
    visualize_path_comparison(my_model, vae_model, test_loader_final, num_samples=5)
    visualize_random_points_comparison(my_model, vae_model, test_loader_final)


def test_node(trained_model, vae_model, test_loader, save_dir, device='cpu', space_name='', latent_frame=10):
    """
    Takes the already-loaded models and data, and runs the visualizations.
    """
    print("\n--- Starting Testing & Visualizations ---")

    # Ensure models are in eval mode and on the right device
    trained_model.eval()
    vae_model.eval()
    vae_model = vae_model.to(device)

    # Run your plotting functions
    print("Generating Manifold Visualization...")
    visualize_metric(vae_model, space_name, latent_frame)
    save_test_plot(save_dir=save_dir, filename=f"{space_name} Manifold.svg")

    print("Generating Path Comparisons...")
    number_samples = 5
    visualize_path_comparison(trained_model, vae_model, test_loader, space_name, latent_frame, num_samples=number_samples)
    save_test_plot(save_dir=save_dir, filename=f"{number_samples}-Path Comparisons.svg")
    number_samples = 30
    visualize_path_comparison(trained_model, vae_model, test_loader, space_name, latent_frame, num_samples=number_samples)
    save_test_plot(save_dir=save_dir, filename=f"{number_samples}-Path Comparisons.svg")

    print("Generating Random Points Comparison...")
    visualize_random_points_comparison(trained_model, vae_model, test_loader, space_name, latent_frame)
    save_test_plot(save_dir=save_dir, filename=f"Trajectory with random start-end point.svg")


    print("✅ Testing complete!")