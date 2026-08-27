import torch
from torch.utils.data import Dataset, DataLoader
import os
from node.node_model import GoalConditionedNODE
from node.utils.plots import visualize_metric, visualize_path_comparison, visualize_random_points_comparison


################################################### Testing My RiemMSE NODE Playground
class LatentTrajectoryDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = torch.load(file_path, weights_only=True)

        # p1: [2], p2: [2], z: [T, 2]
        return data['p1'].float(), data['p2'].float(), data['z'].float()

#Loading the trained RNODE
def load_trained_node(model_path, latent_dim=2, device='cpu'):
    # 1. Recreate the architecture
    model = GoalConditionedNODE(latent_dim=latent_dim)

    # 2. Load the state dictionary (the weights)
    # Use weights_only=True for security as we discussed
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    # 3. Load weights into the model
    model.load_state_dict(state_dict)

    # 4. Set to evaluation mode
    model.to(device)
    model.eval()

    print(f"Successfully loaded model from {model_path}")
    return model

def test_node(vae_model):
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
