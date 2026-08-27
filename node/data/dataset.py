import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import shutil

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

#Prepare data loaders with demonstrations
def prepare_loaders(data_dir, batch_size=32, train_ratio=0.8, val_ratio=0.1):
    full_dataset = LatentTrajectoryDataset(data_dir)
    total_size = len(full_dataset)

    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"✅ Dataset Loaded from {data_dir} | Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
#Export data loaders to file system
def export_test_set(test_loader, source_dir, target_dir="./test_datasets"):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    indices = test_loader.dataset.indices
    file_list = test_loader.dataset.dataset.file_list
    for idx in indices:
        filename = file_list[idx]
        shutil.copy2(os.path.join(source_dir, filename), os.path.join(target_dir, filename))
    print(f"Exported {len(indices)} test files to {target_dir}")