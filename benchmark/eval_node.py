import os
import torch
from run_benchmark.data_utils import load_test_config


def run_rnode_benchmark(dataset_cfg, fw_cfg, dataset_type, shape_name, results_dir, device):
    print("\n" + "-" * 40)
    print(f"🧠 RNODE Benchmark Evaluation - {dataset_type.upper()} ({shape_name})")
    print("-" * 40)

    # 1. LOAD DATA
    gt_data_path = dataset_cfg['gt_data'].replace('{shape}', shape_name)
    print(f"Attempting to load data from: {gt_data_path}")

    z1, z2, ground_truth = load_test_config(full_path=gt_data_path)

    z1 = z1.to(device)
    z2 = z2.to(device)
    ground_truth = ground_truth.to(device)

    print(f"✅ SUCCESS! Data loaded.")
    print(f"z1 shape: {z1.shape}")
    print(f"z2 shape: {z2.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")
    print("Next step: We will add the RNODE model inference here.")