#!/bin/bash

# Function to run the training loop for a specific dataset, once per beta_scale
run_dataset() {
    local dataset=$1
    shift
    local shapes=("$@")

    for shape in "${shapes[@]}"; do
        for beta in "${BETA_SCALES[@]}"; do
            echo "================================================================="
            echo "🚀 STARTING OVERNIGHT RUN FOR: Dataset=$dataset | Shape=$shape | beta_scale=$beta"
            echo "================================================================="

            # python vae_main.py --mode train --dataset $dataset --shape $shape --beta_scale $beta --overwrite
            python vae_main.py --mode test --dataset $dataset --shape $shape --beta_scale $beta --num_segments 0
            # python run_benchmark.py --framework stochman --dataset lasa --shape $shape --beta_scale $beta
            # python vae_main.py --mode visualize --dataset $dataset --shape $shape --beta_scale $beta

            echo "✅ Finished training for $dataset - $shape - beta_scale $beta!"
            echo ""
        done
    done
}

# ---------------------------------------------------------
# 0. Define the RBF beta scales (one VAE per value)
# ---------------------------------------------------------
BETA_SCALES=("1" "5" "10")

# ---------------------------------------------------------
# 1. Define shapes for LASA
# ---------------------------------------------------------
# LASA_SHAPES=("Angle" "P" "Leaf_1")
LASA_SHAPES=("N" "Sine")

# LASA_SHAPES=("P")
# TASKS=("pick" "place")

# # ---------------------------------------------------------
# # 2. Define shapes for TOY
# # (Change these placeholder names to your actual toy shapes)
# # ---------------------------------------------------------
TOY_SHAPES=("None")

# ---------------------------------------------------------
# Execute the runs
# ---------------------------------------------------------
# run_dataset "lerobot" "${TASKS[@]}"
# run_dataset "lasa" "${LASA_SHAPES[@]}"
run_dataset "toy" "${TOY_SHAPES[@]}"

echo "🎉 ALL OVERNIGHT TRAINING RUNS COMPLETED!"
