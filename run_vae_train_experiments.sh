#!/bin/bash

# Function to run the training loop for a specific dataset
run_dataset() {
    local dataset=$1
    shift
    local shapes=("$@")

    for shape in "${shapes[@]}"; do
        echo "================================================================="
        echo "🚀 STARTING OVERNIGHT RUN FOR: Dataset=$dataset | Shape=$shape"
        echo "================================================================="

        python vae_main.py --mode train --dataset $dataset --shape $shape --overwrite
        # python vae_main.py --mode test --dataset $dataset --shape $shape --num_segments 0 --test_id 3
        # python run_benchmark.py --framework stochman --dataset lasa --shape $shape
        python vae_main.py --mode visualize --dataset $dataset --shape $shape

        echo "✅ Finished training for $dataset - $shape!"
        echo ""
    done
}

# ---------------------------------------------------------
# 1. Define shapes for LASA
# ---------------------------------------------------------
# LASA_SHAPES=("Angle" "N" "P" "Leaf_1" "Sine")

LASA_SHAPES=("P")
# TASKS=("pick" "place")

# # ---------------------------------------------------------
# # 2. Define shapes for TOY
# # (Change these placeholder names to your actual toy shapes)
# # ---------------------------------------------------------
# TOY_SHAPES=("None")

# ---------------------------------------------------------
# Execute the runs
# ---------------------------------------------------------
# run_dataset "lerobot" "${TASKS[@]}"
run_dataset "lasa" "${LASA_SHAPES[@]}"
#run_dataset "toy" "${TOY_SHAPES[@]}"

echo "🎉 ALL OVERNIGHT TRAINING RUNS COMPLETED!"