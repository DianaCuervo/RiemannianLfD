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

        python main.py --mode train --dataset $dataset --shape $shape

        echo "✅ Finished training for $dataset - $shape!"
        echo ""
    done
}

# ---------------------------------------------------------
# 1. Define shapes for LASA
# ---------------------------------------------------------
LASA_SHAPES=("Angle" "N" "P" "Leaf_1")

# ---------------------------------------------------------
# 2. Define shapes for TOY
# (Change these placeholder names to your actual toy shapes)
# ---------------------------------------------------------
TOY_SHAPES=("None")

# ---------------------------------------------------------
# Execute the runs
# ---------------------------------------------------------
run_dataset "lasa" "${LASA_SHAPES[@]}"
run_dataset "toy" "${TOY_SHAPES[@]}"

echo "🎉 ALL OVERNIGHT TRAINING RUNS COMPLETED!"