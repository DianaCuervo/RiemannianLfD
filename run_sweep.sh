#!/bin/bash

# Define the LASA shapes and beta scales you want to test
SHAPES=("Angle" "Leaf_1" "N" "P" "Sine")  # Add more shapes here if needed (e.g., "P", "Sine", "Leaf_1")
BETAS=(1 5 10)

# Replace 'main_new.py' with the actual filename of your script if it's different
SCRIPT_NAME="main_new.py"

for shape in "${SHAPES[@]}"; do
    for beta in "${BETAS[@]}"; do

        echo "===================================================="
        echo "🚀 Starting TRAINING: Shape = $shape | Beta = $beta"
        echo "===================================================="
        python "$SCRIPT_NAME" --mode train --dataset lasa --shape "$shape" --beta_scale "$beta"

        # Check if training crashed (exit code != 0)
        if [ $? -ne 0 ]; then
            echo "❌ CRASH DETECTED during Training for Shape: $shape, Beta: $beta"
            exit 1
        fi

        echo "===================================================="
        echo "🧪 Starting TESTING: Shape = $shape | Beta = $beta"
        echo "===================================================="
        python "$SCRIPT_NAME" --mode test --dataset lasa --shape "$shape" --beta_scale "$beta"

        # Check if testing crashed
        if [ $? -ne 0 ]; then
            echo "❌ CRASH DETECTED during Testing for Shape: $shape, Beta: $beta"
            exit 1
        fi

    done
done

echo "🎉 All shapes and beta iterations completed successfully without crashing!"