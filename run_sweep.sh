#!/bin/bash

# Define the LASA shapes and beta scales you want to test
SHAPES=("Angle" "Leaf_1" "N" "P" "Sine")  # Add more shapes here if needed (e.g., "P", "Sine", "Leaf_1")
BETAS=(5 10)

# Replace 'main_new.py' with the actual filename of your script if it's different
SCRIPT_NAME="main_new.py"

# Run from the repo root with the venv's Python, so the sweep works without an activated venv
cd "$(dirname "$0")"
PYTHON="./venv/bin/python"
# Non-interactive plotting backend: plt.show() must not block unattended runs (figures are still saved)
export MPLBACKEND=Agg

# Failed runs are collected here and listed at the end; the sweep continues with the next run
FAILED=()

for shape in "${SHAPES[@]}"; do
    for beta in "${BETAS[@]}"; do

        echo "===================================================="
        echo "🚀 Starting TRAINING: Shape = $shape | Beta = $beta"
        echo "===================================================="
        "$PYTHON" "$SCRIPT_NAME" --mode train --dataset lasa --shape "$shape" --beta_scale "$beta"

        # Check if training crashed (exit code != 0): skip its test, continue with the next run
        if [ $? -ne 0 ]; then
            echo "❌ CRASH DETECTED during Training for Shape: $shape, Beta: $beta (skipping its test)"
            FAILED+=("TRAIN  Shape: $shape, Beta: $beta")
            continue
        fi

        echo "===================================================="
        echo "🧪 Starting TESTING: Shape = $shape | Beta = $beta"
        echo "===================================================="
        "$PYTHON" "$SCRIPT_NAME" --mode test --dataset lasa --shape "$shape" --beta_scale "$beta"

        # Check if testing crashed
        if [ $? -ne 0 ]; then
            echo "❌ CRASH DETECTED during Testing for Shape: $shape, Beta: $beta"
            FAILED+=("TEST   Shape: $shape, Beta: $beta")
        fi

    done
done

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "🎉 All shapes and beta iterations completed successfully without crashing!"
else
    echo "===================================================="
    echo "⚠️ Sweep finished with ${#FAILED[@]} failed run(s):"
    for f in "${FAILED[@]}"; do echo "   ❌ $f"; done
    echo "===================================================="
    exit 1
fi
