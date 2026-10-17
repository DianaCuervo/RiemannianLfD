#!/bin/bash

# Define the beta scales you want to test for the toy dataset
BETAS=(1 5 10)

# Replace 'main_new.py' with the actual filename of your script if it's different
SCRIPT_NAME="main_new.py"

# Run from the repo root with the venv's Python, so the sweep works without an activated venv
cd "$(dirname "$0")"
PYTHON="python"
# Non-interactive plotting backend: plt.show() must not block unattended runs (figures are still saved)
export MPLBACKEND=Agg

# Failed runs are collected here and listed at the end; the sweep continues with the next run
FAILED=()

for beta in "${BETAS[@]}"; do

    echo "===================================================="
    echo "🚀 Starting TRAINING: Dataset = toy | Beta = $beta"
    echo "===================================================="
    "$PYTHON" "$SCRIPT_NAME" --mode train --dataset toy --beta_scale "$beta"

    # Check if training crashed (exit code != 0): skip its test, continue with the next run
    if [ $? -ne 0 ]; then
        echo "❌ CRASH DETECTED during Training for Beta: $beta (skipping its test)"
        FAILED+=("TRAIN  Dataset: toy, Beta: $beta")
        continue
    fi

    echo "===================================================="
    echo "🧪 Starting TESTING: Dataset = toy | Beta = $beta"
    echo "===================================================="
    "$PYTHON" "$SCRIPT_NAME" --mode test --dataset toy --beta_scale "$beta"

    # Check if testing crashed
    if [ $? -ne 0 ]; then
        echo "❌ CRASH DETECTED during Testing for Beta: $beta"
        FAILED+=("TEST   Dataset: toy, Beta: $beta")
    fi

done

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "🎉 All beta iterations for the toy dataset completed successfully!"
else
    echo "===================================================="
    echo "⚠️ Sweep finished with ${#FAILED[@]} failed run(s):"
    for f in "${FAILED[@]}"; do echo "   ❌ $f"; done
    echo "===================================================="
    exit 1
fi