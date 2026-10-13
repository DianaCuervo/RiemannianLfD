#!/bin/bash

# Define the LeRobot tasks and beta scales you want to test
TASKS=("pick" "place")  # Replace these with your actual LeRobot task names
BETAS=(1 5 10)

# Replace 'main_new.py' with the actual filename of your script if it's different
SCRIPT_NAME="main_new.py"

# Run from the repo root with the active environment's Python
cd "$(dirname "$0")"
PYTHON="python"
# Non-interactive plotting backend: plt.show() must not block unattended runs (figures are still saved)
export MPLBACKEND=Agg

# Failed runs are collected here and listed at the end; the sweep continues with the next run
FAILED=()

for task in "${TASKS[@]}"; do
    for beta in "${BETAS[@]}"; do

        echo "===================================================="
        echo "🚀 Starting TRAINING: Task = $task | Beta = $beta"
        echo "===================================================="
        "$PYTHON" "$SCRIPT_NAME" --mode train --dataset lerobot --task "$task" --beta_scale "$beta"

        # Check if training crashed (exit code != 0): skip its test, continue with the next run
        if [ $? -ne 0 ]; then
            echo "❌ CRASH DETECTED during Training for Task: $task, Beta: $beta (skipping its test)"
            FAILED+=("TRAIN  Task: $task, Beta: $beta")
            continue
        fi

        echo "===================================================="
        echo "🧪 Starting TESTING: Task = $task | Beta = $beta"
        echo "===================================================="
        "$PYTHON" "$SCRIPT_NAME" --mode test --dataset lerobot --task "$task" --beta_scale "$beta"

        # Check if testing crashed
        if [ $? -ne 0 ]; then
            echo "❌ CRASH DETECTED during Testing for Task: $task, Beta: $beta"
            FAILED+=("TEST   Task: $task, Beta: $beta")
        fi

    done
done

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "🎉 All tasks and beta iterations completed successfully without crashing!"
else
    echo "===================================================="
    echo "⚠️ Sweep finished with ${#FAILED[@]} failed run(s):"
    for f in "${FAILED[@]}"; do echo "   ❌ $f"; done
    echo "===================================================="
    exit 1
fi