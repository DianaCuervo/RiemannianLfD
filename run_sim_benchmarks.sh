#!/bin/bash

# Simulation benchmark sweep over LASA shapes x VAE beta scales (headless).
# Graph and Stochman depend only on (shape, beta), so they run once per pair; NODE runs once
# per variant (run_suffix). Results: benchmark/results_simulation/<shape>-Shape_beta_scale_<beta>/
SHAPES=("N" "Sine" "Angle" "Leaf_1")  # P is left out for now (see the simulation fix plan)
BETAS=(5 10)
FRAMEWORKS=("graph" "stochman" "node")
NODE_VARIANTS=("energy_imitation" "energy_only")

# Run from the repo root (configs use relative paths) with the geodesic1 env, unless PYTHON is set
cd "$(dirname "$0")"
PYTHON="${PYTHON:-/home/acm/miniconda3/envs/geodesic1/bin/python}"
# Non-interactive plotting backend: nothing is plotted, but some imports load matplotlib
export MPLBACKEND=Agg

# Failed runs are collected here and listed at the end; the sweep continues with the next run
FAILED=()

run() {
    local label=$1
    shift
    "$PYTHON" simulation_benchmark.py "$@" --no_viewer
    if [ $? -ne 0 ]; then
        echo "❌ CRASH DETECTED during $label"
        FAILED+=("$label")
    fi
}

for shape in "${SHAPES[@]}"; do
    for beta in "${BETAS[@]}"; do
        echo "================================================================="
        echo "🚀 SIMULATION: Shape = $shape | Beta = $beta"
        echo "================================================================="

        for fw in "${FRAMEWORKS[@]}"; do
            if [ "$fw" == "node" ]; then
                for variant in "${NODE_VARIANTS[@]}"; do
                    run "NODE ($variant) Shape: $shape, Beta: $beta" --framework node \
                        --shape "$shape" --beta_scale "$beta" --node_variant "$variant"
                done
            else
                run "${fw^^} Shape: $shape, Beta: $beta" --framework "$fw" --shape "$shape" --beta_scale "$beta"
            fi
        done
    done
done

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "🎉 ALL SIMULATION RUNS COMPLETED!"
else
    echo "================================================================="
    echo "⚠️ Sweep finished with ${#FAILED[@]} failed run(s):"
    for f in "${FAILED[@]}"; do echo "   ❌ $f"; done
    echo "================================================================="
    exit 1
fi
