#!/bin/bash

# Benchmark sweep over LASA shapes x VAE beta scales.
# Stochman and Graph depend only on (shape, beta), so they run once per pair; every NODE
# variant (run_suffix) is then evaluated against the same ground truth and off-manifold reference.
SHAPES=("N" "P")  # e.g. ("Angle" "Leaf_1" "N" "P" "Sine")
BETAS=(1 5 10)
NODE_VARIANTS=("energy_imitation" "energy_only")
DEVICE="auto"  # set to "cpu" (or "cuda") for a second pass of CPU/GPU timings

# Run from the repo root (configs use relative paths) with the geodesic1 env, unless PYTHON is set
cd "$(dirname "$0")"
PYTHON="${PYTHON:-/home/acm/miniconda3/envs/geodesic1/bin/python}"
# Non-interactive plotting backend: plt.show() must not block unattended runs (figures are still saved)
export MPLBACKEND=Agg

# Failed runs are collected here and listed at the end; the sweep continues with the next run
FAILED=()

run() {
    local label=$1
    shift
    "$PYTHON" run_benchmark.py "$@" --device "$DEVICE"
    if [ $? -ne 0 ]; then
        echo "❌ CRASH DETECTED during $label"
        FAILED+=("$label")
    fi
}

for shape in "${SHAPES[@]}"; do
    for beta in "${BETAS[@]}"; do
        echo "================================================================="
        echo "🚀 BENCHMARK: Shape = $shape | Beta = $beta"
        echo "================================================================="

        run "STOCHMAN Shape: $shape, Beta: $beta" --framework stochman --dataset lasa --shape "$shape" --beta_scale "$beta"
        run "GRAPH    Shape: $shape, Beta: $beta" --framework graph --dataset lasa --shape "$shape" --beta_scale "$beta"
        run "NODE     Shape: $shape, Beta: $beta" --framework node --dataset lasa --shape "$shape" --beta_scale "$beta" \
            --node_variants "${NODE_VARIANTS[@]}"
    done
done

# One table of everything benchmarked so far
"$PYTHON" benchmark/collect_results.py

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "🎉 ALL BENCHMARK RUNS COMPLETED!"
else
    echo "================================================================="
    echo "⚠️ Sweep finished with ${#FAILED[@]} failed run(s):"
    for f in "${FAILED[@]}"; do echo "   ❌ $f"; done
    echo "================================================================="
    exit 1
fi
