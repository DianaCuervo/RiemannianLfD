# #!/bin/bash

# # Function to run the training loop for a specific dataset
# run_dataset() {
#     local dataset=$1
#     shift
#     local shapes=("$@")

#     for shape in "${shapes[@]}"; do
#         echo "================================================================="
#         echo "🚀 STARTING OVERNIGHT RUN FOR: Dataset=$dataset | Shape=$shape"
#         echo "================================================================="

#         python vae_main.py --mode train --dataset $dataset --task $shape --overwrite
#         python vae_main.py --mode test --dataset $dataset --task $shape --num_segments 0
#         # python run_benchmark.py --framework stochman --dataset lasa --shape $shape
#         python vae_main.py --mode visualize --dataset $dataset --task $shape

#         echo "✅ Finished training for $dataset - $shape!"
#         echo ""
#     done
# }

# # ---------------------------------------------------------
# # 1. Define shapes for LASA
# # ---------------------------------------------------------
# # LASA_SHAPES=("Angle" "P" "Leaf_1")
# # LASA_SHAPES=("N" "Sine")

# # LASA_SHAPES=("P")
# TASKS=("pick" "place")

# # # ---------------------------------------------------------
# # # 2. Define shapes for TOY
# # # (Change these placeholder names to your actual toy shapes)
# # # (the toy beta_scale is selected in config_files/vae_config.yaml)
# # # ---------------------------------------------------------
# # TOY_SHAPES=("None")

# # ---------------------------------------------------------
# # Execute the runs
# # ---------------------------------------------------------
# run_dataset "lerobot" "${TASKS[@]}"
# # run_dataset "lasa" "${LASA_SHAPES[@]}"
# # run_dataset "toy" "${TOY_SHAPES[@]}"

# echo "🎉 ALL OVERNIGHT TRAINING RUNS COMPLETED!"

PYTHON=/home/acm/miniconda3/envs/geodesic1/bin/python
export MPLBACKEND=Agg
CONFIG=config_files/vae_config.yaml

# Restore the config when the sweep ends, even after a crash or Ctrl-C
cp "$CONFIG" "$CONFIG.sweep_backup"
trap 'mv "$CONFIG.sweep_backup" "$CONFIG"' EXIT

set_lerobot_beta() {   # set beta_scale in the lerobot pick + place blocks only
    "$PYTHON" - "$CONFIG" "$1" <<'EOF'
import re, sys
p, beta = sys.argv[1], sys.argv[2]
s = open(p).read()
head, sep, tail = s.partition('\nlerobot:')
tail, n = re.subn(r'(\n\s+beta_scale:\s*)[^\s#]+', rf'\g<1>{beta}', tail)
assert n == 2, f"expected 2 lerobot beta_scale lines, found {n}"
open(p, 'w').write(head + sep + tail)
EOF
}

run_dataset() {
    local dataset=$1 flag=$2; shift 2
    for item in "$@"; do
        "$PYTHON" vae_main.py --mode train --dataset $dataset $flag $item --overwrite
        "$PYTHON" vae_main.py --mode test --dataset $dataset $flag $item --num_segments 0
        "$PYTHON" vae_main.py --mode visualize --dataset $dataset $flag $item
    done
}

LEROBOT_BETAS=(5 10)
for beta in "${LEROBOT_BETAS[@]}"; do
    set_lerobot_beta "$beta" || exit 1
    run_dataset "lerobot" --task place pick
done
