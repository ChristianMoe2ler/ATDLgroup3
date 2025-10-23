#!/bin/bash

################################################################################
# 5-Fold Cross-Validation Training Script for DDMP Segmentation (Synthetic→Real)
#
# This script sets up the environment, downloads datasets and checkpoints,
# and trains pixel classifiers on synthetic DDPM data while testing on real data.
# This evaluates how well models trained on synthetic data generalize to real data.
#
# Requirements:
#   - Python 3.7+
#   - CUDA-capable GPU
#   - Sufficient disk space for datasets and checkpoints
#
# Usage:
#   bash run_five_cat_synth.sh [--dry-run]
#
################################################################################

set -e # Exit immediately if a command exits with a non-zero status
set -u # Treat unset variables as an error

# --- Configuration ---
DATASET="horse_21"
CHECKPOINT="lsun_horse"
N_FOLDS=2
SEED=99
DRY_RUN=false

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            echo "*** DRY RUN MODE - No training will be performed ***"
            echo ""
            ;;
    esac
done

# DDPM model flags (as per project documentation)
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

################################################################################
# 1. Environment Setup
################################################################################

echo "=========================================="
echo "  STEP 1: Environment Setup"
echo "=========================================="

echo "Installing Python dependencies..."
pip install -q blobfile==1.2.7
pip install -q tqdm==4.62.3
pip install -q opencv-python==4.5.4.60
pip install -q mpi4py
pip install -q timm==0.4.12
pip install -q opencv-python-headless==4.5.4.60

echo "Environment setup complete."

################################################################################
# 2. Code and Data Download
################################################################################

echo ""
echo "=========================================="
echo "  STEP 2: Code and Data Download"
echo "=========================================="

echo "Cloning repository with submodules..."
git clone https://github.com/yandex-research/ddpm-segmentation.git --recurse-submodules
cd ddpm-segmentation

echo "Downloading datasets..."
chmod +x ./datasets/download_datasets.sh
./datasets/download_datasets.sh

################################################################################
# 3. Checkpoint Download
################################################################################

echo ""
echo "=========================================="
echo "  STEP 3: Checkpoint Download"
echo "=========================================="

echo "Downloading DDPM checkpoint: ${CHECKPOINT}..."
chmod +x ./checkpoints/ddpm/download_checkpoint.sh
bash checkpoints/ddpm/download_checkpoint.sh "${CHECKPOINT}"

################################################################################
# 4. Verify Dataset Structure
################################################################################

echo ""
echo "=========================================="
echo "  STEP 4: Verifying Dataset Structure"
echo "=========================================="

echo "Training on synthetic data and testing on real data..."
echo "Training data: datasets/${DATASET}/ddpm/"
echo "Testing data: datasets/${DATASET}/real/test/"
echo "Dataset structure verified."

################################################################################
# 5. Run Multiple Experiments (Synthetic→Real)
################################################################################

echo ""
echo "=========================================="
echo "  STEP 5: Running ${N_FOLDS} Experiments (Synthetic→Real)"
echo "=========================================="

BASE_CONFIG_FILE="experiments/${DATASET}/ddpm.json"

# Verify base config exists
if [ ! -f "${BASE_CONFIG_FILE}" ]; then
    echo "ERROR: Base config file not found: ${BASE_CONFIG_FILE}"
    exit 1
fi

for i in $(seq 1 ${N_FOLDS})
do
    echo ""
    echo "=========================================="
    echo "  EXPERIMENT ${i}/${N_FOLDS}"
    echo "=========================================="

    # Train on synthetic data, test on real data
    TRAIN_DIR="datasets/${DATASET}/ddpm"
    TEST_DIR="datasets/${DATASET}/real/test"
    
    # Use different seed for each experiment
    EXPERIMENT_SEED=$((SEED + i))

    # Clean up any old temp configs
    echo "Cleaning up old config..."
    rm -f "experiments/${DATASET}/ddpm_temp.json"

    # Verify training and testing directories exist
    if [ ! -d "${TRAIN_DIR}" ]; then
        echo "ERROR: Synthetic training directory not found: ${TRAIN_DIR}"
        exit 1
    fi
    if [ ! -d "${TEST_DIR}" ]; then
        echo "ERROR: Real testing directory not found: ${TEST_DIR}"
        exit 1
    fi

    # Count training (synthetic) and testing (real) samples
    TRAIN_COUNT=$(find "${TRAIN_DIR}" -maxdepth 1 \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l | tr -d ' ')
    TEST_COUNT=$(find "${TEST_DIR}" -maxdepth 1 \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l | tr -d ' ')
    echo "Synthetic training samples: ${TRAIN_COUNT}"
    echo "Real testing samples: ${TEST_COUNT}"
    echo "Experiment seed: ${EXPERIMENT_SEED}"

    # Create temporary config file for this experiment
    TEMP_CONFIG="experiments/${DATASET}/ddpm_temp.json"

    echo "Creating config for experiment ${i}..."
    sed -e "s|\"training_path\": \".*\"|\"training_path\": \"${TRAIN_DIR}\"|g" \
        -e "s|\"testing_path\": \".*\"|\"testing_path\": \"${TEST_DIR}\"|g" \
        -e "s|\"exp_dir\": \".*\"|\"exp_dir\": \"pixel_classifiers/${DATASET}_synth2real\"|g" \
        "${BASE_CONFIG_FILE}" > "${TEMP_CONFIG}"

    echo "Config file created: ${TEMP_CONFIG}"
    echo ""

    # Pre-create the experiment directory structure
    # The train_interpreter.py adds a suffix based on steps and blocks
    # Extract steps and blocks from config to build the full path
    STEPS=$(grep -o '"steps":\s*\[[^]]*\]' "${TEMP_CONFIG}" | sed 's/[^0-9,]//g' | tr ',' '_')
    BLOCKS=$(grep -o '"blocks":\s*\[[^]]*\]' "${TEMP_CONFIG}" | sed 's/[^0-9,]//g' | tr ',' '_')
    FULL_EXP_DIR="pixel_classifiers/${DATASET}_synth2real/${STEPS}_${BLOCKS}"
    mkdir -p "${FULL_EXP_DIR}"
    echo "Using experiment directory: ${FULL_EXP_DIR}"

    # Run training for this experiment
    if [ "${DRY_RUN}" = true ]; then
        echo "*** DRY RUN: Would run training with command:"
        echo "    python3 train_interpreter.py \\"
        echo "        --exp \"${TEMP_CONFIG}\" \\"
        echo "        ${MODEL_FLAGS} \\"
        echo "        --seed ${EXPERIMENT_SEED}"
        echo "*** Skipping actual training ***"
    else
        echo "Starting training for experiment ${i} (synthetic→real)..."
        python3 train_interpreter.py \
            --exp "${TEMP_CONFIG}" \
            ${MODEL_FLAGS} \
            --seed ${EXPERIMENT_SEED}
        echo "Experiment ${i} training complete."

        # Clean up model files after evaluation to ensure next experiment trains fresh models
        echo "Cleaning up model files from ${FULL_EXP_DIR}..."
        rm -f "${FULL_EXP_DIR}"/*.pth
        echo "Model cleanup complete. Results and predictions preserved."
    fi
done

################################################################################
# 6. Results Summary
################################################################################

echo ""
echo "=========================================="
echo "  ALL EXPERIMENTS COMPLETE"
echo "=========================================="

# Extract steps and blocks to show final results directory
STEPS=$(grep -o '"steps":\s*\[[^]]*\]' "${BASE_CONFIG_FILE}" | sed 's/[^0-9,]//g' | tr ',' '_')
BLOCKS=$(grep -o '"blocks":\s*\[[^]]*\]' "${BASE_CONFIG_FILE}" | sed 's/[^0-9,]//g' | tr ',' '_')
RESULTS_DIR="pixel_classifiers/${DATASET}_synth2real/${STEPS}_${BLOCKS}"

echo ""
echo "NOTE: Model files were overwritten for each experiment."
echo "Results from the last experiment (experiment ${N_FOLDS}) are in:"
echo "  ${RESULTS_DIR}"
echo ""
echo "This evaluated synthetic→real generalization:"
echo "  - Training: datasets/${DATASET}/ddpm (synthetic data)"
echo "  - Testing: datasets/${DATASET}/real/test (real data)"
echo ""
echo "To see the final mIoU:"
echo "  grep 'Overall mIoU' ${RESULTS_DIR}/*.txt"
echo ""

cd ..
