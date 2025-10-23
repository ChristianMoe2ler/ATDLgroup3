#!/bin/bash

################################################################################
# 5-Fold Cross-Validation Training Script for DDPM Segmentation
#
# This script sets up the environment, downloads datasets and checkpoints,
# splits the data into 5 folds, and trains pixel classifiers on each fold.
#
# Requirements:
#   - Python 3.7+
#   - CUDA-capable GPU
#   - Sufficient disk space for datasets and checkpoints
#
# Usage:
#   bash run_5_fold_experiments.sh [--dry-run]
#
################################################################################

set -e # Exit immediately if a command exits with a non-zero status
set -u # Treat unset variables as an error

# --- Configuration ---
DATASET="bedroom_28"
CHECKPOINT="lsun_bedroom"
N_FOLDS=5
SEED=42
TRAINING_IMAGES=1  # Number of training images to use per fold
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

echo "Copying custom scripts into repository..."
cp split_dataset_into_folds2.py ddpm-segmentation/

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
# 4. Split Dataset into Folds
################################################################################

echo ""
echo "=========================================="
echo "  STEP 4: Splitting Dataset into Folds"
echo "=========================================="

echo "Combining train/test data and splitting ${DATASET} into ${N_FOLDS} folds with ${TRAINING_IMAGES} training images per fold..."
python3 split_dataset_into_folds2.py \
    --dataset "${DATASET}" \
    --n_folds ${N_FOLDS} \
    --train_size ${TRAINING_IMAGES} \
    --seed ${SEED} \
    --base_dir datasets

echo "Dataset splitting complete."

################################################################################
# 5. Run Experiments for Each Fold
################################################################################

echo ""
echo "=========================================="
echo "  STEP 5: Running ${N_FOLDS}-Fold Experiments"
echo "  Training images per fold: ${TRAINING_IMAGES}"
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
    echo "  FOLD ${i}/${N_FOLDS}"
    echo "=========================================="

    # Use fold_i's train and test directories
    TRAIN_DIR="datasets/${DATASET}/real/fold_${i}/train"
    TEST_DIR="datasets/${DATASET}/real/fold_${i}/test"

    # Clean up any old temp configs
    echo "Cleaning up old config..."
    rm -f "experiments/${DATASET}/ddpm_temp.json"

    # Verify fold directories exist
    if [ ! -d "${TRAIN_DIR}" ]; then
        echo "ERROR: Training directory not found: ${TRAIN_DIR}"
        exit 1
    fi
    if [ ! -d "${TEST_DIR}" ]; then
        echo "ERROR: Testing directory not found: ${TEST_DIR}"
        exit 1
    fi

    # Count training and test samples (handle both .png and .jpg)
    TRAIN_COUNT=$(find "${TRAIN_DIR}" -maxdepth 1 \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l | tr -d ' ')
    TEST_COUNT=$(find "${TEST_DIR}" -maxdepth 1 \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l | tr -d ' ')
    echo "Available training samples: ${TRAIN_COUNT}"
    echo "Testing samples: ${TEST_COUNT}"
    echo "Using ${TRAINING_IMAGES} training image(s)"

    # Create temporary config file for this fold
    TEMP_CONFIG="experiments/${DATASET}/ddpm_temp.json"

    echo "Creating config for fold ${i}..."
    sed -e "s|\"training_path\": \".*\"|\"training_path\": \"${TRAIN_DIR}\"|g" \
        -e "s|\"testing_path\": \".*\"|\"testing_path\": \"${TEST_DIR}\"|g" \
        -e "s|\"training_number\": [0-9]*|\"training_number\": ${TRAINING_IMAGES}|g" \
        -e "s|\"exp_dir\": \".*\"|\"exp_dir\": \"pixel_classifiers/${DATASET}\"|g" \
        "${BASE_CONFIG_FILE}" > "${TEMP_CONFIG}"

    echo "Config file created: ${TEMP_CONFIG}"
    echo ""

    # Pre-create the experiment directory structure
    # The train_interpreter.py adds a suffix based on steps and blocks
    # Extract steps and blocks from config to build the full path
    STEPS=$(grep -o '"steps":\s*\[[^]]*\]' "${TEMP_CONFIG}" | sed 's/[^0-9,]//g' | tr ',' '_')
    BLOCKS=$(grep -o '"blocks":\s*\[[^]]*\]' "${TEMP_CONFIG}" | sed 's/[^0-9,]//g' | tr ',' '_')
    FULL_EXP_DIR="pixel_classifiers/${DATASET}/${STEPS}_${BLOCKS}"
    mkdir -p "${FULL_EXP_DIR}"
    echo "Using experiment directory: ${FULL_EXP_DIR}"

    # Run training for this fold
    if [ "${DRY_RUN}" = true ]; then
        echo "*** DRY RUN: Would run training with command:"
        echo "    python3 train_interpreter.py \\"
        echo "        --exp \"${TEMP_CONFIG}\" \\"
        echo "        ${MODEL_FLAGS} \\"
        echo "        --seed ${SEED}"
        echo "*** Skipping actual training ***"
    else
        echo "Starting training for fold ${i}..."
        python3 train_interpreter.py \
            --exp "${TEMP_CONFIG}" \
            ${MODEL_FLAGS} \
            --seed ${SEED}
        echo "Fold ${i} training complete."

        # Clean up model files after evaluation to ensure next fold trains fresh models
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
RESULTS_DIR="pixel_classifiers/${DATASET}/${STEPS}_${BLOCKS}"

echo ""
echo "NOTE: Model files (.pth) were cleaned up after each fold to save space."
echo "Results and predictions from all folds are preserved in:"
echo "  ${RESULTS_DIR}"
echo ""
echo "To see mIoU results from all folds:"
echo "  grep 'Overall mIoU' ${RESULTS_DIR}/*.txt"
echo ""

cd ..
