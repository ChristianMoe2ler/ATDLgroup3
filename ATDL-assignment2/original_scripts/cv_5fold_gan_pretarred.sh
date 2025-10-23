#!/bin/bash

################################################################################
# 5-Fold Cross-Validation Training Script for DDPM Segmentation
# Using GAN SYNTHETIC Datasets (DatasetGAN)
#
# This script sets up the environment, downloads checkpoints, extracts pre-created
# fold tar files, and trains pixel classifiers on each fold.
#
# Requirements:
#   - Python 3.7+
#   - CUDA-capable GPU
#   - Sufficient disk space for datasets and checkpoints
#   - Pre-created fold tar files (fold_1.tar.gz ... fold_5.tar.gz) at parent level
#
# Usage:
#   bash run_5_fold_experiments5.sh [--dry-run]
#
################################################################################

set -e # Exit immediately if a command exits with a non-zero status
set -u # Treat unset variables as an error

# --- Configuration ---
DATASET="bedroom_28"
CHECKPOINT="lsun_bedroom"
N_FOLDS=5
SEED=42
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

# Check if GAN dataset is already mounted (for container environment)
NPZ_CHECK="synthetic_datasets/gan/${DATASET}/samples_256x256x3.npz"
if [ -f "${NPZ_CHECK}" ]; then
    echo "GAN synthetic dataset already available (pre-mounted)"
else
    echo "Downloading GAN SYNTHETIC dataset: ${DATASET}..."
    chmod +x ./synthetic_datasets/gan/download_synthetic_dataset.sh
    bash synthetic_datasets/gan/download_synthetic_dataset.sh "${DATASET}"
fi

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
# 4. Download Real Dataset (for testing)
################################################################################

echo ""
echo "=========================================="
echo "  STEP 4: Downloading Real Dataset for Testing"
echo "=========================================="

# Download real dataset - will be used for testing
REAL_TRAIN_DIR="datasets/${DATASET}/real/train"
REAL_TEST_DIR="datasets/${DATASET}/real/test"

if [ ! -d "${REAL_TRAIN_DIR}" ]; then
    echo "Downloading real dataset..."
    chmod +x ./datasets/download_datasets.sh
    ./datasets/download_datasets.sh
else
    echo "Real dataset already exists"
fi

# Count training images to determine synthetic sample size
TRAIN_COUNT=$(find "${REAL_TRAIN_DIR}" -maxdepth 1 \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l | tr -d ' ')
TEST_COUNT=$(find "${REAL_TEST_DIR}" -maxdepth 1 \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l | tr -d ' ')

echo "Real dataset size:"
echo "  Train: ${TRAIN_COUNT} images (will use synthetic instead)"
echo "  Test: ${TEST_COUNT} images (will use REAL for testing)"
echo ""
echo "Strategy: Train on ${TRAIN_COUNT} GAN synthetic samples, Test on ${TEST_COUNT} REAL samples"

################################################################################
# 5. Extract Pre-Created Folds from Tar Files
################################################################################

echo ""
echo "=========================================="
echo "  STEP 5: Extracting Pre-Created Folds"
echo "=========================================="

EXTRACTED_DIR="synthetic_datasets/gan/${DATASET}/extracted"
mkdir -p "${EXTRACTED_DIR}"

# Check if fold tar files exist at the parent directory level
FOLD_TAR_DIR=".."

echo "Extracting ${N_FOLDS} pre-created folds from tar files..."

for i in $(seq 1 ${N_FOLDS})
do
    FOLD_TAR="${FOLD_TAR_DIR}/fold_${i}.tar.gz"

    if [ ! -f "${FOLD_TAR}" ]; then
        echo "ERROR: Fold tar file not found: ${FOLD_TAR}"
        echo "Expected location: Parent directory of this script"
        exit 1
    fi

    echo "Extracting fold ${i} from ${FOLD_TAR}..."
    tar -xzf "${FOLD_TAR}" -C "${EXTRACTED_DIR}"

    # Verify extraction
    TRAIN_DIR="${EXTRACTED_DIR}/fold_${i}/train"
    TEST_DIR="${EXTRACTED_DIR}/fold_${i}/test"

    if [ ! -d "${TRAIN_DIR}" ] || [ ! -d "${TEST_DIR}" ]; then
        echo "ERROR: Fold ${i} extraction failed - directories not found"
        exit 1
    fi

    TRAIN_COUNT=$(find "${TRAIN_DIR}" -maxdepth 1 \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l | tr -d ' ')
    TEST_COUNT=$(find "${TEST_DIR}" -maxdepth 1 \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l | tr -d ' ')

    echo "  Fold ${i} extracted: ${TRAIN_COUNT} train, ${TEST_COUNT} test images"
done

echo "All ${N_FOLDS} folds extracted successfully!"
echo "Location: ${EXTRACTED_DIR}"

################################################################################
# 6. Run Experiments for Each Fold
################################################################################

echo ""
echo "=========================================="
echo "  STEP 6: Running ${N_FOLDS}-Fold Experiments"
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

    # Use fold_i's train and test directories from extracted GAN synthetic dataset
    TRAIN_DIR="synthetic_datasets/gan/${DATASET}/extracted/fold_${i}/train"
    TEST_DIR="synthetic_datasets/gan/${DATASET}/extracted/fold_${i}/test"

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
    echo "Training samples: ${TRAIN_COUNT}"
    echo "Testing samples: ${TEST_COUNT}"

    # Create temporary config file for this fold
    TEMP_CONFIG="experiments/${DATASET}/ddpm_temp.json"

    echo "Creating config for fold ${i}..."
    sed -e "s|\"training_path\": \".*\"|\"training_path\": \"${TRAIN_DIR}\"|g" \
        -e "s|\"testing_path\": \".*\"|\"testing_path\": \"${TEST_DIR}\"|g" \
        -e "s|\"exp_dir\": \".*\"|\"exp_dir\": \"pixel_classifiers_gan/${DATASET}\"|g" \
        "${BASE_CONFIG_FILE}" > "${TEMP_CONFIG}"

    echo "Config file created: ${TEMP_CONFIG}"
    echo ""

    # Pre-create the experiment directory structure
    # The train_interpreter.py adds a suffix based on steps and blocks
    # Extract steps and blocks from config to build the full path
    STEPS=$(grep -o '"steps":\s*\[[^]]*\]' "${TEMP_CONFIG}" | sed 's/[^0-9,]//g' | tr ',' '_')
    BLOCKS=$(grep -o '"blocks":\s*\[[^]]*\]' "${TEMP_CONFIG}" | sed 's/[^0-9,]//g' | tr ',' '_')
    FULL_EXP_DIR="pixel_classifiers_gan/${DATASET}/${STEPS}_${BLOCKS}"
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
# 7. Results Summary
################################################################################

echo ""
echo "=========================================="
echo "  ALL EXPERIMENTS COMPLETE"
echo "=========================================="

# Extract steps and blocks to show final results directory
STEPS=$(grep -o '"steps":\s*\[[^]]*\]' "${BASE_CONFIG_FILE}" | sed 's/[^0-9,]//g' | tr ',' '_')
BLOCKS=$(grep -o '"blocks":\s*\[[^]]*\]' "${BASE_CONFIG_FILE}" | sed 's/[^0-9,]//g' | tr ',' '_')
RESULTS_DIR="pixel_classifiers_gan/${DATASET}/${STEPS}_${BLOCKS}"

echo ""
echo "NOTE: This script used GAN SYNTHETIC datasets from DatasetGAN."
echo "Model files were overwritten for each fold."
echo "Results from the last fold (fold ${N_FOLDS}) are in:"
echo "  ${RESULTS_DIR}"
echo ""
echo "To see the final mIoU:"
echo "  grep 'Overall mIoU' ${RESULTS_DIR}/*.txt"
echo ""

cd ..
