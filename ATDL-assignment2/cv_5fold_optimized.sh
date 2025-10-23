#!/bin/bash

################################################################################
# Optimized 5-Fold Cross-Validation Script for DDPM Segmentation
#
# This script demonstrates how to use the common_functions.sh library to reduce
# code duplication. It provides a flexible template for running k-fold experiments.
#
# Requirements:
#   - Python 3.7+
#   - CUDA-capable GPU
#   - Sufficient disk space for datasets and checkpoints
#
# Usage:
#   bash cv_5fold_optimized.sh [--dry-run]
#
################################################################################

set -e # Exit immediately if a command exits with a non-zero status
set -u # Treat unset variables as an error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common functions
source "${SCRIPT_DIR}/common_functions.sh"

# --- Configuration ---
DATASET="${DATASET:-bedroom_28}"
CHECKPOINT="${CHECKPOINT:-lsun_bedroom}"
N_FOLDS="${N_FOLDS:-5}"
SEED="${SEED:-42}"
TRAIN_SIZE="${TRAIN_SIZE:-}"  # Empty means maintain original ratio
MODEL_NUM="${MODEL_NUM:-10}"  # Number of MLP heads in ensemble
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

################################################################################
# Main Execution
################################################################################

echo "=========================================="
echo "  Optimized K-Fold Cross-Validation"
echo "  Dataset: ${DATASET}"
echo "  Folds: ${N_FOLDS}"
echo "  Seed: ${SEED}"
if [ -n "${TRAIN_SIZE}" ]; then
    echo "  Training images per fold: ${TRAIN_SIZE}"
else
    echo "  Strategy: Maintain original train/test ratio"
fi
echo "  Model ensemble size: ${MODEL_NUM}"
echo "=========================================="

# Step 1: Environment Setup
setup_environment

# Step 2: Clone Repository
clone_repository

# Step 3: Download Datasets
download_datasets

# Step 4: Download Checkpoint
download_checkpoint "${CHECKPOINT}"

# Step 5: Split Dataset into Folds
echo ""
echo "=========================================="
echo "  Splitting Dataset into Folds"
echo "=========================================="

if [ -n "${TRAIN_SIZE}" ]; then
    echo "Using custom train size: ${TRAIN_SIZE}"
    python3 ../split_dataset_optimized.py \
        --dataset "${DATASET}" \
        --n_folds ${N_FOLDS} \
        --train_size ${TRAIN_SIZE} \
        --seed ${SEED} \
        --base_dir datasets
else
    echo "Maintaining original train/test ratio"
    python3 ../split_dataset_optimized.py \
        --dataset "${DATASET}" \
        --n_folds ${N_FOLDS} \
        --seed ${SEED} \
        --base_dir datasets
fi

echo "Dataset splitting complete."

# Step 6: Run Experiments for Each Fold
echo ""
echo "=========================================="
echo "  Running ${N_FOLDS}-Fold Experiments"
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

    # Verify fold directories exist
    verify_directories "${TRAIN_DIR}" "${TEST_DIR}"

    # Count training and test samples
    TRAIN_COUNT=$(count_images "${TRAIN_DIR}")
    TEST_COUNT=$(count_images "${TEST_DIR}")
    echo "Training samples: ${TRAIN_COUNT}"
    echo "Testing samples: ${TEST_COUNT}"

    # Create temporary config file for this fold
    TEMP_CONFIG="experiments/${DATASET}/ddpm_temp.json"

    echo "Creating config for fold ${i}..."
    rm -f "${TEMP_CONFIG}"

    # Create config with optional parameters
    if [ -n "${TRAIN_SIZE}" ]; then
        create_temp_config "${BASE_CONFIG_FILE}" "${TEMP_CONFIG}" \
            "${TRAIN_DIR}" "${TEST_DIR}" "pixel_classifiers/${DATASET}" \
            "s|\\\"training_number\\\": [0-9]*|\\\"training_number\\\": ${TRAIN_SIZE}|g" \
            "s|\\\"model_num\\\": [0-9]*|\\\"model_num\\\": ${MODEL_NUM}|g"
    else
        create_temp_config "${BASE_CONFIG_FILE}" "${TEMP_CONFIG}" \
            "${TRAIN_DIR}" "${TEST_DIR}" "pixel_classifiers/${DATASET}" \
            "s|\\\"model_num\\\": [0-9]*|\\\"model_num\\\": ${MODEL_NUM}|g"
    fi

    echo "Config file created: ${TEMP_CONFIG}"

    # Extract steps and blocks to build experiment directory path
    STEPS=$(extract_config_params "${TEMP_CONFIG}" "steps")
    BLOCKS=$(extract_config_params "${TEMP_CONFIG}" "blocks")
    FULL_EXP_DIR="pixel_classifiers/${DATASET}/${STEPS}_${BLOCKS}"
    mkdir -p "${FULL_EXP_DIR}"
    echo "Using experiment directory: ${FULL_EXP_DIR}"

    # Run training for this fold
    run_training "${TEMP_CONFIG}" "${SEED}" "${DRY_RUN}"

    # Clean up model files if not dry run
    if [ "${DRY_RUN}" = false ]; then
        cleanup_models "${FULL_EXP_DIR}"
    fi
done

# Step 7: Print Results Summary
print_results_summary "${DATASET}" "${BASE_CONFIG_FILE}" "pixel_classifiers"

cd ..
