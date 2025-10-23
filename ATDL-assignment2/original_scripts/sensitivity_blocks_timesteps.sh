#!/bin/bash

################################################################################
# Sensitivity Analysis for DDPM Segmentation
#
# This script performs a sensitivity study on diffusion timesteps and UNet blocks
# to determine which features are most informative for pixel classification.
# It tests each combination of block and timestep using the standard train/test split.
#
# Configuration:
#   - Uses single MLP head (model_num=1) for faster training
#   - Auto-detects feature dimensions for each block/timestep combination
#   - Cleans up model files after each experiment to save disk space
#
# Requirements:
#   - Python 3.7+
#   - CUDA-capable GPU
#   - Sufficient disk space for datasets and checkpoints
#
# Usage:
#   bash run_5_fold_experiments4.sh [--dry-run]
#
################################################################################

set -e # Exit immediately if a command exits with a non-zero status
set -u # Treat unset variables as an error

# --- Configuration ---
DATASET="bedroom_28"
CHECKPOINT="lsun_bedroom"
SEED=42
DRY_RUN=false

# Sensitivity analysis parameters
# Select 4 representative blocks spanning the UNet architecture
BLOCKS=(2 6 10 14)

# 2. Select only 5 key timesteps that span the whole range
TIMESTEPS=(50 250 500 750 950)

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

echo "Patching train_interpreter.py for auto-dimension detection..."
cp train_interpreter.py train_interpreter.py.bak

# Use a more reliable Python-based patching approach
python3 << 'PYTHON_PATCH'
import sys
import re

try:
    with open('train_interpreter.py', 'r') as f:
        content = f.read()

    # Check if already patched
    if 'Auto-detected feature dimension' in content:
        print("  train_interpreter.py already patched, skipping...")
        sys.exit(0)

    # Pattern 1: Find and replace the X allocation line
    # Look for: X = torch.zeros((len(dataset), *args['dim'][::-1]), dtype=torch.float)
    pattern1 = r"(\s+)X = torch\.zeros\(\(len\(dataset\), \*args\['dim'\]\[::?-?1\]\), dtype=torch\.float\)"
    replacement1 = r"\1# Initialize with placeholder - will be resized after first feature extraction\n\1X = None"

    content, n1 = re.subn(pattern1, replacement1, content, count=1)

    if n1 == 0:
        print("ERROR: Could not find X allocation line to patch!")
        print("Looking for pattern: X = torch.zeros((len(dataset), *args['dim'][::-1]), dtype=torch.float)")
        sys.exit(1)

    print(f"  Step 1: Replaced X allocation line")

    # Pattern 2: Find and replace the collect_features line
    # Look for: X[row] = collect_features(args, features).cpu()
    pattern2 = r"(\s+)X\[row\] = collect_features\(args, features\)\.cpu\(\)"
    replacement2 = r'''\1collected_features = collect_features(args, features).cpu()

\1# Auto-detect dimension from first sample and allocate tensor
\1if X is None:
\1    actual_dim = collected_features.shape
\1    print(f'Auto-detected feature dimension: {actual_dim}')
\1    X = torch.zeros((len(dataset), *actual_dim), dtype=torch.float)
\1    # Update args['dim'] with actual dimension for later use
\1    args['dim'] = [actual_dim[1], actual_dim[2], actual_dim[0]]

\1X[row] = collected_features'''

    content, n2 = re.subn(pattern2, replacement2, content, count=1)

    if n2 == 0:
        print("ERROR: Could not find collect_features line to patch!")
        print("Looking for pattern: X[row] = collect_features(args, features).cpu()")
        sys.exit(1)

    print(f"  Step 2: Replaced collect_features line with auto-detection logic")

    # Write the patched content
    with open('train_interpreter.py', 'w') as f:
        f.write(content)

    print("  Successfully patched train_interpreter.py")

except Exception as e:
    print(f"ERROR: Patching failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_PATCH

if [ $? -ne 0 ]; then
    echo "ERROR: Patching failed!"
    exit 1
fi

# Verify the patch was applied
echo "Verifying patch was applied..."
if grep -q "Auto-detected feature dimension" train_interpreter.py; then
    echo "  ✓ Patch verification successful"
else
    echo "  ✗ ERROR: Patch verification failed - 'Auto-detected feature dimension' not found"
    echo "  Showing relevant section of train_interpreter.py:"
    grep -A5 -B5 "collect_features" train_interpreter.py | head -20
    exit 1
fi

echo "Patching complete"

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

# Use the original train/test split from the dataset
TRAIN_DIR="datasets/${DATASET}/real/train"
TEST_DIR="datasets/${DATASET}/real/test"

if [ ! -d "${TRAIN_DIR}" ]; then
    echo "ERROR: Training directory not found: ${TRAIN_DIR}"
    exit 1
fi
if [ ! -d "${TEST_DIR}" ]; then
    echo "ERROR: Testing directory not found: ${TEST_DIR}"
    exit 1
fi

TRAIN_COUNT=$(find "${TRAIN_DIR}" -maxdepth 1 \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l | tr -d ' ')
TEST_COUNT=$(find "${TEST_DIR}" -maxdepth 1 \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l | tr -d ' ')

echo "Training samples: ${TRAIN_COUNT}"
echo "Testing samples: ${TEST_COUNT}"
echo "Dataset verification complete."

################################################################################
# 5. Run Sensitivity Analysis Experiments
################################################################################

echo ""
echo "=========================================="
echo "  STEP 5: Running Sensitivity Analysis"
echo "  Dataset: ${DATASET}"
echo "  Blocks: ${BLOCKS[@]}"
echo "  Timesteps: ${TIMESTEPS[@]}"
echo "  Total experiments: $((${#BLOCKS[@]} * ${#TIMESTEPS[@]}))"
echo "=========================================="

BASE_CONFIG_FILE="experiments/${DATASET}/ddpm.json"

# Verify base config exists
if [ ! -f "${BASE_CONFIG_FILE}" ]; then
    echo "ERROR: Base config file not found: ${BASE_CONFIG_FILE}"
    exit 1
fi

# Initialize CSV file with headers
OUTPUT_FILE="sensitivity_analysis_${DATASET}.csv"
echo "block,timestep,mIoU" > "${OUTPUT_FILE}"
echo "Results will be saved to: ${OUTPUT_FILE}"

# Main sensitivity analysis loop
for block in "${BLOCKS[@]}"; do
    for timestep in "${TIMESTEPS[@]}"; do
        echo ""
        echo "========================================================================"
        echo "  TESTING: Block ${block}, Timestep ${timestep}"
        echo "========================================================================"

        # Clean up any old temp configs
        rm -f "experiments/${DATASET}/ddpm_temp.json"

        # Create temporary config file with single block and timestep
        TEMP_CONFIG="experiments/${DATASET}/ddpm_temp.json"

        echo "Creating config for Block ${block}, Timestep ${timestep}..."
        sed -e "s|\"training_path\": \".*\"|\"training_path\": \"${TRAIN_DIR}\"|g" \
            -e "s|\"testing_path\": \".*\"|\"testing_path\": \"${TEST_DIR}\"|g" \
            -e "s|\"steps\": \[[^]]*\]|\"steps\": [${timestep}]|g" \
            -e "s|\"blocks\": \[[^]]*\]|\"blocks\": [${block}]|g" \
            -e "s|\"model_num\": [0-9]*|\"model_num\": 1|g" \
            -e "s|\"exp_dir\": \".*\"|\"exp_dir\": \"pixel_classifiers/${DATASET}/sensitivity\"|g" \
            "${BASE_CONFIG_FILE}" > "${TEMP_CONFIG}"

        echo "Note: Feature dimensions will be auto-detected from actual block output"

        echo "Config created: single block [${block}], single timestep [${timestep}], single MLP head (model_num=1)"

        # Build experiment directory path
        FULL_EXP_DIR="pixel_classifiers/${DATASET}/sensitivity/${timestep}_${block}"
        mkdir -p "${FULL_EXP_DIR}"
        echo "Using experiment directory: ${FULL_EXP_DIR}"

        # Run training
        if [ "${DRY_RUN}" = true ]; then
            echo "*** DRY RUN: Would run training with command:"
            echo "    python3 train_interpreter.py \\"
            echo "        --exp \"${TEMP_CONFIG}\" \\"
            echo "        ${MODEL_FLAGS} \\"
            echo "        --seed ${SEED}"
            echo "*** Skipping actual training ***"

            # Simulate a result for dry run
            MIOU="0.123"
        else
            echo "Starting training..."
            echo "(This may take several minutes. Progress will be shown below.)"
            echo ""

            # Create temporary log file to capture output while showing it
            TEMP_LOG="/tmp/train_output_${block}_${timestep}_$$.log"

            # Run training with tee to show output AND save it
            python3 train_interpreter.py \
                --exp "${TEMP_CONFIG}" \
                ${MODEL_FLAGS} \
                --seed ${SEED} 2>&1 | tee "${TEMP_LOG}"

            echo ""
            echo "Training complete."

            # Extract mIoU from saved output
            MIOU=$(grep -i "overall miou" "${TEMP_LOG}" | tail -1 | grep -oE "[0-9]+\.[0-9]+")

            if [ -z "${MIOU}" ]; then
                echo "WARNING: Could not extract mIoU from output"
                echo "Checking log file for any mIoU patterns..."
                tail -20 "${TEMP_LOG}"
                MIOU="N/A"
            fi

            echo "Result: mIoU = ${MIOU}"

            # Clean up temporary log file
            rm -f "${TEMP_LOG}"

            # Clean up model files after evaluation
            echo "Cleaning up model files from ${FULL_EXP_DIR}..."
            rm -f "${FULL_EXP_DIR}"/*.pth
            echo "Model cleanup complete. Results and predictions preserved."
        fi

        # Save result to CSV
        echo "${block},${timestep},${MIOU}" >> "${OUTPUT_FILE}"
    done
done

################################################################################
# 6. Results Summary
################################################################################

echo ""
echo "=========================================="
echo "  SENSITIVITY ANALYSIS COMPLETE"
echo "=========================================="

echo ""
echo "Total experiments run: $((${#BLOCKS[@]} * ${#TIMESTEPS[@]}))"
echo "Results saved to: ${OUTPUT_FILE}"
echo ""
echo "Summary statistics:"
echo "  Blocks tested: ${BLOCKS[@]}"
echo "  Timesteps tested: ${TIMESTEPS[@]}"
echo ""
echo "To analyze results:"
echo "  cat ${OUTPUT_FILE}"
echo "  # Or import into Python/R for visualization"
echo ""
echo "NOTE: Model files (.pth) were cleaned up after each experiment to save space."
echo "Results and predictions are preserved in:"
echo "  pixel_classifiers/${DATASET}/sensitivity/"
echo ""

cd ..
