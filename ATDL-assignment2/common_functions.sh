#!/bin/bash

################################################################################
# Common Functions Library for DDPM Segmentation Experiments
#
# This library contains shared functions used across multiple experiment scripts
# to reduce code duplication and improve maintainability.
################################################################################

# DDPM model flags (as per project documentation)
export MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

################################################################################
# Environment Setup Function
################################################################################
setup_environment() {
    echo "=========================================="
    echo "  Environment Setup"
    echo "=========================================="

    echo "Installing Python dependencies..."
    pip install -q blobfile==1.2.7
    pip install -q tqdm==4.62.3
    pip install -q opencv-python==4.5.4.60
    pip install -q mpi4py
    pip install -q timm==0.4.12
    pip install -q opencv-python-headless==4.5.4.60

    echo "Environment setup complete."
}

################################################################################
# Repository Clone Function
################################################################################
clone_repository() {
    echo ""
    echo "=========================================="
    echo "  Cloning Repository"
    echo "=========================================="

    echo "Cloning repository with submodules..."
    git clone https://github.com/yandex-research/ddpm-segmentation.git --recurse-submodules
    cd ddpm-segmentation
}

################################################################################
# Checkpoint Download Function
################################################################################
download_checkpoint() {
    local checkpoint=$1

    echo ""
    echo "=========================================="
    echo "  Checkpoint Download"
    echo "=========================================="

    echo "Downloading DDPM checkpoint: ${checkpoint}..."
    chmod +x ./checkpoints/ddpm/download_checkpoint.sh
    bash checkpoints/ddpm/download_checkpoint.sh "${checkpoint}"
}

################################################################################
# Dataset Download Function
################################################################################
download_datasets() {
    echo ""
    echo "=========================================="
    echo "  Dataset Download"
    echo "=========================================="

    echo "Downloading datasets..."
    chmod +x ./datasets/download_datasets.sh
    ./datasets/download_datasets.sh
}

################################################################################
# Verify Directories Function
################################################################################
verify_directories() {
    local train_dir=$1
    local test_dir=$2

    if [ ! -d "${train_dir}" ]; then
        echo "ERROR: Training directory not found: ${train_dir}"
        exit 1
    fi
    if [ ! -d "${test_dir}" ]; then
        echo "ERROR: Testing directory not found: ${test_dir}"
        exit 1
    fi
}

################################################################################
# Count Images Function
################################################################################
count_images() {
    local directory=$1
    find "${directory}" -maxdepth 1 \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l | tr -d ' '
}

################################################################################
# Extract Config Parameters Function
################################################################################
extract_config_params() {
    local config_file=$1
    local param=$2

    grep -o "\"${param}\":\s*\[[^]]*\]" "${config_file}" | sed 's/[^0-9,]//g' | tr ',' '_'
}

################################################################################
# Create Temp Config Function
################################################################################
create_temp_config() {
    local base_config=$1
    local temp_config=$2
    local train_dir=$3
    local test_dir=$4
    local exp_dir=$5
    shift 5

    # Build sed command with additional optional replacements
    local sed_cmd="sed"
    sed_cmd+=" -e \"s|\\\"training_path\\\": \\\".*\\\"|\\\"training_path\\\": \\\"${train_dir}\\\"|g\""
    sed_cmd+=" -e \"s|\\\"testing_path\\\": \\\".*\\\"|\\\"testing_path\\\": \\\"${test_dir}\\\"|g\""
    sed_cmd+=" -e \"s|\\\"exp_dir\\\": \\\".*\\\"|\\\"exp_dir\\\": \\\"${exp_dir}\\\"|g\""

    # Add any extra sed replacements passed as arguments
    for extra_sed in "$@"; do
        sed_cmd+=" -e \"${extra_sed}\""
    done

    sed_cmd+=" \"${base_config}\" > \"${temp_config}\""

    eval "${sed_cmd}"
}

################################################################################
# Run Training Function
################################################################################
run_training() {
    local temp_config=$1
    local seed=$2
    local dry_run=$3

    if [ "${dry_run}" = true ]; then
        echo "*** DRY RUN: Would run training with command:"
        echo "    python3 train_interpreter.py \\"
        echo "        --exp \"${temp_config}\" \\"
        echo "        ${MODEL_FLAGS} \\"
        echo "        --seed ${seed}"
        echo "*** Skipping actual training ***"
    else
        echo "Starting training..."
        python3 train_interpreter.py \
            --exp "${temp_config}" \
            ${MODEL_FLAGS} \
            --seed ${seed}
        echo "Training complete."
    fi
}

################################################################################
# Cleanup Models Function
################################################################################
cleanup_models() {
    local exp_dir=$1

    echo "Cleaning up model files from ${exp_dir}..."
    rm -f "${exp_dir}"/*.pth
    echo "Model cleanup complete. Results and predictions preserved."
}

################################################################################
# Print Results Summary Function
################################################################################
print_results_summary() {
    local dataset=$1
    local base_config=$2
    local results_prefix=$3

    echo ""
    echo "=========================================="
    echo "  EXPERIMENTS COMPLETE"
    echo "=========================================="

    local steps=$(extract_config_params "${base_config}" "steps")
    local blocks=$(extract_config_params "${base_config}" "blocks")
    local results_dir="${results_prefix}/${dataset}/${steps}_${blocks}"

    echo ""
    echo "Results directory: ${results_dir}"
    echo ""
    echo "To see mIoU results:"
    echo "  grep 'Overall mIoU' ${results_dir}/*.txt"
    echo ""
}
