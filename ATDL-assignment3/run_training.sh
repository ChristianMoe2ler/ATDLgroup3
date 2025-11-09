#!/bin/bash

# Training command
TRAIN_CMD="torchrun --nproc_per_node=3 training_models/main.py \
        --model resnet34 \
        --mode baseline \
        --learning_rate 9e-4 \
        --epoch 200 \
        --dataset cifar100 \
        --batch_size 64 \
        --start_revision 199 \
        --threshold 0.3 \
        --task longtail \
        --long_tail \
        --save_path results/efficientnet_b0_dbpd_cifar100_long_tail"

# Run training 3 times
for i in {1..3}
do
    echo "=========================================="
    echo "Starting training run $i of 3"
    echo "=========================================="

    # Execute training command
    eval $TRAIN_CMD

    echo "Training run $i completed. Running final evaluation..."

    # Run final evaluation
    python final_eval.py

    echo "Evaluation for run $i completed."
    echo ""
done

echo "All 3 training runs and evaluations completed!"
