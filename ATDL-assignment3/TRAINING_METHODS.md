# Training Methods Overview

This document describes the different training strategies implemented in this Progressive Data Dropout codebase.

## Table of Contents
- [Baseline Training](#baseline-training)
- [Difficulty-Based Progressive Dropout (DBPD)](#difficulty-based-progressive-dropout-dbpd)
- [Scheduled Match Random Dropout (SMRD)](#scheduled-match-random-dropout-smrd)
- [Scalar Random Dropout (SRD)](#scalar-random-dropout-srd)
- [Adaptive Dropout](#adaptive-dropout)
- [Alternative Dropout](#alternative-dropout)
- [Periodic Full Random Dropout](#periodic-full-random-dropout)
- [Parameter Comparison](#parameter-comparison)

---

## Baseline Training

**Mode:** `--mode baseline`

**Description:** Standard deep learning training without any sample dropout.

**How it works:**
- Trains on the full dataset every epoch
- No progressive dropout applied
- Serves as the reference for comparison

**Key Parameters:**
- `--epoch`: Number of training epochs
- `--batch_size`: Batch size (default: 128)
- `--learning_rate`: Learning rate (default: 3e-4)

**Example Command:**
```bash
python training_models/main.py \
  --model efficientnet_b0 \
  --mode baseline \
  --epoch 100 \
  --dataset cifar100 \
  --batch_size 64 \
  --learning_rate 3e-4 \
  --task classification \
  --save_path results/baseline
```

**Use Case:** Establish baseline performance for comparison with progressive dropout methods.

---

## Difficulty-Based Progressive Dropout (DBPD)

**Mode:** `--mode train_with_revision`

**Description:** The core progressive dropout method that selectively trains on "difficult" samples based on model confidence.

**How it works:**
1. **Phase 1 (epochs 0 to start_revision-1):** Train on difficult samples only
   - Compute model predictions for each batch
   - Calculate confidence (softmax probability for correct class)
   - Select samples where confidence < threshold
   - Only backpropagate on these difficult samples

2. **Phase 2 (epochs start_revision to end):** Train on all samples
   - Resume normal training on full dataset
   - Helps model refine what it learned from difficult samples

**Key Parameters:**
- `--start_revision`: Epoch to switch from DBPD to full training (e.g., 199)
- `--threshold`: Confidence threshold for sample selection (e.g., 0.3)
  - Lower threshold = fewer, more difficult samples
  - Higher threshold = more samples included
- `--epoch`: Total training epochs
- `--learning_rate`: Learning rate

**Example Command:**
```bash
python training_models/main.py \
  --model efficientnet_b0 \
  --pretrained \
  --mode train_with_revision \
  --epoch 200 \
  --dataset cifar100 \
  --batch_size 64 \
  --start_revision 199 \
  --threshold 0.3 \
  --learning_rate 3e-4 \
  --task classification \
  --save_path results/dbpd
```

**Advantages:**
- Focuses computation on samples that matter most
- Faster training (fewer samples processed)
- Often achieves similar or better accuracy than baseline

**Effective Epochs:**
- Reports computational cost as "effective epochs"
- Effective epochs = (samples_processed) / (dataset_size)
- Example: If you process 30% of data, 100 epochs = 30 effective epochs

**Use Case:** When you want intelligent sample selection to speed up training while maintaining accuracy.

---

## Scheduled Match Random Dropout (SMRD)

**Mode:** `--mode train_with_random`

**Description:** Randomly drops samples to match the computational budget of DBPD, serving as a controlled baseline.

**How it works:**
1. **Phase 1 (epochs 0 to start_revision-1):** Random sample dropout
   - Randomly select same percentage of samples as DBPD would
   - No intelligence - purely random selection
   - Matches DBPD's computational cost

2. **Phase 2 (epochs start_revision to end):** Train on all samples
   - Same as DBPD phase 2

**Key Parameters:**
- Same as DBPD: `--start_revision`, `--threshold`, `--epoch`, `--learning_rate`
- `--threshold` determines dropout percentage (indirectly, by matching DBPD behavior)

**Example Command:**
```bash
python training_models/main.py \
  --model efficientnet_b0 \
  --pretrained \
  --mode train_with_random \
  --epoch 200 \
  --dataset cifar100 \
  --batch_size 64 \
  --start_revision 199 \
  --threshold 0.3 \
  --learning_rate 3e-4 \
  --task classification \
  --save_path results/smrd
```

**Purpose:**
- Control experiment to validate DBPD's intelligent selection
- If DBPD outperforms SMRD with same computational cost, proves difficulty-based selection works

**Use Case:** Research baseline to demonstrate that difficulty-based selection is better than random selection.

---

## Scalar Random Dropout (SRD)

**Mode:** `--mode train_with_percentage`

**Description:** Drops a fixed percentage of samples randomly throughout training.

**How it works:**
- Each epoch, randomly drop a percentage of the dataset
- Percentage is fixed and doesn't change during training
- Simpler than SMRD (no phase switching)

**Key Parameters:**
- `--threshold`: Determines dropout percentage
- `--epoch`: Total training epochs
- `--learning_rate`: Learning rate

**Example Command:**
```bash
python training_models/main.py \
  --model efficientnet_b0 \
  --mode train_with_percentage \
  --epoch 200 \
  --dataset cifar100 \
  --batch_size 64 \
  --threshold 0.3 \
  --learning_rate 3e-4 \
  --task classification \
  --save_path results/srd
```

**Use Case:** Simple random dropout baseline for ablation studies.

---

## Adaptive Dropout

**Mode:** `--mode train_with_adaptive`

**Description:** Dynamically adjusts dropout based on validation performance at regular intervals.

**How it works:**
- Evaluates model performance every `--interval` steps
- Adjusts dropout rate based on performance
- Uses `--increment` to control adjustment magnitude

**Key Parameters:**
- `--interval`: Steps between evaluations (default: 50)
- `--increment`: Adjustment increment (default: 0.1)
- `--start_revision`: When to stop adaptive behavior
- `--threshold`: Initial dropout threshold

**Example Command:**
```bash
python training_models/main.py \
  --model efficientnet_b0 \
  --mode train_with_adaptive \
  --epoch 200 \
  --dataset cifar100 \
  --batch_size 64 \
  --interval 50 \
  --increment 0.1 \
  --start_revision 199 \
  --threshold 0.3 \
  --learning_rate 3e-4 \
  --task classification \
  --save_path results/adaptive
```

**Use Case:** Experimental method for automatic dropout adjustment based on training dynamics.

---

## Alternative Dropout

**Mode:** `--mode train_with_alternative`

**Description:** Alternates between different dropout strategies on an epoch-by-epoch basis.

**How it works:**
- Switches between different training strategies each epoch
- Implementation details vary based on codebase version

**Key Parameters:**
- `--start_revision`: When to change behavior
- `--threshold`: Sample selection threshold
- `--epoch`: Total epochs

**Example Command:**
```bash
python training_models/main.py \
  --model efficientnet_b0 \
  --mode train_with_alternative \
  --epoch 200 \
  --dataset cifar100 \
  --batch_size 64 \
  --start_revision 199 \
  --threshold 0.3 \
  --learning_rate 3e-4 \
  --task classification \
  --save_path results/alternative
```

**Use Case:** Experimental method for exploring different dropout patterns.

---

## Periodic Full Random Dropout

**Mode:** `--mode train_with_periodic_full_random`

**Description:** SMRD variant that periodically trains on the full dataset to prevent catastrophic forgetting.

**How it works:**
- Mostly uses random dropout like SMRD
- Every N "effective epochs", trains on full dataset
- Helps maintain knowledge of easy samples

**Key Parameters:**
- `--refresh_interval`: Effective epochs between full dataset training (default: 5.0)
- `--start_revision`: When to stop periodic refresh
- `--threshold`: Dropout threshold
- `--warmup_epochs`: Initial epochs with different behavior (default: 5)

**Example Command:**
```bash
python training_models/main.py \
  --model efficientnet_b0 \
  --mode train_with_periodic_full_random \
  --epoch 200 \
  --dataset cifar100 \
  --batch_size 64 \
  --refresh_interval 5.0 \
  --warmup_epochs 5 \
  --start_revision 199 \
  --threshold 0.3 \
  --learning_rate 3e-4 \
  --task classification \
  --save_path results/periodic
```

**Use Case:** Variant of SMRD that prevents forgetting of easy samples through periodic full-dataset training.

---

## Parameter Comparison

| Method | Key Parameters | Computational Cost | Sample Selection |
|--------|---------------|-------------------|------------------|
| **Baseline** | epochs | 100% | All samples |
| **DBPD** | start_revision, threshold | 30-70% (varies) | Difficulty-based |
| **SMRD** | start_revision, threshold | Matches DBPD | Random |
| **SRD** | threshold | 30-70% (varies) | Random |
| **Adaptive** | interval, increment | Varies | Adaptive difficulty |
| **Alternative** | start_revision | Varies | Alternating |
| **Periodic** | refresh_interval | Varies | Random + periodic full |

---

## Common Parameters Across All Methods

### Required Parameters
- `--model`: Model architecture (e.g., efficientnet_b0, resnet18, mobilenet_v2)
- `--mode`: Training method (see modes above)
- `--dataset`: Dataset to use (cifar10, cifar100, imagenet, etc.)
- `--task`: Task type (classification, segmentation, longtail)

### Optional But Important
- `--pretrained`: Use ImageNet pretrained weights (recommended for transfer learning)
- `--learning_rate`: Learning rate (default: 3e-4)
  - **Multi-GPU:** Scale by number of GPUs (e.g., 9e-4 for 3 GPUs)
- `--batch_size`: Per-GPU batch size (default: 128)
- `--save_path`: Directory to save results and plots
- `--epoch`: Total training epochs

### Special Cases
- `--long_tail`: Enable long-tail (imbalanced) dataset
- `--ldam`: Use LDAM-DRW loss for long-tail classification
- `--noisy`: Use noisy label datasets
- `--mae_checkpoint`: Path to MAE pretrained weights (for mae_vit_b_16)

---

## Multi-GPU Training

All methods support multi-GPU training via PyTorch DDP:

```bash
# Single GPU
python training_models/main.py [args]

# Multi-GPU (e.g., 3 GPUs)
torchrun --nproc_per_node=3 training_models/main.py [args]
```

**Important for Multi-GPU:**
1. **Effective batch size** = batch_size × num_gpus
2. **Learning rate scaling**: Use `--learning_rate` scaled by num_gpus
   - Single GPU (batch=32): `--learning_rate 3e-4`
   - 3 GPUs (batch=32 each, effective=96): `--learning_rate 9e-4` (recommended)
3. **SyncBatchNorm**: Automatically enabled for multi-GPU (already implemented)
4. **Effective epochs**: Reported value should be multiplied by num_gpus

---

## Typical Experimental Setup

### Research Comparison (DBPD vs Baselines)

**1. Baseline:**
```bash
python training_models/main.py \
  --model efficientnet_b0 --pretrained \
  --mode baseline --epoch 200 \
  --dataset cifar100 --batch_size 64 \
  --learning_rate 3e-4 --task classification \
  --save_path results/baseline
```

**2. DBPD (Proposed Method):**
```bash
python training_models/main.py \
  --model efficientnet_b0 --pretrained \
  --mode train_with_revision --epoch 200 \
  --dataset cifar100 --batch_size 64 \
  --start_revision 199 --threshold 0.3 \
  --learning_rate 3e-4 --task classification \
  --save_path results/dbpd
```

**3. SMRD (Control - Random Selection):**
```bash
python training_models/main.py \
  --model efficientnet_b0 --pretrained \
  --mode train_with_random --epoch 200 \
  --dataset cifar100 --batch_size 64 \
  --start_revision 199 --threshold 0.3 \
  --learning_rate 3e-4 --task classification \
  --save_path results/smrd
```

**Compare:**
- Test accuracy: DBPD should match/exceed baseline
- Training time: DBPD should be faster than baseline
- Effective epochs: DBPD uses fewer samples than baseline
- DBPD vs SMRD: Both use similar compute, but DBPD should be more accurate

---

## Tips and Best Practices

### Choosing `--threshold`:
- **0.1-0.2**: Very aggressive, only hardest samples (~10-20% of data)
- **0.3**: Moderate, balanced approach (~30-40% of data) ← **Recommended**
- **0.5**: Conservative, includes more samples (~50-60% of data)

### Choosing `--start_revision`:
- Typically set to `total_epochs - 1` or `total_epochs - 5`
- Examples:
  - `--epoch 200 --start_revision 199`: DBPD for 199 epochs, full training for 1
  - `--epoch 200 --start_revision 195`: DBPD for 195 epochs, full training for 5

### Fine-tuning vs Training from Scratch:
- **With `--pretrained`**: Start from ImageNet weights
  - Achieves higher accuracy (e.g., 83-86% on CIFAR-100)
  - Faster convergence
  - Recommended for most cases

- **Without `--pretrained`**: Random initialization
  - Lower final accuracy (e.g., 68-70% on CIFAR-100)
  - Slower convergence
  - Useful for ablation studies

---

## Output and Metrics

All methods produce:
1. **Console output**: Per-epoch training/test accuracy and loss
2. **Plots**: Accuracy and loss curves over time (saved to `--save_path`)
3. **Effective epochs**: Computational cost metric
4. **Trained model**: `trained_model.pth` in current directory

### Understanding Effective Epochs:
```
Effective Epochs = Total Samples Processed / Dataset Size

Example:
- Dataset: 50,000 samples (CIFAR-100)
- Trained for 200 epochs
- DBPD processed 15,000,000 samples total
- Effective Epochs = 15,000,000 / 50,000 = 300 epochs

This means DBPD did computational work equivalent to
300 baseline epochs, but only took 200 wall-clock epochs.
```

---

## Troubleshooting

### Low Accuracy with Multi-GPU:
- Check if `--learning_rate` is scaled properly
- Verify SyncBatchNorm is enabled (look for message in console)
- Ensure effective batch size isn't too large

### Training Too Slow:
- Reduce `--batch_size` if GPU memory is underutilized
- Check `num_workers` in data.py (should be 4)
- Verify GPU utilization with `nvidia-smi`

### DBPD Not Saving Time:
- Threshold might be too high (try 0.3 or lower)
- Check effective epochs calculation
- Ensure you're actually in DBPD phase (epoch < start_revision)

---

## References

For implementation details, see:
- `training_models/selective_gradient.py`: All progressive dropout methods
- `training_models/baseline.py`: Baseline training
- `training_models/main.py`: Entry point and argument parsing
- `CLAUDE.md`: Project overview and common commands
