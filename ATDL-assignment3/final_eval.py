#!/usr/bin/env python
"""
Final Model Evaluation Script

Evaluates a trained model on the test set and reports:
- Accuracy
- Macro F1 Score
- Mean Precision
- Mean Recall

Supports:
- CIFAR-10 (10 classes)
- CIFAR-100 (100 classes)
- Long-tail versions of both datasets

Usage:
    1. Set configuration parameters below:
       - MODEL_PATH: Path to trained model checkpoint
       - MODEL_NAME: Model architecture ('efficientnet_b0', 'resnet18', 'mobilenet_v2')
       - DATASET: 'cifar10' or 'cifar100'
       - LONG_TAIL: True/False for long-tail versions
       - SHOW_PER_CLASS_METRICS: True/False for detailed per-class analysis
    2. Run: python final_eval.py

Examples:
    # Evaluate on CIFAR-10
    DATASET = 'cifar10'
    LONG_TAIL = False

    # Evaluate on long-tail CIFAR-100
    DATASET = 'cifar100'
    LONG_TAIL = True
"""

import sys
import os

# Add training_models directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.join(script_dir, 'training_models')
sys.path.insert(0, training_dir)


import torch
from model_zoo import ModelZoo
from data import load_cifar10, load_cifar100

# Configuration - CHANGE THESE to match your training
MODEL_PATH = 'trained_model.pth'
MODEL_NAME = 'resnet34'  # Options: 'efficientnet_b0', 'resnet18', 'mobilenet_v2'
DATASET = 'cifar10'  # Options: 'cifar10', 'cifar100'
PRETRAINED = False
LONG_TAIL = False
BATCH_SIZE = 128
SHOW_PER_CLASS_METRICS = False  # Set to True to see detailed per-class F1 scores

# Automatically set NUM_CLASSES based on DATASET
NUM_CLASSES = 10 if DATASET == 'cifar10' else 100

print("="*60)
print("Final Model Evaluation")
print("="*60)
print(f"Dataset: {DATASET.upper()}")
print(f"Number of classes: {NUM_CLASSES}")
print(f"Long-tail: {LONG_TAIL}")

# Setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model
print(f"Loading model from: {MODEL_PATH}")
loaded = torch.load(MODEL_PATH, map_location=device, weights_only=False)

# Check what was loaded
if isinstance(loaded, dict):
    # Loaded state_dict
    print("Detected state_dict format")
    mz = ModelZoo(num_classes=NUM_CLASSES, pretrained=PRETRAINED)

    if MODEL_NAME == 'efficientnet_b0':
        model = mz.efficientnet_b0()
    elif MODEL_NAME == 'resnet18':
        model = mz.resnet18()
    elif MODEL_NAME == 'mobilenet_v2':
        model = mz.mobilenet_v2()
    elif MODEL_NAME == 'resnet34':
        model = mz.resnet34()
    else:
        raise ValueError(f"Add your model {MODEL_NAME} to this script")

    # Handle DDP-saved models (with 'module.' prefix)
    if list(loaded.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in loaded.items():
            name = k[7:]  # remove 'module.' prefix
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(loaded)
else:
    # Loaded full model object
    print("Detected full model format")
    model = loaded

    # Unwrap DDP if needed
    if hasattr(model, 'module'):
        model = model.module

model.to(device)
model.eval()

# Load both train and test datasets
print("Loading datasets...")
if DATASET == 'cifar10':
    train_loader, test_loader, _, _ = load_cifar10(long_tail=LONG_TAIL, batch_size=BATCH_SIZE)
elif DATASET == 'cifar100':
    train_loader, test_loader, _, _ = load_cifar100(long_tail=LONG_TAIL, batch_size=BATCH_SIZE)
else:
    raise ValueError(f"Unsupported dataset: {DATASET}. Options: 'cifar10', 'cifar100'")

# Build class distribution from training dataset
print("Computing class distribution from training dataset...")
train_dataset = train_loader.dataset
class_counts = {}
for i in range(len(train_dataset)):
    _, label = train_dataset[i]
    label_item = label.item() if torch.is_tensor(label) else label
    class_counts[label_item] = class_counts.get(label_item, 0) + 1

cls_num_list = [class_counts.get(i, 0) for i in range(NUM_CLASSES)]
print(f"Class distribution computed: {len([c for c in cls_num_list if c > 0])} classes with samples")

# Evaluate
print("Evaluating on full test set...")
correct = 0
total = 0

# For F1 macro calculation
true_positives = torch.zeros(NUM_CLASSES, device=device)
false_positives = torch.zeros(NUM_CLASSES, device=device)
false_negatives = torch.zeros(NUM_CLASSES, device=device)

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Update confusion matrix for F1 calculation
        correct_mask = (preds == labels)
        true_positives += torch.bincount(labels[correct_mask], minlength=NUM_CLASSES)
        incorrect_mask = ~correct_mask
        false_positives += torch.bincount(preds[incorrect_mask], minlength=NUM_CLASSES)
        false_negatives += torch.bincount(labels[incorrect_mask], minlength=NUM_CLASSES)

accuracy = correct / total

# Calculate F1 macro
precision = true_positives / (true_positives + false_positives + 1e-10)
recall = true_positives / (true_positives + false_negatives + 1e-10)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
f1_scores = torch.nan_to_num(f1_scores, nan=0.0)
f1_macro = f1_scores.mean().item()

print("="*60)
print(f"Final Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Final Test F1 Macro: {f1_macro:.4f}")
print(f"Correct Predictions: {correct}/{total}")
print("="*60)
print(f"\nAggregate metrics:")
print(f"  Precision (mean): {precision.mean().item():.4f}")
print(f"  Recall (mean): {recall.mean().item():.4f}")
print(f"  F1 Score (macro): {f1_macro:.4f}")
print("="*60)

# Per-class F1 analysis based on training class distribution
print("\n" + "="*60)
print("PER-CLASS F1 SCORE ANALYSIS")
print("="*60)

f1_numpy = f1_scores.cpu().numpy()

# Find most and least popular classes based on training distribution
if max(cls_num_list) > 0:
    most_popular_classes = [i for i, count in enumerate(cls_num_list) if count == max(cls_num_list)]
    least_popular_classes = [i for i, count in enumerate(cls_num_list) if count == min(cls_num_list) and count > 0]

    print(f"\nMost Popular Class(es) in Training (Training samples: {max(cls_num_list):,}):")
    for cls_id in most_popular_classes:
        print(f"  Class {cls_id:3d}: F1 = {f1_numpy[cls_id]:.4f}, "
              f"Precision = {precision[cls_id].item():.4f}, "
              f"Recall = {recall[cls_id].item():.4f}")

    if len(least_popular_classes) > 0 and min(cls_num_list) > 0:
        print(f"\nLeast Popular Class(es) in Training (Training samples: {min([c for c in cls_num_list if c > 0]):,}):")
        for cls_id in least_popular_classes:
            print(f"  Class {cls_id:3d}: F1 = {f1_numpy[cls_id]:.4f}, "
                  f"Precision = {precision[cls_id].item():.4f}, "
                  f"Recall = {recall[cls_id].item():.4f}")

        # Summary statistics
        avg_f1_popular = sum(f1_numpy[i] for i in most_popular_classes) / len(most_popular_classes)
        avg_f1_least = sum(f1_numpy[i] for i in least_popular_classes) / len(least_popular_classes)
        print(f"\nSummary:")
        print(f"  Average F1 for most popular class(es): {avg_f1_popular:.4f}")
        print(f"  Average F1 for least popular class(es): {avg_f1_least:.4f}")
        print(f"  F1 gap (popular - least): {avg_f1_popular - avg_f1_least:.4f}")

        # Calculate imbalance ratio
        imbalance_ratio = max(cls_num_list) / min([c for c in cls_num_list if c > 0])
        print(f"  Training imbalance ratio: {imbalance_ratio:.2f}:1")
else:
    print("\nWarning: No class distribution information available")

print("="*60)

# Optional: Show per-class metrics (useful for long-tail analysis)
if SHOW_PER_CLASS_METRICS:
    print("\nPer-class F1 Scores:")
    print("-" * 60)

    # Sort classes by F1 score for easier analysis
    f1_numpy = f1_scores.cpu().numpy()
    sorted_indices = f1_numpy.argsort()

    # Show worst 10 classes
    print("\nWorst 10 classes:")
    for idx in sorted_indices[:10]:
        print(f"  Class {idx:3d}: F1={f1_numpy[idx]:.4f}, "
              f"Precision={precision[idx].item():.4f}, "
              f"Recall={recall[idx].item():.4f}")

    # Show best 10 classes
    print("\nBest 10 classes:")
    for idx in sorted_indices[-10:][::-1]:
        print(f"  Class {idx:3d}: F1={f1_numpy[idx]:.4f}, "
              f"Precision={precision[idx].item():.4f}, "
              f"Recall={recall[idx].item():.4f}")

    print("-" * 60)
    print(f"F1 Score - Min: {f1_numpy.min():.4f}, Max: {f1_numpy.max():.4f}, Std: {f1_numpy.std():.4f}")
    print("="*60)

