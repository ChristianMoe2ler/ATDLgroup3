#!/usr/bin/env python3
"""
Optimized dataset splitting script with reduced code duplication.

This script combines functionality from multiple splitting scripts into one
configurable tool that can handle different splitting strategies.
"""

import argparse
from pathlib import Path
from split_dataset_lib import (
    collect_images_from_dirs,
    create_fold_directories,
    copy_images_batch,
    print_fold_info,
    split_maintain_ratio,
    split_custom_train_size
)


def split_dataset(dataset_dir: Path, n_folds: int, seed: int,
                 train_size: int = None, dry_run: bool = False):
    """
    Split dataset into k-folds with flexible configuration.

    Args:
        dataset_dir: Directory containing train/ and test/ subdirectories
        n_folds: Number of folds to create
        seed: Random seed for reproducibility
        train_size: If specified, use this many training images per fold.
                   If None, maintain original train/test ratio.
        dry_run: If True, only print what would be done without copying files
    """
    # Collect all images from both train and test directories
    train_dir = dataset_dir / 'train'
    test_dir = dataset_dir / 'test'

    all_images, original_train_count, original_test_count = collect_images_from_dirs(
        train_dir, test_dir
    )

    if not all_images:
        print("ERROR: No images found in train/ or test/ directories")
        return

    total_images = len(all_images)
    print(f"\n  Total images collected: {total_images}")
    print(f"  Original train/test split: {original_train_count}/{original_test_count}")

    # Determine splitting strategy
    if train_size is None:
        # Maintain original ratio
        print(f"  Each fold will use the same {original_train_count}/{original_test_count} split")
        folds = split_maintain_ratio(all_images, original_train_count, seed, n_folds)
    else:
        # Use custom train size
        if train_size + original_test_count > total_images:
            print(f"\nERROR: Cannot create folds with {train_size} train + {original_test_count} test images")
            print(f"       Total available: {total_images} images")
            print(f"       Required: {train_size + original_test_count} images")
            return

        print(f"  Each fold will have: {train_size} train / {original_test_count} test images")
        folds = split_custom_train_size(all_images, train_size, original_test_count, seed, n_folds)

    # Create folds
    print(f"\n  Creating {n_folds} random train/test splits...")

    for fold_idx, (train_images, test_images) in enumerate(folds):
        fold_train_dir, fold_test_dir = create_fold_directories(dataset_dir, fold_idx)

        if dry_run:
            print(f"\n  Would create fold_{fold_idx + 1}:")
            print(f"    - train/: {len(train_images)} images")
            print(f"    - test/: {len(test_images)} images")
        else:
            # Copy images
            train_count, train_ann, train_missing = copy_images_batch(
                train_images, fold_train_dir
            )
            test_count, test_ann, test_missing = copy_images_batch(
                test_images, fold_test_dir
            )

            print_fold_info(fold_idx, train_count, test_count,
                          train_ann, test_ann, train_missing, test_missing)


def main():
    parser = argparse.ArgumentParser(
        description='Optimized dataset splitting for k-fold cross-validation'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (e.g., bedroom_28)'
    )
    parser.add_argument(
        '--n_folds',
        type=int,
        default=5,
        help='Number of folds to create (default: 5)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        default='datasets',
        help='Base directory containing datasets (default: datasets)'
    )
    parser.add_argument(
        '--train_size',
        type=int,
        default=None,
        help='Number of training images per fold. If not specified, maintains original ratio.'
    )
    parser.add_argument(
        '--dataset_type',
        type=str,
        default='real',
        choices=['real', 'ddpm'],
        help='Dataset type: real (datasets/<name>/real) or ddpm (datasets/<name>/ddpm)'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Print what would be done without actually copying files'
    )

    args = parser.parse_args()

    # Construct dataset path based on type
    if args.dataset_type == 'real':
        dataset_dir = Path(args.base_dir) / args.dataset / "real"
    else:  # ddpm
        dataset_dir = Path(args.base_dir) / args.dataset / "ddpm"

    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        return 1

    print(f"="*60)
    print(f"Optimized K-Fold Cross-Validation Dataset Split")
    print(f"="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Number of folds: {args.n_folds}")
    if args.train_size:
        print(f"Training images per fold: {args.train_size}")
    else:
        print(f"Strategy: Maintain original train/test ratio")
    print(f"Random seed: {args.seed}")
    print(f"Dataset directory: {dataset_dir}")

    if args.dry_run:
        print("\n*** DRY RUN MODE - No files will be copied ***")

    print(f"\n{'='*60}")
    print("Processing dataset...")
    print(f"{'='*60}")

    split_dataset(dataset_dir, args.n_folds, args.seed, args.train_size, args.dry_run)

    print(f"\n{'='*60}")
    print(f"{'DRY RUN ' if args.dry_run else ''}Complete!")
    print(f"{'='*60}")

    if not args.dry_run:
        print(f"\nFolds created in: {dataset_dir}")
        print(f"Use these folds for {args.n_folds}-fold cross-validation:")
        for i in range(args.n_folds):
            print(f"  - fold_{i+1}/")

    return 0


if __name__ == '__main__':
    exit(main())
