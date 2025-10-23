#!/usr/bin/env python3
"""
Split dataset into k-folds for cross-validation experiments.

This script combines all images from train/ and test/ subdirectories,
then creates k random folds for cross-validation. Each fold can be used
as a test set with the remaining folds serving as the training set.

Usage:
    python split_dataset_into_folds.py --dataset bedroom_28 --n_folds 5 --seed 42
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from typing import List


def get_image_files(directory: Path) -> List[Path]:
    """
    Get all image files from a directory.

    Args:
        directory: Path to search for images

    Returns:
        Sorted list of image file paths
    """
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif']
    images = []
    for ext in extensions:
        images.extend(directory.glob(ext))
    return sorted(images)


def combine_and_split_into_folds(dataset_dir: Path, n_folds: int, seed: int, dry_run: bool = False):
    """
    Combine train/ and test/ images, then create n random train/test splits.
    Each split maintains the original train/test ratio.

    Args:
        dataset_dir: Directory containing train/ and test/ subdirectories
        n_folds: Number of random splits to create
        seed: Random seed for reproducibility
        dry_run: If True, only print what would be done without copying files
    """
    # Collect all images from both train and test directories
    all_images = []
    original_train_count = 0
    original_test_count = 0

    train_dir = dataset_dir / 'train'
    if train_dir.exists():
        train_images = get_image_files(train_dir)
        original_train_count = len(train_images)
        all_images.extend(train_images)
        print(f"  Found {len(train_images)} images in train/")
    else:
        print(f"  WARNING: train/ directory not found")

    test_dir = dataset_dir / 'test'
    if test_dir.exists():
        test_images = get_image_files(test_dir)
        original_test_count = len(test_images)
        all_images.extend(test_images)
        print(f"  Found {len(test_images)} images in test/")
    else:
        print(f"  WARNING: test/ directory not found")

    if not all_images:
        print("ERROR: No images found in train/ or test/ directories")
        return

    total_images = len(all_images)
    print(f"\n  Total images collected: {total_images}")
    print(f"  Original train/test split: {original_train_count}/{original_test_count}")
    print(f"  Each fold will use the same {original_train_count}/{original_test_count} split")

    # Create n different random splits
    print(f"\n  Creating {n_folds} random train/test splits...")

    for fold_idx in range(n_folds):
        # Shuffle with different seed for each fold
        random.seed(seed + fold_idx)
        shuffled_images = all_images.copy()
        random.shuffle(shuffled_images)

        # Split into train and test maintaining original ratio
        train_images = shuffled_images[:original_train_count]
        test_images = shuffled_images[original_train_count:]

        # Create fold directories
        fold_train_dir = dataset_dir / f"fold_{fold_idx + 1}" / "train"
        fold_test_dir = dataset_dir / f"fold_{fold_idx + 1}" / "test"

        if dry_run:
            print(f"\n  Would create fold_{fold_idx + 1}:")
            print(f"    - train/: {len(train_images)} images")
            print(f"    - test/: {len(test_images)} images")
        else:
            fold_train_dir.mkdir(parents=True, exist_ok=True)
            fold_test_dir.mkdir(parents=True, exist_ok=True)

            train_annotations = 0
            test_annotations = 0
            train_missing = []
            test_missing = []

            # Copy training images
            for img_path in train_images:
                label_path = img_path.with_suffix('.npy')
                shutil.copy2(img_path, fold_train_dir / img_path.name)
                if label_path.exists():
                    shutil.copy2(label_path, fold_train_dir / label_path.name)
                    train_annotations += 1
                else:
                    train_missing.append(img_path.name)

            # Copy test images
            for img_path in test_images:
                label_path = img_path.with_suffix('.npy')
                shutil.copy2(img_path, fold_test_dir / img_path.name)
                if label_path.exists():
                    shutil.copy2(label_path, fold_test_dir / label_path.name)
                    test_annotations += 1
                else:
                    test_missing.append(img_path.name)

            print(f"\n  Created fold_{fold_idx + 1}:")
            print(f"    - train/: {len(train_images)} images, {train_annotations} annotations")
            print(f"    - test/: {len(test_images)} images, {test_annotations} annotations")

            if train_missing:
                print(f"    - WARNING: {len(train_missing)} missing train annotations")
            if test_missing:
                print(f"    - WARNING: {len(test_missing)} missing test annotations")


def main():
    parser = argparse.ArgumentParser(
        description='Combine train/test data and split into k-folds for cross-validation'
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
        '--dry_run',
        action='store_true',
        help='Print what would be done without actually copying files'
    )

    args = parser.parse_args()

    # Construct dataset path
    dataset_dir = Path(args.base_dir) / args.dataset / "real"

    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        return 1

    print(f"="*60)
    print(f"K-Fold Cross-Validation Dataset Split")
    print(f"="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Number of folds: {args.n_folds}")
    print(f"Random seed: {args.seed}")
    print(f"Dataset directory: {dataset_dir}")

    if args.dry_run:
        print("\n*** DRY RUN MODE - No files will be copied ***")

    print(f"\n{'='*60}")
    print("Combining train and test data...")
    print(f"{'='*60}")

    combine_and_split_into_folds(dataset_dir, args.n_folds, args.seed, args.dry_run)

    print(f"\n{'='*60}")
    print(f"{'DRY RUN ' if args.dry_run else ''}Complete!")
    print(f"{'='*60}")

    if not args.dry_run:
        print(f"\nFolds created in: {dataset_dir}")
        print(f"Use these folds for {args.n_folds}-fold cross-validation:")
        for i in range(args.n_folds):
            print(f"  - fold_{i+1}/ (test set for experiment {i+1})")

    return 0


if __name__ == '__main__':
    exit(main())


