#!/usr/bin/env python3
"""
Shared library for dataset splitting functionality.

This module contains common functions used across different dataset splitting scripts
to reduce code duplication and improve maintainability.
"""

import shutil
import random
from pathlib import Path
from typing import List, Tuple


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


def copy_image_with_label(img_path: Path, dest_dir: Path) -> Tuple[bool, str]:
    """
    Copy an image and its corresponding .npy label file to destination directory.

    Args:
        img_path: Path to the image file
        dest_dir: Destination directory

    Returns:
        Tuple of (has_annotation: bool, image_name: str)
    """
    label_path = img_path.with_suffix('.npy')
    shutil.copy2(img_path, dest_dir / img_path.name)

    has_annotation = label_path.exists()
    if has_annotation:
        shutil.copy2(label_path, dest_dir / label_path.name)

    return has_annotation, img_path.name


def copy_images_batch(images: List[Path], dest_dir: Path) -> Tuple[int, int, List[str]]:
    """
    Copy a batch of images with their labels to destination directory.

    Args:
        images: List of image paths to copy
        dest_dir: Destination directory

    Returns:
        Tuple of (num_images, num_annotations, missing_annotations_list)
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    annotations_count = 0
    missing = []

    for img_path in images:
        has_annotation, img_name = copy_image_with_label(img_path, dest_dir)
        if has_annotation:
            annotations_count += 1
        else:
            missing.append(img_name)

    return len(images), annotations_count, missing


def collect_images_from_dirs(train_dir: Path, test_dir: Path) -> Tuple[List[Path], int, int]:
    """
    Collect all images from train and test directories.

    Args:
        train_dir: Training directory path
        test_dir: Testing directory path

    Returns:
        Tuple of (all_images, train_count, test_count)
    """
    all_images = []
    train_count = 0
    test_count = 0

    if train_dir.exists():
        train_images = get_image_files(train_dir)
        train_count = len(train_images)
        all_images.extend(train_images)
        print(f"  Found {len(train_images)} images in train/")
    else:
        print(f"  WARNING: train/ directory not found")

    if test_dir.exists():
        test_images = get_image_files(test_dir)
        test_count = len(test_images)
        all_images.extend(test_images)
        print(f"  Found {len(test_images)} images in test/")
    else:
        print(f"  WARNING: test/ directory not found")

    return all_images, train_count, test_count


def create_fold_directories(dataset_dir: Path, fold_idx: int) -> Tuple[Path, Path]:
    """
    Create train and test directories for a specific fold.

    Args:
        dataset_dir: Base dataset directory
        fold_idx: Fold index (0-based)

    Returns:
        Tuple of (train_dir, test_dir) paths
    """
    fold_train_dir = dataset_dir / f"fold_{fold_idx + 1}" / "train"
    fold_test_dir = dataset_dir / f"fold_{fold_idx + 1}" / "test"
    return fold_train_dir, fold_test_dir


def print_fold_info(fold_idx: int, train_count: int, test_count: int,
                   train_ann: int, test_ann: int,
                   train_missing: List[str], test_missing: List[str]) -> None:
    """
    Print information about a created fold.

    Args:
        fold_idx: Fold index (0-based)
        train_count: Number of training images
        test_count: Number of test images
        train_ann: Number of training annotations
        test_ann: Number of test annotations
        train_missing: List of training images without annotations
        test_missing: List of test images without annotations
    """
    print(f"\n  Created fold_{fold_idx + 1}:")
    print(f"    - train/: {train_count} images, {train_ann} annotations")
    print(f"    - test/: {test_count} images, {test_ann} annotations")

    if train_missing:
        print(f"    - WARNING: {len(train_missing)} missing train annotations")
    if test_missing:
        print(f"    - WARNING: {len(test_missing)} missing test annotations")


def split_maintain_ratio(all_images: List[Path], original_train_count: int,
                        seed: int, n_folds: int) -> List[Tuple[List[Path], List[Path]]]:
    """
    Split images into folds maintaining original train/test ratio.

    Args:
        all_images: List of all images
        original_train_count: Original number of training images
        seed: Random seed base
        n_folds: Number of folds to create

    Returns:
        List of tuples (train_images, test_images) for each fold
    """
    folds = []
    for fold_idx in range(n_folds):
        random.seed(seed + fold_idx)
        shuffled = all_images.copy()
        random.shuffle(shuffled)

        train_imgs = shuffled[:original_train_count]
        test_imgs = shuffled[original_train_count:]
        folds.append((train_imgs, test_imgs))

    return folds


def split_custom_train_size(all_images: List[Path], train_size: int,
                            test_size: int, seed: int, n_folds: int) -> List[Tuple[List[Path], List[Path]]]:
    """
    Split images into folds with custom train/test sizes.

    Args:
        all_images: List of all images
        train_size: Number of training images per fold
        test_size: Number of test images per fold
        seed: Random seed base
        n_folds: Number of folds to create

    Returns:
        List of tuples (train_images, test_images) for each fold
    """
    folds = []
    for fold_idx in range(n_folds):
        random.seed(seed + fold_idx)
        shuffled = all_images.copy()
        random.shuffle(shuffled)

        train_imgs = shuffled[:train_size]
        test_imgs = shuffled[train_size:train_size + test_size]
        folds.append((train_imgs, test_imgs))

    return folds
