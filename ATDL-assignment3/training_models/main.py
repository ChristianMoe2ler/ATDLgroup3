import argparse
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from model import resnet18, efficientnet_b0
from model_zoo import ModelZoo
from data import load_cifar100, load_mnist, load_imagenet, load_cityscapes, load_cifar10, load_medmnist3D, load_noisy
from data import load_cifar100, load_mnist, load_imagenet, load_cityscapes, load_cifar10, load_medmnist3D, load_cub2011, load_aircraft, load_flowers
from baseline import train_baseline, train_baseline_noisy
from selective_gradient import TrainRevision
from test import test_model
from longtail_train import train_baseline_longtail, train_with_revision_longtail

def setup_ddp():
    """Initialize DDP for single machine multi-GPU training"""
    dist.init_process_group(backend='nccl')

def cleanup_ddp():
    """Clean up DDP process group"""
    if dist.is_initialized():
        dist.destroy_process_group()

def get_rank():
    """Get current process rank (0 for main process)"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0

def get_world_size():
    """Get total number of processes"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1

def is_main_process():
    """Check if this is the main process (rank 0)"""
    return get_rank() == 0

def is_distributed():
    """Check if running in distributed mode"""
    return dist.is_available() and dist.is_initialized()

def make_ddp_dataloader(loader, shuffle=True):
    """
    Recreate a DataLoader with DistributedSampler for DDP training.

    Args:
        loader: Original DataLoader
        shuffle: Whether to shuffle data

    Returns:
        New DataLoader with DistributedSampler, or original loader if not in DDP mode
    """
    if not is_distributed():
        return loader

    # Get dataset from original loader
    dataset = loader.dataset

    # Create DistributedSampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle
    )

    # Create new loader with DistributedSampler
    new_loader = DataLoader(
        dataset,
        batch_size=loader.batch_size,
        sampler=sampler,
        num_workers=loader.num_workers if hasattr(loader, 'num_workers') else 4,
        pin_memory=loader.pin_memory if hasattr(loader, 'pin_memory') else True,
        drop_last=loader.drop_last if hasattr(loader, 'drop_last') else False
    )

    return new_loader

def main():
    parser = argparse.ArgumentParser(description="Train ResNet on CIFAR-100")
    parser.add_argument("--mode", type=str, choices=["baseline", "selective_gradient", "selective_epoch", "train_with_revision", "train_with_samples", "train_with_revision_3d", "train_with_random", "train_with_warmup_random", "train_with_periodic_full_random", "train_with_inv_lin", "train_with_log", "train_with_percentage",
                                                     "train_with_adaptive", "train_with_alternative", "train_with_stratified", "train_with_adaptive_dropout"], required=True,
                        help="Choose training mode: 'baseline' or 'selective_gradient'")
    parser.add_argument("--epoch", type=int, required=False, default=10,
                        help="Number of epochs to train for")
    parser.add_argument("--task", type=str, required=True, default="classification",
                        help="segmentation or classification or longtail")
    parser.add_argument("--model", type=str, choices=["resnet18", "resnet_3d", "resnet34", "resnet50", "resnet101", "efficientnet_b0","efficientnet_b7", "efficientnet_b4", "mobilenet_v2", "mobilenet_v3", "vit_b_16", "mae_vit_b_16", "efficientformer", "segformer_b2"], required=True,
                        help="Choose the model: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'mobilenet_v2', 'mobilenet_v3', 'efficientnet_b0', 'vit_b_16', 'mae_vit_b_16'")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained versions (applies to torchvision models, not MAE)")
    parser.add_argument("--mae_checkpoint", type=str, default=None, help="Path to MAE pretrained checkpoint file (used with --model mae_vit_b_16)")
    parser.add_argument("--save_path", type=str, help="to save graphs")
    parser.add_argument("--threshold", type=float, help="threshold to remove samples")
    parser.add_argument("--epoch_threshold", type=int, help="threshold to reintroduce correct samples in epoch")
    parser.add_argument("--dataset", type=str, help="CIFAR or MNIST")
    parser.add_argument("--batch_size", type=int, help="32,64,128 etc.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for optimizer (default: 3e-4). Scale by number of GPUs for multi-GPU training.")
    parser.add_argument("--start_revision", type=int, help="Start revision after the given epoch")
    parser.add_argument("--long_tail", action="store_true", help="LongTail CIFAR100 or native version")
    parser.add_argument("--ldam", action="store_true", help="Use LDAM-DRW method for long tail classification")
    parser.add_argument("--noisy", action="store_true", help="Use noisy dataset")
    parser.add_argument("--interval", type=int, default=50)
    parser.add_argument("--increment", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs for train_with_warmup_random mode")
    parser.add_argument("--refresh_interval", type=float, default=5.0, help="Number of effective epochs between full dataset training for train_with_periodic_full_random mode")
    parser.add_argument("--download", action="store_true", help="Download dataset if not exists (for aircraft and cub2011 datasets)")
    parser.add_argument("--scheduler_type", type=str, choices=["step", "cosine"], default="step", help="Learning rate scheduler type: 'step' (default) or 'cosine'")
    parser.add_argument("--ema_alpha", type=float, default=0.1, help="EMA smoothing factor for train_with_adaptive_dropout (default: 0.1). Higher = faster adaptation, lower = smoother")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Small constant for train_with_adaptive_dropout to prevent zero proportions (default: 0.01)")
    args = parser.parse_args()

    # Initialize DDP if running with torchrun
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        setup_ddp()
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device(f'cuda:{local_rank}')
        if is_main_process():
            print(f"Running in DDP mode with {get_world_size()} GPUs")
            print(f"Process {get_rank()}: Using GPU {local_rank}")
    else:
        # Check for CUDA (NVIDIA GPU), then MPS (Apple Silicon), then fall back to CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        if is_main_process():
            print(f"Running in single-device mode on: {device}")

    pretrained = False
    if args.dataset == "mnist":
        num_classes = 10
        train_loader, test_loader = load_mnist()
    elif args.dataset == "cifar" or args.dataset == "cifar100":
        if args.batch_size:
            train_loader, test_loader, cls_num_list, data_size = load_cifar100(args.long_tail, args.batch_size)
        else:
            train_loader, test_loader, cls_num_list, data_size = load_cifar100(args.long_tail)

        num_classes = 100
    elif args.dataset == "cifar10":
        if args.noisy:
            train_loader, test_loader, data_size = load_noisy(args.batch_size)
        else:
            train_loader, test_loader, cls_num_list, data_size = load_cifar10(args.long_tail, args.batch_size)
        num_classes = 10
    elif args.dataset == "imagenet":
        num_classes = 1000
        train_loader, test_loader, data_size = load_imagenet(args.batch_size)
    elif args.dataset == "cityscapes":
        num_classes = 19
        train_loader, test_loader = load_cityscapes()
    elif args.dataset == "organ_medmnist3d":
        num_classes = 11
        train_loader, test_loader, data_size = load_medmnist3D(args.batch_size)
    
    if not args.long_tail:
        cls_num_list = None
    elif args.dataset == "aircraft":
        num_classes = 100  # FGVC-Aircraft variant 有 100 个类别
        if args.batch_size:
            train_loader, test_loader, data_size = load_aircraft(args.batch_size, root='/root/autodl-tmp/project/training_models/dataset', download=args.download)
        else:
            train_loader, test_loader, data_size = load_aircraft(download=args.download)
    elif args.dataset == "cub2011":
        num_classes = 200  # CUB-200-2011 有 200 个类别
        if args.batch_size:
            train_loader, test_loader, data_size = load_cub2011(args.batch_size, root='/root/autodl-tmp/project/training_models/dataset', download=args.download)
        else:
            train_loader, test_loader, data_size = load_cub2011(root='/root/autodl-tmp/project/training_models/dataset', download=args.download)
    elif args.dataset == "flowers":
        num_classes = 102  # Oxford 102 Category Flower 有 102 个类别
        if args.batch_size:
            train_loader, val_loader, test_loader, data_size = load_flowers(args.batch_size, root='/root/autodl-tmp/project/training_models/dataset', download=args.download)
        else:
            train_loader, val_loader, test_loader, data_size = load_flowers(root='/root/autodl-tmp/project/training_models/dataset', download=args.download)

    # Recreate data loaders with DistributedSampler for DDP
    if is_distributed():
        if is_main_process():
            print("Recreating data loaders with DistributedSampler for DDP training...")
        train_loader = make_ddp_dataloader(train_loader, shuffle=True)
        test_loader = make_ddp_dataloader(test_loader, shuffle=False)
        if 'val_loader' in locals():
            val_loader = make_ddp_dataloader(val_loader, shuffle=False)

    if args.pretrained:
        pretrained = True
    
    if args.task == "classification" or args.task=="longtail":
        mz = ModelZoo(num_classes, pretrained)
    elif args.task == "segmentation":
        mz = ModelZoo(num_classes, pretrained)

    ###Models From Scratch###
    if args.model == "resnet18":
        # model = resnet18(num_classes=100)
        model = mz.resnet18()
    elif args.model == "efficientnet_b0":
        # model = efficientnet_b0(num_classes=100)
        model = mz.efficientnet_b0()

    ###PyTorch Models###
    elif args.model == "mobilenet_v2":
        model = mz.mobilenet_v2()
    elif args.model == "mobilenet_v3":
        model = mz.mobilenet_v3()
    elif args.model == "resnet34":
        model = mz.resnet34()
    elif args.model == "resnet50":
        model = mz.resnet50()
    elif args.model == "resnet101":
        model = mz.resnet101()
    elif args.model == "vit_b_16":
        model = mz.vit_b_16()
    elif args.model == "mae_vit_b_16":
        if not args.mae_checkpoint:
            parser.error("--mae_checkpoint is required when using --model mae_vit_b_16")
        # The 'pretrained' flag for ModelZoo is not directly used by mae_vit_b_16,
        # as it loads weights from the checkpoint_path.
        # However, ModelZoo still needs to be initialized.
        # We can pass False for pretrained here, or adjust ModelZoo if needed.
        # For simplicity, let's assume ModelZoo's pretrained flag is for its other models.
        model = mz.mae_vit_b_16(checkpoint_path=args.mae_checkpoint)
    elif args.model == "efficientformer":
        model = mz.efficientformer()
    elif args.model == "efficientnet_b7":
        model = mz.efficientnet_b7()
    elif args.model == "efficientnet_b4":
        model = mz.efficientnet_b4()
    elif args.model == "segformer_b2":
        # model = mz.segformer_b2()
        # model = mz.mmseg_model()
        # model = mz.lraspp_mobilenet_v3_large()
        model = mz.segformer()
    elif args.model == "resnet_3d":
        model = mz.resnet18_3d()

    # Move model to device
    model = model.to(device)

    # Wrap model with DDP if in distributed mode
    if is_distributed():
        if is_main_process():
            print(f"Converting BatchNorm to SyncBatchNorm for multi-GPU training...")
        # Convert all BatchNorm layers to SyncBatchNorm for proper multi-GPU training
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if is_main_process():
            print(f"Wrapping model with DistributedDataParallel on GPU {get_rank()}...")
        model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])], output_device=int(os.environ['LOCAL_RANK']))

    if args.pretrained:
        args.model = args.model + "_" + "pretrained" + "_" + str(args.threshold)
    else:
        args.model = args.model + "_" + str(args.threshold)

    if args.mode == "baseline":
        args.model = args.model + "_" + "baseline"

    if args.long_tail and args.ldam:
        if args.mode == "baseline":
            trained_model = train_baseline_longtail(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, cls_num_list)
        elif args.mode == "train_with_revision":
            trained_model = train_with_revision_longtail(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold, args.start_revision, args.task, cls_num_list)

    else: 
        if args.mode == "baseline":
            print("Training in baseline mode...")
            if args.noisy:
                trained_model = train_baseline_noisy(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, args.task, cls_num_list, args.learning_rate)
            else:
                trained_model = train_baseline(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, args.task, cls_num_list, args.learning_rate)
        elif args.mode == "selective_gradient":
            train_revision = TrainRevision(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold, args.learning_rate)
            print("Training with selective gradient updates...")
            trained_model = train_revision.train_selective()
        elif args.mode == "selective_epoch":
            train_revision = TrainRevision(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold, args.learning_rate)
            print(f"Reintroducing correct examples and training...")
            trained_model = train_revision.train_selective_epoch()
        elif args.mode == "train_with_revision":
            train_revision = TrainRevision(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold, args.learning_rate)
            print(f"Training {args.mode}, will start revision after {args.start_revision}")
            if args.noisy:
                trained_model, num_step = train_revision.train_with_noisy_revision(args.start_revision, args.task, cls_num_list)
            else:
                trained_model, num_step = train_revision.train_with_revision(args.start_revision, args.task, cls_num_list)
            print("Number of steps : ", num_step)
            eff_epoch = int(num_step / data_size)
            print(f"Effective Epochs: {eff_epoch} (out of {args.epoch} total epochs)")
        elif args.mode == "train_with_random":
            train_revision = TrainRevision(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold, args.learning_rate)
            print(f"Training {args.mode}, will start revision after {args.start_revision}")
            if args.noisy:
                trained_model, num_step = train_revision.train_with_noisy_random(args.start_revision, args.task)
            else:
                trained_model, num_step = train_revision.train_with_random(args.start_revision, args.task, num_classes, cls_num_list)
            print("Number of steps : ", num_step)
            eff_epoch = int(num_step / data_size)
            print(f"Effective Epochs: {eff_epoch} (out of {args.epoch} total epochs)")
        elif args.mode == "train_with_warmup_random":
            train_revision = TrainRevision(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold, args.learning_rate)
            print(f"Training {args.mode}, with {args.warmup_epochs} warmup epochs, will start revision after {args.start_revision}")
            trained_model, num_step = train_revision.train_with_warmup_random(args.start_revision, args.task, args.warmup_epochs)
            print("Number of steps : ", num_step)
            eff_epoch = int(num_step / data_size)
            print(f"Effective Epochs: {eff_epoch} (out of {args.epoch} total epochs)")
        elif args.mode == "train_with_periodic_full_random":
            train_revision = TrainRevision(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold, args.learning_rate)
            print(f"Training {args.mode}, with full dataset refresh every {args.refresh_interval} effective epochs, will start revision after {args.start_revision}")
            trained_model, num_step = train_revision.train_with_periodic_full_random(args.start_revision, args.task, args.refresh_interval)
            print("Number of steps : ", num_step)
            eff_epoch = int(num_step / data_size)
            print(f"Effective Epochs: {eff_epoch} (out of {args.epoch} total epochs)")
        elif args.mode == "train_with_revision_3d":
            train_revision = TrainRevision(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold, args.learning_rate)
            print(f"Training {args.mode}, will start revision after {args.start_revision}")
            trained_model, num_step = train_revision.train_with_revision_3d(args.start_revision, args.task)
            print("Number of steps : ", num_step)
            eff_epoch = int(num_step / data_size)
            print(f"Effective Epochs: {eff_epoch} (out of {args.epoch} total epochs)")
        elif args.mode == "train_with_percentage":
            train_revision = TrainRevision(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold, args.learning_rate)
            print(f"Training {args.mode}, will start revision after {args.start_revision}")
            if args.noisy:
                trained_model, num_step = train_revision.train_with_noisy_percentage(args.start_revision)
            else:
                trained_model, num_step = train_revision.train_with_percentage(args.start_revision)
            print("Number of steps : ", num_step)
            eff_epoch = int(num_step / data_size)
            print(f"Effective Epochs: {eff_epoch} (out of {args.epoch} total epochs)")
        elif args.mode == "train_with_inv_lin":
            train_revision = TrainRevision(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold, args.learning_rate)
            print(f"Training {args.mode}, will start revision after {args.start_revision}")
            trained_model, num_step = train_revision.train_with_inverse_linear(args.start_revision, data_size)
            print("Number of steps : ", num_step)
            eff_epoch = int(num_step / data_size)
            print(f"Effective Epochs: {eff_epoch} (out of {args.epoch} total epochs)")
        elif args.mode == "train_with_log":
            train_revision = TrainRevision(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold, args.learning_rate)
            print(f"Training {args.mode}, will start revision after {args.start_revision}")
            trained_model, num_step = train_revision.train_with_log(args.start_revision, data_size)
            print("Number of steps : ", num_step)
            eff_epoch = int(num_step / data_size)
            print(f"Effective Epochs: {eff_epoch} (out of {args.epoch} total epochs)")
        elif args.mode == "train_with_adaptive":
            train_revision = TrainRevision(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold, args.learning_rate)
            print(f"Training {args.mode}, will start revision after {args.start_revision}")
            trained_model, num_step = train_revision.train_with_adaptive(args.start_revision, args.task, cls_num_list, args.interval, args.increment)
            print("Number of steps : ", num_step)
            eff_epoch = int(num_step / data_size)
            print(f"Effective Epochs: {eff_epoch} (out of {args.epoch} total epochs)")
        elif args.mode == "train_with_alternative":
            train_revision = TrainRevision(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold, args.learning_rate)
            trained_model, num_step = train_revision.train_with_alternative(args.start_revision, args.task, cls_num_list)
            print("Number of steps : ", num_step)
            eff_epoch = int(num_step / data_size)
            print(f"Effective Epochs: {eff_epoch} (out of {args.epoch} total epochs)")
        elif args.mode == "train_with_stratified":
            train_revision = TrainRevision(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold, args.learning_rate)
            print(f"Training {args.mode} (Stratified Difficulty-Based Progressive Dropout)")
            print(f"Will start revision (full dataset) after epoch {args.start_revision}")
            trained_model, num_step = train_revision.train_with_stratified(args.start_revision, args.task, num_classes, cls_num_list)
            print("Number of steps : ", num_step)
            eff_epoch = int(num_step / data_size)
            print(f"Effective Epochs: {eff_epoch} (out of {args.epoch} total epochs)")
        elif args.mode == "train_with_adaptive_dropout":
            train_revision = TrainRevision(args.model, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold, args.learning_rate, args.scheduler_type)
            print(f"Training {args.mode} (Adaptive Proportionality Dropout)")
            print(f"Scheduler: {args.scheduler_type}, EMA Alpha: {args.ema_alpha}, Epsilon: {args.epsilon}")
            print(f"Will start revision (full dataset) after epoch {args.start_revision}")
            trained_model, num_step = train_revision.train_with_adaptive_dropout(args.start_revision, args.task, num_classes, args.ema_alpha, args.epsilon, cls_num_list)
            print("Number of steps : ", num_step)
            eff_epoch = int(num_step / data_size)
            print(f"Effective Epochs: {eff_epoch} (out of {args.epoch} total epochs)")

    # Only save from main process in DDP mode
    if is_main_process():
        # Save model (unwrap DDP wrapper if necessary)
        model_to_save = trained_model.module if isinstance(trained_model, DDP) else trained_model
        torch.save(model_to_save, "trained_model.pth")
        print("Model saved to trained_model.pth")

    # Cleanup DDP
    cleanup_ddp()

if __name__ == "__main__":
    main()
