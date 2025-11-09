import argparse
import torch
import torch.nn as nn
from model import resnet18, efficientnet_b0
from model_zoo import ModelZoo
from data import load_cifar100, load_mnist, load_imagenet, load_cityscapes, load_cifar10, load_medmnist3D, load_noisy
from data import load_cifar100, load_mnist, load_imagenet, load_cityscapes, load_cifar10, load_medmnist3D, load_cub2011, load_aircraft, load_flowers
from baseline import train_baseline, train_baseline_noisy
from selective_gradient import TrainRevision
from test import test_model
from longtail_train import train_baseline_longtail, train_with_revision_longtail

def freeze_layers(model, freeze_strategy, model_name):
    """
    Freeze layers of the model based on the specified strategy.

    Args:
        model: The model to freeze layers in
        freeze_strategy: Strategy for freezing ('all', 'partial', 'none', or layer count)
        model_name: Name of the model architecture
    """
    if freeze_strategy == "none":
        print("No layers frozen - full fine-tuning")
        return model

    if freeze_strategy == "all":
        print("Freezing all layers except classifier head")
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the classifier/head based on model type
        if "resnet" in model_name:
            for param in model.fc.parameters():
                param.requires_grad = True
        elif "mobilenet" in model_name:
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif "efficientnet" in model_name:
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif "vit" in model_name or "mae" in model_name:
            for param in model.heads.parameters():
                param.requires_grad = True
        elif "segformer" in model_name:
            for param in model.decode_head.parameters():
                param.requires_grad = True

    elif freeze_strategy == "partial":
        print("Freezing early layers (backbone) - partial fine-tuning")
        freeze_count = 0
        total_params = sum(1 for _ in model.parameters())
        freeze_threshold = int(total_params * 0.7)  # Freeze first 70% of layers

        for i, param in enumerate(model.parameters()):
            if i < freeze_threshold:
                param.requires_grad = False
                freeze_count += 1
            else:
                param.requires_grad = True
        print(f"Froze {freeze_count}/{total_params} parameter groups")

    elif freeze_strategy.isdigit():
        # Freeze specific number of layers
        num_freeze = int(freeze_strategy)
        print(f"Freezing first {num_freeze} layers")

        frozen = 0
        for name, param in model.named_parameters():
            if frozen < num_freeze:
                param.requires_grad = False
                frozen += 1
            else:
                param.requires_grad = True

    return model

def load_checkpoint(model, checkpoint_path, device):
    """
    Load model weights from checkpoint file.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint to
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    print("Checkpoint loaded successfully")
    return model

def main():
    parser = argparse.ArgumentParser(description="Fine-tune pretrained models with Progressive Dropout")
    parser.add_argument("--mode", type=str, choices=["baseline", "selective_gradient", "selective_epoch", "train_with_revision", "train_with_samples", "train_with_revision_3d", "train_with_random", "train_with_inv_lin", "train_with_log", "train_with_percentage",
                                                     "train_with_adaptive", "train_with_alternative"], required=True,
                        help="Choose training mode: 'baseline', 'train_with_revision', etc.")
    parser.add_argument("--epoch", type=int, required=False, default=10,
                        help="Number of epochs to fine-tune for")
    parser.add_argument("--task", type=str, required=True, default="classification",
                        help="segmentation or classification or longtail")
    parser.add_argument("--model", type=str, choices=["resnet18", "resnet_3d", "resnet34", "resnet50", "resnet101", "efficientnet_b0","efficientnet_b7", "efficientnet_b4", "mobilenet_v2", "mobilenet_v3", "vit_b_16", "mae_vit_b_16", "efficientformer", "segformer_b2"], required=True,
                        help="Choose the model architecture")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file to load pretrained weights from")
    parser.add_argument("--mae_checkpoint", type=str, default=None, help="Path to MAE pretrained checkpoint file (used with --model mae_vit_b_16)")
    parser.add_argument("--freeze", type=str, default="none", choices=["all", "partial", "none"],
                        help="Layer freezing strategy: 'all' (freeze all except head), 'partial' (freeze 70%% backbone), 'none' (full fine-tuning)")
    parser.add_argument("--freeze_layers", type=int, default=None, help="Number of layers to freeze from the beginning (alternative to --freeze)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for fine-tuning (default: 1e-4)")
    parser.add_argument("--save_path", type=str, help="Path to save results and plots")
    parser.add_argument("--threshold", type=float, help="Threshold for sample dropout")
    parser.add_argument("--epoch_threshold", type=int, help="Threshold to reintroduce correct samples in epoch")
    parser.add_argument("--dataset", type=str, help="Dataset to fine-tune on: cifar, cifar10, mnist, imagenet, etc.")
    parser.add_argument("--batch_size", type=int, help="Batch size (e.g., 32, 64, 128)")
    parser.add_argument("--start_revision", type=int, help="Start revision after the given epoch")
    parser.add_argument("--long_tail", action="store_true", help="Use long-tail (imbalanced) dataset")
    parser.add_argument("--ldam", action="store_true", help="Use LDAM-DRW method for long tail classification")
    parser.add_argument("--noisy", action="store_true", help="Use noisy dataset")
    parser.add_argument("--interval", type=int, default=50, help="Interval for adaptive mode")
    parser.add_argument("--increment", type=float, default=0.1, help="Increment for adaptive mode")
    parser.add_argument("--download", action="store_true", help="Download dataset if not exists (for aircraft and cub2011 datasets)")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd", "adam"], help="Optimizer for fine-tuning")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--save_model_path", type=str, default="finetuned_model.pth", help="Path to save the fine-tuned model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    if args.dataset == "mnist":
        num_classes = 10
        train_loader, test_loader = load_mnist()
        data_size = len(train_loader.dataset)
    elif args.dataset == "cifar":
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
        data_size = len(train_loader.dataset)
    elif args.dataset == "organ_medmnist3d":
        num_classes = 11
        train_loader, test_loader, data_size = load_medmnist3D(args.batch_size)
    elif args.dataset == "aircraft":
        num_classes = 100
        if args.batch_size:
            train_loader, test_loader, data_size = load_aircraft(args.batch_size, root='/root/autodl-tmp/project/training_models/dataset', download=args.download)
        else:
            train_loader, test_loader, data_size = load_aircraft(download=args.download)
    elif args.dataset == "cub2011":
        num_classes = 200
        if args.batch_size:
            train_loader, test_loader, data_size = load_cub2011(args.batch_size, root='/root/autodl-tmp/project/training_models/dataset', download=args.download)
        else:
            train_loader, test_loader, data_size = load_cub2011(root='/root/autodl-tmp/project/training_models/dataset', download=args.download)
    elif args.dataset == "flowers":
        num_classes = 102
        if args.batch_size:
            train_loader, val_loader, test_loader, data_size = load_flowers(args.batch_size, root='/root/autodl-tmp/project/training_models/dataset', download=args.download)
        else:
            train_loader, val_loader, test_loader, data_size = load_flowers(root='/root/autodl-tmp/project/training_models/dataset', download=args.download)

    if not args.long_tail:
        cls_num_list = None

    # Always use pretrained models for fine-tuning
    pretrained = True
    print("Fine-tuning mode: Loading pretrained model")

    if args.task == "classification" or args.task == "longtail":
        mz = ModelZoo(num_classes, pretrained)
    elif args.task == "segmentation":
        mz = ModelZoo(num_classes, pretrained)

    # Load model architecture with pretrained weights
    if args.model == "resnet18":
        model = mz.resnet18()
    elif args.model == "efficientnet_b0":
        model = mz.efficientnet_b0()
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
        model = mz.mae_vit_b_16(checkpoint_path=args.mae_checkpoint)
    elif args.model == "efficientformer":
        model = mz.efficientformer()
    elif args.model == "efficientnet_b7":
        model = mz.efficientnet_b7()
    elif args.model == "efficientnet_b4":
        model = mz.efficientnet_b4()
    elif args.model == "segformer_b2":
        model = mz.segformer()
    elif args.model == "resnet_3d":
        model = mz.resnet18_3d()

    # Load custom checkpoint if provided (for continuing from previous training)
    if args.checkpoint:
        model = load_checkpoint(model, args.checkpoint, device)

    model = model.to(device)

    # Apply layer freezing strategy
    if args.freeze_layers is not None:
        model = freeze_layers(model, str(args.freeze_layers), args.model)
    else:
        model = freeze_layers(model, args.freeze, args.model)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Update model name for saving
    model_save_name = f"{args.model}_finetuned_threshold_{args.threshold}"
    if args.mode == "baseline":
        model_save_name = model_save_name + "_baseline"

    # Training logic - same as main.py but adapted for fine-tuning
    if args.long_tail and args.ldam:
        if args.mode == "baseline":
            trained_model = train_baseline_longtail(model_save_name, model, train_loader, test_loader, device, args.epoch, args.save_path, cls_num_list)
        elif args.mode == "train_with_revision":
            trained_model = train_with_revision_longtail(model_save_name, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold, args.start_revision, args.task, cls_num_list)
    else:
        if args.mode == "baseline":
            print("Fine-tuning in baseline mode...")
            if args.noisy:
                trained_model = train_baseline_noisy(model_save_name, model, train_loader, test_loader, device, args.epoch, args.save_path, args.task, cls_num_list)
            else:
                trained_model = train_baseline(model_save_name, model, train_loader, test_loader, device, args.epoch, args.save_path, args.task, cls_num_list)
        elif args.mode == "selective_gradient":
            train_revision = TrainRevision(model_save_name, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold)
            print("Fine-tuning with selective gradient updates...")
            trained_model = train_revision.train_selective()
        elif args.mode == "selective_epoch":
            train_revision = TrainRevision(model_save_name, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold)
            print(f"Fine-tuning with selective epoch strategy...")
            trained_model = train_revision.train_selective_epoch()
        elif args.mode == "train_with_revision":
            train_revision = TrainRevision(model_save_name, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold)
            print(f"Fine-tuning with {args.mode}, will start revision after epoch {args.start_revision}")
            if args.noisy:
                trained_model, num_step = train_revision.train_with_noisy_revision(args.start_revision, args.task, cls_num_list)
            else:
                trained_model, num_step = train_revision.train_with_revision(args.start_revision, args.task, cls_num_list)
            print("Number of steps : ", num_step)
        elif args.mode == "train_with_random":
            train_revision = TrainRevision(model_save_name, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold)
            print(f"Fine-tuning with {args.mode}, will start revision after epoch {args.start_revision}")
            if args.noisy:
                trained_model, num_step = train_revision.train_with_noisy_random(args.start_revision, args.task)
            else:
                trained_model, num_step = train_revision.train_with_random(args.start_revision, args.task)
            print("Number of steps : ", num_step)
        elif args.mode == "train_with_revision_3d":
            train_revision = TrainRevision(model_save_name, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold)
            print(f"Fine-tuning with {args.mode}, will start revision after epoch {args.start_revision}")
            trained_model, num_step = train_revision.train_with_revision_3d(args.start_revision, args.task)
            print("Number of steps : ", num_step)
        elif args.mode == "train_with_percentage":
            train_revision = TrainRevision(model_save_name, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold)
            print(f"Fine-tuning with {args.mode}, will start revision after epoch {args.start_revision}")
            if args.noisy:
                trained_model, num_step = train_revision.train_with_noisy_percentage(args.start_revision)
            else:
                trained_model, num_step = train_revision.train_with_percentage(args.start_revision)
            print("Number of steps : ", num_step)
        elif args.mode == "train_with_inv_lin":
            train_revision = TrainRevision(model_save_name, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold)
            print(f"Fine-tuning with {args.mode}, will start revision after epoch {args.start_revision}")
            trained_model, num_step = train_revision.train_with_inverse_linear(args.start_revision, data_size)
            print("Number of steps : ", num_step)
        elif args.mode == "train_with_log":
            train_revision = TrainRevision(model_save_name, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold)
            print(f"Fine-tuning with {args.mode}, will start revision after epoch {args.start_revision}")
            trained_model, num_step = train_revision.train_with_log(args.start_revision, data_size)
            print("Number of steps : ", num_step)
        elif args.mode == "train_with_adaptive":
            train_revision = TrainRevision(model_save_name, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold)
            print(f"Fine-tuning with {args.mode}, will start revision after epoch {args.start_revision}")
            trained_model, num_step = train_revision.train_with_adaptive(args.start_revision, args.task, cls_num_list, args.interval, args.increment)
            print("Number of steps : ", num_step)
        elif args.mode == "train_with_alternative":
            train_revision = TrainRevision(model_save_name, model, train_loader, test_loader, device, args.epoch, args.save_path, args.threshold)
            print(f"Fine-tuning with {args.mode}, will start revision after epoch {args.start_revision}")
            trained_model, num_step = train_revision.train_with_alternative(args.start_revision, args.task, cls_num_list)
            print("Number of steps : ", num_step)

    # Calculate effective epochs
    if args.mode == "baseline":
        num_step = data_size * args.epoch

    if 'num_step' in locals():
        eff_epoch = int(num_step / data_size)
        print("Effective Epochs: ", eff_epoch)
    else:
        print("Fine-tuning completed in baseline mode")

    # Save the fine-tuned model
    print(f"Saving fine-tuned model to {args.save_model_path}")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'model_name': args.model,
        'num_classes': num_classes,
        'freeze_strategy': args.freeze,
        'learning_rate': args.learning_rate,
        'epochs': args.epoch,
        'dataset': args.dataset
    }, args.save_model_path)
    print("Fine-tuning complete!")

if __name__ == "__main__":
    main()
