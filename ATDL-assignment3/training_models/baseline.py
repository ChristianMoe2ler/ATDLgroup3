import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import time
from utils import log_memory, plot_metrics, plot_metrics_test, plot_accuracy_time_multi, plot_accuracy_time_multi_test, plot_f1_macro_time
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)

def train_baseline(model_name, model, train_loader, test_loader, device, epochs, save_path, task, cls_num_list, learning_rate=3e-4):
    model.to(device)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    if task=='classification':
            criterion = nn.CrossEntropyLoss()
    elif 'longtail':
        train_sampler = None
        idx = epochs // 160
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], cls_num_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(device)
        criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda(device)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
    epoch_losses = []
    epoch_accuracies = []
    epoch_test_accuracies = []
    epoch_test_losses = []
    epoch_train_f1_scores = []
    epoch_test_f1_scores = []
    time_per_epoch = []
    start_time = time.time()
    num_step = 0
    samples_used_per_epoch = []

    for epoch in range(epochs):
        samples_used = 0
        model.train()
        epoch_start_time = time.time()
        # Keep tensors on GPU to avoid sync - convert only at epoch end
        running_loss = torch.tensor(0.0, device=device)
        correct = torch.tensor(0, device=device)
        total = 0
        train_predictions = []
        train_labels = []

        print(f"Epoch [{epoch+1/epochs}]")
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")

        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            num_step+=len(outputs)
            samples_used+=len(outputs)

            # Accumulate on GPU - no sync!
            running_loss += loss.detach()

            # Reuse outputs from above - no need for second forward pass
            preds = torch.argmax(outputs.detach(), dim=1)
            correct += (preds == labels).sum()
            total += labels.size(0)

            # Collect predictions and labels for F1 calculation
            train_predictions.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            # Update progress bar less frequently
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{running_loss.item()/(batch_idx+1):.4f}"})

        # Only sync at epoch end - single transfer
        epoch_loss = (running_loss / len(train_loader)).item()
        epoch_accuracy = (correct / total).item()
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)

        # Calculate F1 macro score for training
        train_f1_macro = f1_score(train_labels, train_predictions, average='macro', zero_division=0)
        epoch_train_f1_scores.append(train_f1_macro)

        epoch_end_time = time.time()
        time_per_epoch.append(epoch_end_time - epoch_start_time)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Train F1 Macro: {train_f1_macro:.4f}")

        model.eval()
        # Keep on GPU to avoid sync
        test_correct = torch.tensor(0, device=device)
        test_total = 0
        test_loss = torch.tensor(0.0, device=device)
        test_predictions = []
        test_labels = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                test_loss += batch_loss.detach() * labels.size(0)
                predictions = torch.argmax(outputs, dim=-1)
                test_correct += (predictions == labels).sum()
                test_total += labels.size(0)

                # Collect predictions and labels for F1 calculation
                test_predictions.extend(predictions.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        # Only sync at end
        accuracy = (test_correct / test_total).item()
        val_loss = (test_loss / test_total).item()

        # Calculate F1 macro score for test
        test_f1_macro = f1_score(test_labels, test_predictions, average='macro', zero_division=0)
        epoch_test_f1_scores.append(test_f1_macro)

        print(f"Epoch {epoch + 1}/{epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}, Test F1 Macro: {test_f1_macro:.4f}")
        scheduler.step(val_loss)
        epoch_test_accuracies.append(accuracy)
        epoch_test_losses.append(val_loss)

    end_time = time.time()
    samples_used_per_epoch.append(samples_used)
    log_memory(start_time, end_time)
    print(num_step)

    # plot_metrics(epoch_losses, epoch_accuracies, "Baseline Training")
    # plot_metrics_test(epoch_test_accuracies, "Baseline Training")
    plot_accuracy_time_multi(
    model_name=model_name,
    accuracy=epoch_accuracies,
    time_per_epoch=time_per_epoch,
    save_path=save_path,
    data_file=save_path
    )
    plot_accuracy_time_multi_test(
        model_name = model_name,
        accuracy=epoch_test_accuracies,
        time_per_epoch=time_per_epoch,
        samples_per_epoch=samples_used_per_epoch,
        threshold=None,
        save_path=save_path,
        data_file=save_path
    )
    plot_f1_macro_time(
        model_name=model_name,
        f1_scores=epoch_train_f1_scores,
        time_per_epoch=time_per_epoch,
        save_path=save_path,
        data_file=save_path
    )
    return model

def train_baseline_noisy(model_name, model, train_loader, test_loader, device, epochs, save_path, task, cls_num_list, learning_rate=3e-4):
    model.to(device)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    if task=='classification':
            criterion = nn.CrossEntropyLoss()
    elif 'longtail':
        train_sampler = None
        idx = epochs // 160
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], cls_num_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(device)
        criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda(device)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
    epoch_losses = []
    epoch_accuracies = []
    epoch_test_accuracies = []
    epoch_test_losses = []
    epoch_train_f1_scores = []
    epoch_test_f1_scores = []
    time_per_epoch = []
    start_time = time.time()
    num_step = 0
    samples_used_per_epoch = []

    for epoch in range(epochs):
        samples_used = 0
        model.train()
        epoch_start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        train_predictions = []
        train_labels = []

        print(f"Epoch [{epoch+1/epochs}]")
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")


        for batch_idx, (inputs, labels, _) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            num_step+=len(outputs)
            samples_used+=len(outputs)

            running_loss += loss.item()

            # Reuse outputs from above - no need for second forward pass
            preds = torch.argmax(outputs.detach(), dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Collect predictions and labels for F1 calculation
            train_predictions.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)

        # Calculate F1 macro score for training
        train_f1_macro = f1_score(train_labels, train_predictions, average='macro', zero_division=0)
        epoch_train_f1_scores.append(train_f1_macro)

        epoch_end_time = time.time()
        time_per_epoch.append(epoch_end_time - epoch_start_time)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Train F1 Macro: {train_f1_macro:.4f}")

        model.eval()
        # Keep on GPU to avoid sync
        test_correct = torch.tensor(0, device=device)
        test_total = 0
        test_loss = torch.tensor(0.0, device=device)
        test_predictions = []
        test_labels = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                test_loss += batch_loss.detach() * labels.size(0)
                predictions = torch.argmax(outputs, dim=-1)
                test_correct += (predictions == labels).sum()
                test_total += labels.size(0)

                # Collect predictions and labels for F1 calculation
                test_predictions.extend(predictions.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        # Only sync at end
        accuracy = (test_correct / test_total).item()
        val_loss = (test_loss / test_total).item()

        # Calculate F1 macro score for test
        test_f1_macro = f1_score(test_labels, test_predictions, average='macro', zero_division=0)
        epoch_test_f1_scores.append(test_f1_macro)

        print(f"Epoch {epoch + 1}/{epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}, Test F1 Macro: {test_f1_macro:.4f}")
        scheduler.step(val_loss)
        epoch_test_accuracies.append(accuracy)
        epoch_test_losses.append(val_loss)

    end_time = time.time()
    samples_used_per_epoch.append(samples_used)
    log_memory(start_time, end_time)
    print(num_step)

    # plot_metrics(epoch_losses, epoch_accuracies, "Baseline Training")
    # plot_metrics_test(epoch_test_accuracies, "Baseline Training")
    plot_accuracy_time_multi(
    model_name=model_name,
    accuracy=epoch_accuracies,
    time_per_epoch=time_per_epoch,
    save_path=save_path,
    data_file=save_path
    )
    plot_accuracy_time_multi_test(
        model_name = model_name,
        accuracy=epoch_test_accuracies,
        time_per_epoch=time_per_epoch,
        samples_per_epoch=samples_used_per_epoch,
        threshold=None,
        save_path=save_path,
        data_file=save_path
    )
    plot_f1_macro_time(
        model_name=model_name,
        f1_scores=epoch_train_f1_scores,
        time_per_epoch=time_per_epoch,
        save_path=save_path,
        data_file=save_path
    )
    return model