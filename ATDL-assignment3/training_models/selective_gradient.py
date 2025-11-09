import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch.distributed as dist
import time
from utils import log_memory, plot_metrics, plot_metrics_test, plot_accuracy_time_multi, plot_accuracy_time_multi_test, plot_f1_macro_time
from tqdm import tqdm
import json
import os
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
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

class TrainRevision:
    def __init__(self, model_name, model, train_loader, test_loader, device, epochs, save_path, threshold, learning_rate=3e-4, scheduler_type="step"):
        self.model_name = model_name
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.save_path = save_path
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.scheduler_type = scheduler_type

    def _set_sampler_epoch(self, epoch):
        """Set epoch for DistributedSampler to ensure proper shuffling in DDP mode"""
        if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)

    def _evaluate_with_f1(self, criterion, num_classes, data_loader=None):
        """
        Evaluate model with F1 macro score in addition to accuracy and loss.

        Args:
            criterion: Loss function to use
            num_classes: Number of classes in the dataset
            data_loader: DataLoader to evaluate on (defaults to self.test_loader)

        Returns:
            accuracy: Test accuracy
            f1_macro: F1 macro score
            val_loss: Average validation loss
        """
        if data_loader is None:
            data_loader = self.test_loader

        self.model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating", leave=False):
                inputs = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                outputs = self.model(inputs)

                batch_loss = criterion(outputs, labels)
                test_loss += batch_loss.item()

                predictions = torch.argmax(outputs, dim=-1)
                test_correct += (predictions == labels).sum().item()
                test_total += labels.size(0)

                # Collect predictions and labels for F1 calculation
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = test_correct / test_total if test_total > 0 else 0
        val_loss = test_loss / len(data_loader) if len(data_loader) > 0 else 0

        # Calculate F1 macro score
        f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

        return accuracy, f1_macro, val_loss

    def train_selective(self):
        self.model.to(self.device)
        save_path = self.save_path
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
        # optimizer = optim.AdamW(self.model.parameters(), lr=3e-4)
        epoch_losses = []
        epoch_accuracies = []
        epoch_test_accuracies = []
        epoch_test_losses = []
        time_per_epoch = []
        start_time = time.time()
        for epoch in range(self.epochs):
            self.model.train()
            epoch_start_time = time.time()
            running_loss = 0.0
            correct = 0
            total_correct = 0
            total_samples = 0
            total = 0
            print(f"Epoch [{epoch+1}/{self.epochs}]")
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")

            for batch_idx, (inputs, labels) in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    
                    if self.threshold == 0:
                        mask = preds != labels
                    else:
                        prob = torch.softmax(outputs, dim=1)
                        correct_class = prob[torch.arange(labels.size(0)), labels]
                        mask = correct_class < self.threshold

                # In DDP mode, ensure we process every batch
                if not mask.any():
                    mask[0] = True

                inputs_misclassified = inputs[mask]
                labels_misclassified = labels[mask]

                # if inputs_misclassified.size(0) < 2:
                #     continue

                # if inputs_misclassified.size(0) < 2:
                #     required_samples = 2 - inputs_misclassified.size(0)
                #     correctly_classified_mask = ~mask
                #     correct_inputs = inputs[correctly_classified_mask][:required_samples]
                #     correct_labels = labels[correctly_classified_mask][:required_samples]

                #     inputs_misclassified = torch.cat((inputs_misclassified, correct_inputs), dim=0)
                #     labels_misclassified = torch.cat((labels_misclassified, correct_labels), dim=0)

                optimizer.zero_grad()

                outputs_misclassified = self.model(inputs_misclassified)
                # outputs_misclassified = outputs[mask]
                loss = criterion(outputs_misclassified, labels_misclassified)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # preds_misclassified = torch.argmax(outputs_misclassified, dim=1)
                # correct += (preds_misclassified == labels_misclassified).sum().item()
                # # total += labels_misclassified.size(0)
                # with torch.no_grad():
                    # outputs = model(inputs)
                    # preds = torch.argmax(outputs, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                progress_bar.set_postfix({"Loss": loss.item()})

            epoch_loss = running_loss / len(self.train_loader)
            # epoch_accuracy = correct / total if total > 0 else 0
            epoch_accuracy = total_correct/total_samples if total_samples > 0 else 0 
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)

            epoch_end_time = time.time()
            time_per_epoch.append(epoch_end_time-epoch_start_time)

            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in tqdm(self.test_loader, desc="Evaluating"):
                    inputs = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                    outputs = self.model(inputs)
                    predictions = torch.argmax(outputs, dim=-1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

            accuracy = correct / total
            val_loss = criterion(correct, total)

            print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
            scheduler.step(val_loss)
            epoch_test_accuracies.append(accuracy)
            epoch_test_losses.append(val_loss)
            

        end_time = time.time()
        log_memory(start_time, end_time)

        plot_metrics(epoch_losses, epoch_accuracies, "Selective Training")
        plot_metrics_test(epoch_test_accuracies, "Selective Training")
        # plot_accuracy_time(epoch_accuracies, time_per_epoch, title="Accuracy and Time per Epoch", save_path=save_path)
        plot_accuracy_time_multi(
        model_name= self.model_name,  
        accuracy=epoch_accuracies,
        time_per_epoch=time_per_epoch,  
        save_path=save_path,
        data_file=save_path
        )
        plot_accuracy_time_multi_test(
            model_name = self.model_name,
            accuracy=epoch_test_accuracies,
            time_per_epoch=time_per_epoch,
            save_path=save_path,
            data_file=save_path
        )

        return self.model

    def train_selective_epoch(self):
        self.model.to(self.device)
        save_path = self.save_path
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        epoch_losses = []
        epoch_accuracies = []
        time_per_epoch = []
        start_time = time.time()

        accumulated_inputs = []
        accumulated_labels = []
        max_accumulated_samples = 128

        for epoch in range(self.epochs):
            self.model.train()
            epoch_start_time = time.time()
            running_loss = 0.0
            total_correct = 0
            total_samples = 0

            print(f"Epoch [{epoch+1}/{self.epochs}]")
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")

            for batch_idx, (inputs, labels) in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if epoch < self.epochs:
                    with torch.no_grad():
                        outputs = self.model(inputs)
                        preds = torch.argmax(outputs, dim=1)
                        
                        if self.threshold == 0:
                            mask = preds != labels
                            mask_correct = preds == labels
                        else:
                            prob = torch.softmax(outputs, dim=1)
                            correct_class = prob[torch.arange(labels.size(0)), labels]
                            mask = correct_class < self.threshold
                            mask_correct = correct_class > self.threshold

                    # Keep on GPU - no need to transfer to CPU and back
                    accumulated_inputs.append(inputs[mask_correct])
                    accumulated_labels.append(labels[mask_correct])

                    if len(accumulated_inputs) >= max_accumulated_samples:
                        reintroduced_inputs = torch.cat(accumulated_inputs, dim=0)
                        reintroduced_labels = torch.cat(accumulated_labels, dim=0)

                        accumulated_inputs = []  
                        accumulated_labels = []

                        inputs_selected = torch.cat((inputs, reintroduced_inputs), dim=0)
                        labels_selected = torch.cat((labels, reintroduced_labels), dim=0)

                    else:
                        # In DDP mode, ensure we process every batch
                        if not mask.any():
                            mask[0] = True

                        inputs_selected = inputs[mask]
                        labels_selected = labels[mask]

                    # In DDP mode, ensure we process every batch
                    if not mask.any():
                        mask[0] = True

                    inputs_selected = inputs[mask]
                    labels_selected = labels[mask]
                else:
                    if accumulated_inputs:
                        # Already on GPU from previous fix
                        reintroduced_inputs = torch.cat(accumulated_inputs, dim=0)
                        reintroduced_labels = torch.cat(accumulated_labels, dim=0)

                        accumulated_inputs = []
                        accumulated_labels = []

                        inputs_selected = torch.cat((inputs, reintroduced_inputs), dim=0)
                        labels_selected = torch.cat((labels, reintroduced_labels), dim=0)
                    else:
                        print("No accumulated samples")
                        inputs_selected = inputs
                        labels_selected = labels

                optimizer.zero_grad()
                outputs_selected = self.model(inputs_selected)
                loss = criterion(outputs_selected, labels_selected)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                with torch.no_grad():
                    outputs = self.model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    total_correct += (preds == labels).sum().item()
                    total_samples += labels.size(0)
                
                progress_bar.set_postfix({"Loss": loss.item()})

            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)

            epoch_end_time = time.time()
            time_per_epoch.append(epoch_end_time - epoch_start_time)

            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        end_time = time.time()
        log_memory(start_time, end_time)

        plot_metrics(epoch_losses, epoch_accuracies, "Selective Training with Reintroduction")
        plot_accuracy_time_multi(
            model_name=self.model_name,
            accuracy=epoch_accuracies,
            time_per_epoch=time_per_epoch,
            save_path=save_path,
            data_file=save_path
        )

        return self.model


    def train_with_revision(self, start_revision, task, cls_num_list):

        save_path = self.save_path
        self.model.to(self.device)
        if task=='classification':
            criterion = nn.CrossEntropyLoss()
        elif 'longtail':
            train_sampler = None
            idx = self.epochs // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(self.device)
            criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda(self.device)
        # optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        #as per implementation LR=0.045, they use 16 GPU. https://discuss.pytorch.org/t/training-mobilenet-on-imagenet/174391/6 from this blog
        #we use the idea to divide the learning rate by the number of GPUs. 
        # optimizer = optim.RMSprop(self.model.parameters(), weight_decay=0.00004, momentum=0.9, lr=0.0028125)   
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
        epoch_losses = []
        epoch_accuracies = []
        epoch_test_accuracies = []
        epoch_test_losses = []
        epoch_train_f1_scores = []
        epoch_test_f1_scores = []
        time_per_epoch = []
        survival_log = defaultdict(list)
        label_log = defaultdict(int)
        start_time = time.time()
        num_step = 0
        samples_used_per_epoch = []
        for epoch in range(self.epochs):
            # Set epoch for DistributedSampler (important for DDP)
            self._set_sampler_epoch(epoch)

            samples_used = 0
            if epoch < start_revision :
                self.model.train()
                epoch_start_time = time.time()
                # Keep tensors on GPU to avoid sync - convert only at epoch end
                running_loss = torch.tensor(0.0, device=self.device)
                total_correct = torch.tensor(0, device=self.device)
                total_samples = 0
                train_predictions = []
                train_labels = []
                print(f"Epoch [{epoch+1}/{self.epochs}]")
                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")

                for batch_idx, (inputs, labels) in progress_bar:
                    batch_start_idx = batch_idx * self.train_loader.batch_size
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    with torch.no_grad():
                        outputs = self.model(inputs)
                        preds = torch.argmax(outputs, dim=1)

                        if self.threshold == 0:
                            mask = preds != labels
                        else:
                            prob = torch.softmax(outputs, dim=1)
                            correct_class = prob[torch.arange(labels.size(0)), labels]
                            mask = correct_class < self.threshold

                    # In DDP mode, we must process every batch to avoid desync
                    # If no samples pass threshold, use at least 1 sample
                    if not mask.any():
                        mask[0] = True

                    inputs_misclassified = inputs[mask]
                    labels_misclassified = labels[mask]

                    # Logging disabled for performance - these operations are very slow
                    # used_labels = labels_misclassified
                    # for label in used_labels.tolist():
                    #     label_log[int(label)] += 1
                    #
                    # misclassified_in_batch = torch.nonzero(mask, as_tuple=False).squeeze(1)
                    # absolute_indices = (misclassified_in_batch + batch_start_idx).tolist()
                    # survival_log[epoch].extend(absolute_indices)

                    optimizer.zero_grad()

                    outputs_misclassified = self.model(inputs_misclassified)
                    loss = criterion(outputs_misclassified, labels_misclassified)
                    num_step+=len(outputs_misclassified)
                    samples_used+=len(outputs_misclassified)
                    loss.backward()
                    optimizer.step()

                    # Accumulate on GPU - no sync!
                    running_loss += loss.detach()
                    total_correct += (preds == labels).sum()
                    total_samples += labels.size(0)

                    # Collect predictions and labels for F1 calculation
                    train_predictions.extend(preds.cpu().numpy())
                    train_labels.extend(labels.cpu().numpy())

                    # Update progress bar less frequently to reduce overhead
                    if batch_idx % 10 == 0:
                        progress_bar.set_postfix({"Loss": f"{running_loss.item()/(batch_idx+1):.4f}"})

                # Only sync at epoch end - single transfer
                epoch_loss = (running_loss / len(self.train_loader)).item()
                epoch_accuracy = (total_correct / total_samples).item() if total_samples > 0 else 0
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                # Calculate F1 macro score for training
                train_f1_macro = f1_score(train_labels, train_predictions, average='macro', zero_division=0)
                epoch_train_f1_scores.append(train_f1_macro)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time-epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Train F1 Macro: {train_f1_macro:.4f}")

                self.model.eval()
                # Keep on GPU to avoid sync
                correct = torch.tensor(0, device=self.device)
                total = 0
                test_loss = torch.tensor(0.0, device=self.device)
                test_predictions = []
                test_labels = []
                with torch.no_grad():
                    for batch in tqdm(self.test_loader, desc="Evaluating"):
                        inputs = batch[0].to(self.device)
                        labels = batch[1].to(self.device)
                        outputs = self.model(inputs)

                        batch_loss = criterion(outputs, labels)
                        test_loss += batch_loss.detach() * labels.size(0)

                        predictions = torch.argmax(outputs, dim=-1)
                        correct += (predictions == labels).sum()
                        total += labels.size(0)

                        # Collect predictions and labels for F1 calculation
                        test_predictions.extend(predictions.cpu().numpy())
                        test_labels.extend(labels.cpu().numpy())

                # Only sync at end
                accuracy = (correct / total).item()
                val_loss = (test_loss / total).item()

                # Calculate F1 macro score for test
                test_f1_macro = f1_score(test_labels, test_predictions, average='macro', zero_division=0)
                epoch_test_f1_scores.append(test_f1_macro)

                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}, Test F1 Macro: {test_f1_macro:.4f}")
                scheduler.step(val_loss)
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)

            else:
                self.model.train()
                # Keep tensors on GPU to avoid sync - convert only at epoch end
                running_loss = torch.tensor(0.0, device=self.device)
                correct = torch.tensor(0, device=self.device)
                total = 0
                train_predictions = []
                train_labels = []

                print(f"Epoch [{epoch+1}/{self.epochs}]")
                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")

                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()

                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    # Logging disabled for performance
                    # absolute_indices = list(range(batch_start_idx, batch_start_idx + inputs.size(0)))
                    # survival_log[epoch].extend(absolute_indices)
                    # used_labels = labels
                    # for label in used_labels.tolist():
                    #     label_log[int(label)] += 1
                    num_step+=len(outputs)
                    samples_used+=len(outputs)
                    loss.backward()
                    optimizer.step()

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
                epoch_loss = (running_loss / len(self.train_loader)).item()
                epoch_accuracy = (correct / total).item()
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                # Calculate F1 macro score for training
                train_f1_macro = f1_score(train_labels, train_predictions, average='macro', zero_division=0)
                epoch_train_f1_scores.append(train_f1_macro)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time - epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Train F1 Macro: {train_f1_macro:.4f}")

                self.model.eval()
                test_correct = 0
                test_total = 0
                test_loss = 0.0
                test_predictions = []
                test_labels = []
                with torch.no_grad():
                    for batch in tqdm(self.test_loader, desc="Evaluating"):
                        inputs = batch[0].to(self.device)
                        labels = batch[1].to(self.device)
                        outputs = self.model(inputs)

                        batch_loss = criterion(outputs, labels)
                        test_loss+=batch_loss.item()

                        predictions = torch.argmax(outputs, dim=-1)
                        test_correct += (predictions == labels).sum().item()
                        test_total += labels.size(0)

                        # Collect predictions and labels for F1 calculation
                        test_predictions.extend(predictions.cpu().numpy())
                        test_labels.extend(labels.cpu().numpy())

                accuracy = test_correct / test_total
                val_loss = test_loss / len(self.test_loader)

                # Calculate F1 macro score for test
                test_f1_macro = f1_score(test_labels, test_predictions, average='macro', zero_division=0)
                epoch_test_f1_scores.append(test_f1_macro)

                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}, Test F1 Macro: {test_f1_macro:.4f}")
                scheduler.step(val_loss)
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)

            samples_used_per_epoch.append(samples_used)


        end_time = time.time()
        log_memory(start_time, end_time)
        print(num_step)

        total_wall_time = end_time - start_time
        print(f"\n Total Wall Time for {self.epochs} epochs: {total_wall_time:.2f} seconds "
            f"({total_wall_time / 60:.2f} minutes)")


        plot_accuracy_time_multi(
        model_name=self.model_name,  
        accuracy=epoch_accuracies,
        time_per_epoch=time_per_epoch,  
        save_path=save_path,
        data_file=save_path
        )
        plot_accuracy_time_multi_test(
            model_name = self.model_name,
            accuracy=epoch_test_accuracies,
            time_per_epoch=time_per_epoch,
            samples_per_epoch=samples_used_per_epoch,
            threshold=self.threshold,
            save_path=save_path,
            data_file=save_path
        )
        plot_f1_macro_time(
            model_name=self.model_name,
            f1_scores=epoch_train_f1_scores,
            time_per_epoch=time_per_epoch,
            save_path=save_path,
            data_file=save_path
        )

        # Logging disabled for performance
        # survival_log_path = os.path.join(os.path.dirname(save_path), "survival_log_eff.json")
        # with open(survival_log_path, "w") as f:
        #     json.dump(dict(survival_log), f, indent=2)
        # print(f"Survival log saved to {survival_log_path}")
        #
        # label_log_path = os.path.join(os.path.dirname(save_path), "label_log_eff.json")
        # with open(label_log_path, "w") as f:
        #     json.dump(dict(label_log), f, indent=2)
        # print(f"Survival log saved to {label_log_path}")

        return self.model, num_step


    def train_with_random(self, start_revision, task, num_classes, cls_num_list=None):

        save_path = self.save_path
        self.model.to(self.device)

        if task=='classification':
            criterion = nn.CrossEntropyLoss()
        elif task == 'longtail':
            train_sampler = None
            idx = self.epochs // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(self.device)
            criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

        epoch_losses = []
        epoch_accuracies = []
        epoch_test_accuracies = []
        epoch_test_losses = []
        epoch_test_f1_macros = []  # Track F1 macro scores
        time_per_epoch = []
        samples_used_per_epoch = []

        start_time = time.time()
        num_step = 0

        for epoch in range(self.epochs):
            # Set epoch for DistributedSampler (important for DDP)
            self._set_sampler_epoch(epoch)

            samples_used = 0
            print(f"Epoch [{epoch+1}/{self.epochs}]")

            if epoch < start_revision:
                self.model.train()
                epoch_start_time = time.time()
                running_loss = 0.0
                total_correct = 0
                total_samples = 0

                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")

                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    with torch.no_grad():
                        outputs = self.model(inputs)
                        preds = torch.argmax(outputs, dim=1)

                        if self.threshold == 0:
                            mask = preds != labels
                        else:
                            prob = torch.softmax(outputs, dim=1)
                            correct_class = prob[torch.arange(labels.size(0)), labels]
                            mask = correct_class < self.threshold

                        num_to_select = mask.sum().item()

                    # In DDP mode, we must process every batch to avoid desync
                    # Use at least 1 sample to ensure gradient sync happens
                    if num_to_select == 0:
                        num_to_select = 1

                    # 🔁 Random sampling based on how many passed threshold
                    indices = torch.randperm(inputs.size(0))[:num_to_select]
                    inputs_sampled = inputs[indices]
                    labels_sampled = labels[indices]

                    optimizer.zero_grad()
                    outputs_sampled = self.model(inputs_sampled)
                    loss = criterion(outputs_sampled, labels_sampled)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    num_step += len(outputs_sampled)
                    samples_used += len(outputs_sampled)

                    # Stats on original batch
                    with torch.no_grad():
                        outputs = self.model(inputs)
                        preds = torch.argmax(outputs, dim=1)
                        total_correct += (preds == labels).sum().item()
                        total_samples += labels.size(0)

                    progress_bar.set_postfix({"Loss": loss.item()})

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time - epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

                # Evaluation with F1 macro score
                accuracy, f1_macro, val_loss = self._evaluate_with_f1(criterion, num_classes)
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, F1 Macro: {f1_macro:.4f}, Test Loss: {val_loss:.4f}")
                scheduler.step(val_loss)
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)
                epoch_test_f1_macros.append(f1_macro)

            else:
                self.model.train()
                epoch_start_time = time.time()
                running_loss = 0.0
                correct = 0
                total = 0

                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")
                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    num_step += len(outputs)
                    samples_used += len(outputs)

                    with torch.no_grad():
                        preds = torch.argmax(outputs, dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)

                    progress_bar.set_postfix({"Loss": loss.item()})

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = correct / total
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time - epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

                # Evaluation with F1 macro score
                accuracy, f1_macro, val_loss = self._evaluate_with_f1(criterion, num_classes)
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, F1 Macro: {f1_macro:.4f}, Test Loss: {val_loss:.4f}")
                scheduler.step(val_loss)
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)
                epoch_test_f1_macros.append(f1_macro)

            samples_used_per_epoch.append(samples_used)

        end_time = time.time()
        log_memory(start_time, end_time)
        print(num_step)

        # Visualization
        plot_accuracy_time_multi(
            model_name=self.model_name,
            accuracy=epoch_accuracies,
            time_per_epoch=time_per_epoch,
            save_path=save_path,
            data_file=save_path
        )
        plot_accuracy_time_multi_test(
            model_name=self.model_name,
            accuracy=epoch_test_accuracies,
            time_per_epoch=time_per_epoch,
            samples_per_epoch=samples_used_per_epoch,
            threshold=self.threshold,
            save_path=save_path,
            data_file=save_path
        )
        plot_f1_macro_time(
            model_name=self.model_name,
            f1_scores=epoch_test_f1_macros,
            time_per_epoch=time_per_epoch,
            save_path=save_path,
            data_file=save_path
        )

        return self.model, num_step

    def train_with_warmup_random(self, start_revision, task, warmup_epochs):
        """
        Warmup Scheduled Match Random Dropout (Warmup-SMRD)

        This method extends the Scheduled Match Random Dropout strategy by adding
        warmup epochs where the full dataset is used before applying random dropout.

        Args:
            start_revision: Epoch to stop dropout and use full dataset again
            task: Task type (classification, segmentation, etc.)
            warmup_epochs: Number of initial epochs to train on full dataset
        """

        save_path = self.save_path
        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

        epoch_losses = []
        epoch_accuracies = []
        epoch_test_accuracies = []
        epoch_test_losses = []
        time_per_epoch = []
        samples_used_per_epoch = []

        start_time = time.time()
        num_step = 0

        for epoch in range(self.epochs):
            # Set epoch for DistributedSampler (important for DDP)
            self._set_sampler_epoch(epoch)

            samples_used = 0
            print(f"Epoch [{epoch+1}/{self.epochs}]")

            # Warmup phase: use full dataset
            if epoch < warmup_epochs:
                print(f"Warmup phase: Using full dataset")
                self.model.train()
                epoch_start_time = time.time()
                running_loss = 0.0
                total_correct = 0
                total_samples = 0

                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training (Warmup)")

                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    num_step += len(outputs)
                    samples_used += len(outputs)

                    with torch.no_grad():
                        preds = torch.argmax(outputs, dim=1)
                        total_correct += (preds == labels).sum().item()
                        total_samples += labels.size(0)

                    progress_bar.set_postfix({"Loss": loss.item()})

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time - epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

                # Evaluation
                self.model.eval()
                correct = 0
                total = 0
                test_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(self.test_loader, desc="Evaluating"):
                        inputs = batch[0].to(self.device)
                        labels = batch[1].to(self.device)
                        outputs = self.model(inputs)

                        batch_loss = criterion(outputs, labels)
                        test_loss += batch_loss.item()

                        predictions = torch.argmax(outputs, dim=-1)
                        correct += (predictions == labels).sum().item()
                        total += labels.size(0)

                accuracy = correct / total
                val_loss = test_loss / len(self.test_loader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
                scheduler.step(val_loss)
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)

            # Random dropout phase: match samples selected by difficulty threshold
            elif epoch < start_revision:
                print(f"Random dropout phase: Matching difficulty-based sample count")
                self.model.train()
                epoch_start_time = time.time()
                running_loss = 0.0
                total_correct = 0
                total_samples = 0

                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training (Random Dropout)")

                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    with torch.no_grad():
                        outputs = self.model(inputs)
                        preds = torch.argmax(outputs, dim=1)

                        if self.threshold == 0:
                            mask = preds != labels
                        else:
                            prob = torch.softmax(outputs, dim=1)
                            correct_class = prob[torch.arange(labels.size(0)), labels]
                            mask = correct_class < self.threshold

                        num_to_select = mask.sum().item()

                    # In DDP mode, we must process every batch to avoid desync
                    # Use at least 1 sample to ensure gradient sync happens
                    if num_to_select == 0:
                        num_to_select = 1

                    # Random sampling based on how many passed threshold
                    indices = torch.randperm(inputs.size(0))[:num_to_select]
                    inputs_sampled = inputs[indices]
                    labels_sampled = labels[indices]

                    optimizer.zero_grad()
                    outputs_sampled = self.model(inputs_sampled)
                    loss = criterion(outputs_sampled, labels_sampled)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    num_step += len(outputs_sampled)
                    samples_used += len(outputs_sampled)

                    # Stats on original batch
                    with torch.no_grad():
                        outputs = self.model(inputs)
                        preds = torch.argmax(outputs, dim=1)
                        total_correct += (preds == labels).sum().item()
                        total_samples += labels.size(0)

                    progress_bar.set_postfix({"Loss": loss.item()})

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time - epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

                # Evaluation
                self.model.eval()
                correct = 0
                total = 0
                test_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(self.test_loader, desc="Evaluating"):
                        inputs = batch[0].to(self.device)
                        labels = batch[1].to(self.device)
                        outputs = self.model(inputs)

                        batch_loss = criterion(outputs, labels)
                        test_loss += batch_loss.item()

                        predictions = torch.argmax(outputs, dim=-1)
                        correct += (predictions == labels).sum().item()
                        total += labels.size(0)

                accuracy = correct / total
                val_loss = test_loss / len(self.test_loader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
                scheduler.step(val_loss)
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)

            # Final phase: use full dataset again
            else:
                print(f"Final phase: Using full dataset")
                self.model.train()
                epoch_start_time = time.time()
                running_loss = 0.0
                correct = 0
                total = 0

                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training (Full Dataset)")
                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    num_step += len(outputs)
                    samples_used += len(outputs)

                    with torch.no_grad():
                        preds = torch.argmax(outputs, dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)

                    progress_bar.set_postfix({"Loss": loss.item()})

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = correct / total
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time - epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

                # Evaluation
                self.model.eval()
                test_correct = 0
                test_total = 0
                test_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(self.test_loader, desc="Evaluating"):
                        inputs = batch[0].to(self.device)
                        labels = batch[1].to(self.device)
                        outputs = self.model(inputs)

                        batch_loss = criterion(outputs, labels)
                        test_loss += batch_loss.item()

                        predictions = torch.argmax(outputs, dim=-1)
                        test_correct += (predictions == labels).sum().item()
                        test_total += labels.size(0)

                accuracy = test_correct / test_total
                val_loss = test_loss / len(self.test_loader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
                scheduler.step(val_loss)
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)

            samples_used_per_epoch.append(samples_used)

        end_time = time.time()
        log_memory(start_time, end_time)
        print(f"Total training steps: {num_step}")
        print(f"Effective epochs: {num_step / len(self.train_loader.dataset):.2f}")

        # Visualization
        plot_accuracy_time_multi(
            model_name=self.model_name,
            accuracy=epoch_accuracies,
            time_per_epoch=time_per_epoch,
            save_path=save_path,
            data_file=save_path
        )
        plot_accuracy_time_multi_test(
            model_name=self.model_name,
            accuracy=epoch_test_accuracies,
            time_per_epoch=time_per_epoch,
            samples_per_epoch=samples_used_per_epoch,
            threshold=self.threshold,
            save_path=save_path,
            data_file=save_path
        )

        return self.model, num_step

    def train_with_periodic_full_random(self, start_revision, task, refresh_interval):
        """
        Periodic Full Dataset Scheduled Match Random Dropout (Periodic-SMRD)

        This method extends SMRD by periodically training on the full dataset every N effective epochs,
        then continuing with random dropout. This provides periodic "recalibration" with all data.

        Args:
            start_revision: Epoch to stop dropout and use full dataset again
            task: Task type (classification, segmentation, etc.)
            refresh_interval: Number of effective epochs between full dataset training (e.g., 5.0)
        """

        save_path = self.save_path
        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

        epoch_losses = []
        epoch_accuracies = []
        epoch_test_accuracies = []
        epoch_test_losses = []
        time_per_epoch = []
        samples_used_per_epoch = []

        start_time = time.time()
        num_step = 0
        dataset_size = len(self.train_loader.dataset)
        next_refresh_threshold = refresh_interval * dataset_size  # Next threshold in terms of num_step

        for epoch in range(self.epochs):
            # Set epoch for DistributedSampler (important for DDP)
            self._set_sampler_epoch(epoch)

            samples_used = 0
            print(f"Epoch [{epoch+1}/{self.epochs}]")

            # Calculate current effective epochs
            current_eff_epochs = num_step / dataset_size

            # Check if we should use full dataset this epoch (periodic refresh)
            # We use full dataset if we've crossed a refresh_interval threshold
            use_full_dataset = (num_step >= next_refresh_threshold and epoch < start_revision)

            if use_full_dataset:
                print(f"Periodic refresh: Using full dataset (Effective epochs: {current_eff_epochs:.2f})")
                # Update next refresh threshold
                next_refresh_threshold += refresh_interval * dataset_size

            if epoch < start_revision:
                self.model.train()
                epoch_start_time = time.time()
                running_loss = 0.0
                total_correct = 0
                total_samples = 0

                if use_full_dataset:
                    # Use full dataset for this epoch
                    progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training (Full Dataset - Refresh)")

                    for batch_idx, (inputs, labels) in progress_bar:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)

                        optimizer.zero_grad()
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)

                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                        num_step += len(outputs)
                        samples_used += len(outputs)

                        with torch.no_grad():
                            preds = torch.argmax(outputs, dim=1)
                            total_correct += (preds == labels).sum().item()
                            total_samples += labels.size(0)

                        progress_bar.set_postfix({"Loss": loss.item()})

                else:
                    # Use random dropout (SMRD)
                    progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training (Random Dropout)")

                    for batch_idx, (inputs, labels) in progress_bar:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)

                        with torch.no_grad():
                            outputs = self.model(inputs)
                            preds = torch.argmax(outputs, dim=1)

                            if self.threshold == 0:
                                mask = preds != labels
                            else:
                                prob = torch.softmax(outputs, dim=1)
                                correct_class = prob[torch.arange(labels.size(0)), labels]
                                mask = correct_class < self.threshold

                            num_to_select = mask.sum().item()

                        # Skip batch if no samples pass threshold
                        if num_to_select == 0:
                            continue

                        # Random sampling based on how many passed threshold
                        indices = torch.randperm(inputs.size(0))[:num_to_select]
                        inputs_sampled = inputs[indices]
                        labels_sampled = labels[indices]

                        optimizer.zero_grad()
                        outputs_sampled = self.model(inputs_sampled)
                        loss = criterion(outputs_sampled, labels_sampled)

                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                        num_step += len(outputs_sampled)
                        samples_used += len(outputs_sampled)

                        # Stats on original batch
                        with torch.no_grad():
                            outputs = self.model(inputs)
                            preds = torch.argmax(outputs, dim=1)
                            total_correct += (preds == labels).sum().item()
                            total_samples += labels.size(0)

                        progress_bar.set_postfix({"Loss": loss.item()})

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time - epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Effective Epochs: {num_step/dataset_size:.2f}")

                # Evaluation
                self.model.eval()
                correct = 0
                total = 0
                test_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(self.test_loader, desc="Evaluating"):
                        inputs = batch[0].to(self.device)
                        labels = batch[1].to(self.device)
                        outputs = self.model(inputs)

                        batch_loss = criterion(outputs, labels)
                        test_loss += batch_loss.item()

                        predictions = torch.argmax(outputs, dim=-1)
                        correct += (predictions == labels).sum().item()
                        total += labels.size(0)

                accuracy = correct / total
                val_loss = test_loss / len(self.test_loader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
                scheduler.step(val_loss)
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)

            else:
                # Final phase: use full dataset
                print(f"Final phase: Using full dataset")
                self.model.train()
                epoch_start_time = time.time()
                running_loss = 0.0
                correct = 0
                total = 0

                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training (Full Dataset)")
                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    num_step += len(outputs)
                    samples_used += len(outputs)

                    with torch.no_grad():
                        preds = torch.argmax(outputs, dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)

                    progress_bar.set_postfix({"Loss": loss.item()})

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = correct / total
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time - epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

                # Evaluation
                self.model.eval()
                test_correct = 0
                test_total = 0
                test_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(self.test_loader, desc="Evaluating"):
                        inputs = batch[0].to(self.device)
                        labels = batch[1].to(self.device)
                        outputs = self.model(inputs)

                        batch_loss = criterion(outputs, labels)
                        test_loss += batch_loss.item()

                        predictions = torch.argmax(outputs, dim=-1)
                        test_correct += (predictions == labels).sum().item()
                        test_total += labels.size(0)

                accuracy = test_correct / test_total
                val_loss = test_loss / len(self.test_loader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
                scheduler.step(val_loss)
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)

            samples_used_per_epoch.append(samples_used)

        end_time = time.time()
        log_memory(start_time, end_time)
        print(f"Total training steps: {num_step}")
        print(f"Effective epochs: {num_step / dataset_size:.2f}")

        # Visualization
        plot_accuracy_time_multi(
            model_name=self.model_name,
            accuracy=epoch_accuracies,
            time_per_epoch=time_per_epoch,
            save_path=save_path,
            data_file=save_path
        )
        plot_accuracy_time_multi_test(
            model_name=self.model_name,
            accuracy=epoch_test_accuracies,
            time_per_epoch=time_per_epoch,
            samples_per_epoch=samples_used_per_epoch,
            threshold=self.threshold,
            save_path=save_path,
            data_file=save_path
        )

        return self.model, num_step

    def train_with_revision_3d(self, start_revision, task):

        save_path = self.save_path
        self.model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        #as per implementation LR=0.045, they use 16 GPU. https://discuss.pytorch.org/t/training-mobilenet-on-imagenet/174391/6 from this blog
        #we use the idea to divide the learning rate by the number of GPUs. 
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
        epoch_losses = []
        epoch_accuracies = []
        epoch_test_accuracies = []
        epoch_test_losses = []
        time_per_epoch = []
        start_time = time.time()
        num_step = 0
        samples_used_per_epoch = []
        for epoch in range(self.epochs):
            # Set epoch for DistributedSampler (important for DDP)
            self._set_sampler_epoch(epoch)

            samples_used = 0
            if epoch < start_revision : 
                self.model.train()
                epoch_start_time = time.time()
                running_loss = 0.0
                correct = 0
                total_correct = 0
                total_samples = 0
                total = 0
                print(f"Epoch [{epoch+1}/{self.epochs}]")
                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")
                
                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device).float(), labels.to(self.device).long().view(-1)
                    
                    with torch.no_grad():
                        outputs = self.model(inputs)
                        preds = torch.argmax(outputs, dim=1)
                        
                        if self.threshold == 0:
                            mask = preds != labels
                        else:
                            prob = torch.softmax(outputs, dim=1)
                            correct_class = prob[torch.arange(labels.size(0)), labels]
                            mask = correct_class < self.threshold

                    # In DDP mode, we must process every batch to avoid desync
                    # If no samples pass threshold, use at least 1 sample
                    if not mask.any():
                        mask[0] = True

                    inputs_misclassified = inputs[mask]
                    labels_misclassified = labels[mask]

                    optimizer.zero_grad()

                    outputs_misclassified = self.model(inputs_misclassified)
                    loss = criterion(outputs_misclassified, labels_misclassified)
                    num_step+=len(outputs_misclassified)
                    samples_used+=len(outputs_misclassified)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    total_correct += (preds == labels).sum().item()
                    total_samples += labels.size(0)
                    progress_bar.set_postfix({"Loss": loss.item()})

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = total_correct/total_samples if total_samples > 0 else 0 
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time-epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

                self.model.eval()
                correct = 0
                total = 0
                test_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(self.test_loader, desc="Evaluating"):
                        inputs = batch[0].to(self.device).float()
                        labels = batch[1].to(self.device).long().view(-1)
                        outputs = self.model(inputs)

                        batch_loss = criterion(outputs, labels)
                        test_loss+=batch_loss.item()

                        predictions = torch.argmax(outputs, dim=-1)
                        correct += (predictions == labels).sum().item()
                        total += labels.size(0)

                accuracy = correct / total
                val_loss = test_loss / len(self.test_loader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
                scheduler.step(val_loss)
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)

            else:
                self.model.train()
                running_loss = 0.0
                correct = 0
                total = 0

                print(f"Epoch [{epoch+1}/{self.epochs}]")
                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")


                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device).float(), labels.to(self.device).long().view(-1)

                    optimizer.zero_grad()

                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    num_step+=len(outputs)
                    samples_used+=len(outputs)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    
                    outputs = self.model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = correct / total
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time - epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

                self.model.eval()
                test_correct = 0
                test_total = 0
                test_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(self.test_loader, desc="Evaluating"):
                        inputs = batch[0].to(self.device).float()
                        labels = batch[1].to(self.device).long().view(-1)
                        outputs = self.model(inputs)

                        batch_loss = criterion(outputs, labels)
                        test_loss+=batch_loss.item()

                        predictions = torch.argmax(outputs, dim=-1)
                        test_correct += (predictions == labels).sum().item()
                        test_total += labels.size(0)

                accuracy = test_correct / test_total
                val_loss = test_loss / len(self.test_loader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
                scheduler.step(val_loss)
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)
            
            samples_used_per_epoch.append(samples_used)


        end_time = time.time()
        log_memory(start_time, end_time)
        print(num_step)




        # plot_metrics(epoch_losses, epoch_accuracies, "Revision")
        # plot_metrics_test(epoch_test_accuracies, "Revision Test")
        plot_accuracy_time_multi(
        model_name=self.model_name,  
        accuracy=epoch_accuracies,
        time_per_epoch=time_per_epoch,  
        save_path=save_path,
        data_file=save_path
        )
        plot_accuracy_time_multi_test(
            model_name = self.model_name,
            accuracy=epoch_test_accuracies,
            time_per_epoch=time_per_epoch,
            samples_per_epoch=samples_used_per_epoch,
            threshold=self.threshold,
            save_path=save_path,
            data_file=save_path
        )

        return self.model, num_step
    

    def train_with_percentage(self, start_revision):
        save_path = self.save_path
        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

        epoch_losses = []
        epoch_accuracies = []
        epoch_test_accuracies = []
        epoch_test_losses = []
        time_per_epoch = []
        samples_used_per_epoch = []
        num_step = 0
        start_time = time.time()

        for epoch in range(self.epochs):
            samples_used = 0
            self.model.train()
            epoch_start_time = time.time()
            running_loss = 0.0
            correct = 0
            total = 0
            total_correct = 0
            total_samples = 0

            print(f"Epoch [{epoch+1}/{self.epochs}]")
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")

            if epoch < start_revision:
                decay_factor = 0.99 ** epoch  ##percentage to be sampled
                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    batch_size = inputs.size(0)
                    selected_count = int(decay_factor * batch_size)

                    if selected_count == 0:
                        continue

                    selected_indices = torch.randperm(batch_size)[:selected_count]
                    inputs_selected = inputs[selected_indices]
                    labels_selected = labels[selected_indices]

                    optimizer.zero_grad()
                    outputs = self.model(inputs_selected)
                    loss = criterion(outputs, labels_selected)
                    num_step += selected_count
                    samples_used += selected_count
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    with torch.no_grad():
                        preds = torch.argmax(self.model(inputs), dim=1)
                        total_correct += (preds == labels).sum().item()
                        total_samples += labels.size(0)

                    progress_bar.set_postfix({"Loss": loss.item()})
            else:
                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    num_step += inputs.size(0)
                    samples_used += inputs.size(0)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    with torch.no_grad():
                        preds = torch.argmax(outputs, dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)

            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = (
                total_correct / total_samples if epoch < start_revision else correct / total
            )
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)

            epoch_end_time = time.time()
            time_per_epoch.append(epoch_end_time - epoch_start_time)

            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

            self.model.eval()
            test_correct = 0
            test_total = 0
            test_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(self.test_loader, desc="Evaluating"):
                    inputs = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                    outputs = self.model(inputs)
                    batch_loss = criterion(outputs, labels)
                    test_loss += batch_loss.item()
                    predictions = torch.argmax(outputs, dim=-1)
                    test_correct += (predictions == labels).sum().item()
                    test_total += labels.size(0)

            accuracy = test_correct / test_total
            val_loss = test_loss / len(self.test_loader)
            print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
            scheduler.step(val_loss)
            epoch_test_accuracies.append(accuracy)
            epoch_test_losses.append(val_loss)
            samples_used_per_epoch.append(samples_used)

        end_time = time.time()
        log_memory(start_time, end_time)
        print(num_step)

        plot_accuracy_time_multi(
            model_name=self.model_name,
            accuracy=epoch_accuracies,
            time_per_epoch=time_per_epoch,
            save_path=save_path,
            data_file=save_path,
        )
        plot_accuracy_time_multi_test(
            model_name=self.model_name,
            accuracy=epoch_test_accuracies,
            time_per_epoch=time_per_epoch,
            samples_per_epoch=samples_used_per_epoch,
            threshold=self.threshold,
            save_path=save_path,
            data_file=save_path,
        )

        return self.model, num_step

    def inverse_linear(self, epoch, alpha):
        if epoch == 200:
            return 50000
        x = np.arange(1, 200)
        y = 1 / (x + alpha)
        y_scaled = (y / np.max(y)) * 50000
        return y_scaled[epoch - 1]

    def train_with_inverse_linear(self, start_revision, data_size):
        save_path = self.save_path
        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

        epoch_losses = []
        epoch_accuracies = []
        epoch_test_accuracies = []
        epoch_test_losses = []
        time_per_epoch = []
        samples_used_per_epoch = []
        num_step = 0
        alpha=2
        start_time = time.time()

        for epoch in range(self.epochs):
            samples_used = 0
            self.model.train()
            epoch_start_time = time.time()
            running_loss = 0.0
            correct = 0
            total = 0
            total_correct = 0
            total_samples = 0

            print(f"Epoch [{epoch+1}/{self.epochs}]")
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")

            if epoch < start_revision:
                # Use inverse linear decay
                scaled_value = self.inverse_linear(epoch + 1, alpha)  # epoch+1 to match 1-based indexing
                sample_ratio = scaled_value / data_size

                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    batch_size = inputs.size(0)
                    selected_count = int(sample_ratio * batch_size)

                    if selected_count == 0:
                        continue

                    selected_indices = torch.randperm(batch_size)[:selected_count]
                    inputs_selected = inputs[selected_indices]
                    labels_selected = labels[selected_indices]

                    optimizer.zero_grad()
                    outputs = self.model(inputs_selected)
                    loss = criterion(outputs, labels_selected)
                    num_step += selected_count
                    samples_used += selected_count
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    with torch.no_grad():
                        preds = torch.argmax(self.model(inputs), dim=1)
                        total_correct += (preds == labels).sum().item()
                        total_samples += labels.size(0)

                    progress_bar.set_postfix({"Loss": loss.item()})
            else:
                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    num_step += inputs.size(0)
                    samples_used += inputs.size(0)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    with torch.no_grad():
                        preds = torch.argmax(outputs, dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)

            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = (
                total_correct / total_samples if epoch < start_revision else correct / total
            )
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)

            epoch_end_time = time.time()
            time_per_epoch.append(epoch_end_time - epoch_start_time)

            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

            self.model.eval()
            test_correct = 0
            test_total = 0
            test_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(self.test_loader, desc="Evaluating"):
                    inputs = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                    outputs = self.model(inputs)
                    batch_loss = criterion(outputs, labels)
                    test_loss += batch_loss.item()
                    predictions = torch.argmax(outputs, dim=-1)
                    test_correct += (predictions == labels).sum().item()
                    test_total += labels.size(0)

            accuracy = test_correct / test_total
            val_loss = test_loss / len(self.test_loader)
            print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
            scheduler.step(val_loss)
            epoch_test_accuracies.append(accuracy)
            epoch_test_losses.append(val_loss)
            samples_used_per_epoch.append(samples_used)

        end_time = time.time()
        log_memory(start_time, end_time)
        print(num_step)

        plot_accuracy_time_multi(
            model_name=self.model_name,
            accuracy=epoch_accuracies,
            time_per_epoch=time_per_epoch,
            save_path=save_path,
            data_file=save_path,
        )
        plot_accuracy_time_multi_test(
            model_name=self.model_name,
            accuracy=epoch_test_accuracies,
            time_per_epoch=time_per_epoch,
            samples_per_epoch=samples_used_per_epoch,
            threshold=self.threshold,
            save_path=save_path,
            data_file=save_path,
        )

        return self.model, num_step
    
    def log_schedule(self, epoch, data_size, alpha):
        x = np.arange(1, 200)
        y = 1 / np.log(x + alpha)  # make sure alpha > 1 to avoid log(0)
        y_scaled = (y / np.max(y)) * data_size
        return y_scaled[epoch - 1]

    def train_with_log(self, start_revision, data_size):
        save_path = self.save_path
        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

        epoch_losses = []
        epoch_accuracies = []
        epoch_test_accuracies = []
        epoch_test_losses = []
        time_per_epoch = []
        samples_used_per_epoch = []
        num_step = 0
        alpha=2
        start_time = time.time()

        for epoch in range(self.epochs):
            samples_used = 0
            self.model.train()
            epoch_start_time = time.time()
            running_loss = 0.0
            correct = 0
            total = 0
            total_correct = 0
            total_samples = 0

            print(f"Epoch [{epoch+1}/{self.epochs}]")
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")

            if epoch < start_revision:
                # Use inverse linear decay
                scaled_value = self.log_schedule(epoch + 1, data_size, alpha)  # epoch+1 to match 1-based indexing
                sample_ratio = scaled_value / data_size

                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    batch_size = inputs.size(0)
                    selected_count = int(sample_ratio * batch_size)

                    if selected_count == 0:
                        continue

                    selected_indices = torch.randperm(batch_size)[:selected_count]
                    inputs_selected = inputs[selected_indices]
                    labels_selected = labels[selected_indices]

                    optimizer.zero_grad()
                    outputs = self.model(inputs_selected)
                    loss = criterion(outputs, labels_selected)
                    num_step += selected_count
                    samples_used += selected_count
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    with torch.no_grad():
                        preds = torch.argmax(self.model(inputs), dim=1)
                        total_correct += (preds == labels).sum().item()
                        total_samples += labels.size(0)

                    progress_bar.set_postfix({"Loss": loss.item()})
            else:
                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    num_step += inputs.size(0)
                    samples_used += inputs.size(0)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    with torch.no_grad():
                        preds = torch.argmax(outputs, dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)

            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = (
                total_correct / total_samples if epoch < start_revision else correct / total
            )
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)

            epoch_end_time = time.time()
            time_per_epoch.append(epoch_end_time - epoch_start_time)

            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

            self.model.eval()
            test_correct = 0
            test_total = 0
            test_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(self.test_loader, desc="Evaluating"):
                    inputs = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                    outputs = self.model(inputs)
                    batch_loss = criterion(outputs, labels)
                    test_loss += batch_loss.item()
                    predictions = torch.argmax(outputs, dim=-1)
                    test_correct += (predictions == labels).sum().item()
                    test_total += labels.size(0)

            accuracy = test_correct / test_total
            val_loss = test_loss / len(self.test_loader)
            print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
            scheduler.step(val_loss)
            epoch_test_accuracies.append(accuracy)
            epoch_test_losses.append(val_loss)
            samples_used_per_epoch.append(samples_used)

        end_time = time.time()
        log_memory(start_time, end_time)
        print(num_step)

        plot_accuracy_time_multi(
            model_name=self.model_name,
            accuracy=epoch_accuracies,
            time_per_epoch=time_per_epoch,
            save_path=save_path,
            data_file=save_path,
        )
        plot_accuracy_time_multi_test(
            model_name=self.model_name,
            accuracy=epoch_test_accuracies,
            time_per_epoch=time_per_epoch,
            samples_per_epoch=samples_used_per_epoch,
            threshold=self.threshold,
            save_path=save_path,
            data_file=save_path,
        )

        return self.model, num_step 
    
    def train_with_adaptive(self, start_revision, task, cls_num_list, interval, increment):

        save_path = self.save_path
        self.model.to(self.device)
        if task=='classification':
            criterion = nn.CrossEntropyLoss()
        elif 'longtail':
            train_sampler = None
            idx = self.epochs // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(self.device)
            criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda(self.device)
        # optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        #as per implementation LR=0.045, they use 16 GPU. https://discuss.pytorch.org/t/training-mobilenet-on-imagenet/174391/6 from this blog
        #we use the idea to divide the learning rate by the number of GPUs. 
        # optimizer = optim.RMSprop(self.model.parameters(), weight_decay=0.00004, momentum=0.9, lr=0.0028125)   
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
        epoch_losses = []
        epoch_accuracies = []
        epoch_test_accuracies = []
        epoch_test_losses = []
        time_per_epoch = []
        start_time = time.time()
        num_step = 0
        samples_used_per_epoch = []
        init_threshold = self.threshold
        for epoch in range(self.epochs):
            samples_used = 0
            
            self.threshold = self.threshold + (epoch // interval)*increment
            if self.threshold>=1.0:
                self.threshold = init_threshold
            if epoch < start_revision : 
                self.model.train()
                epoch_start_time = time.time()
                running_loss = 0.0
                correct = 0
                total_correct = 0
                total_samples = 0
                total = 0
                print(f"Epoch [{epoch+1}/{self.epochs}]")
                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")
                
                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(inputs)
                        preds = torch.argmax(outputs, dim=1)
                        
                        if self.threshold == 0:
                            mask = preds != labels
                        else:
                            prob = torch.softmax(outputs, dim=1)
                            correct_class = prob[torch.arange(labels.size(0)), labels]
                            mask = correct_class < self.threshold

                    # In DDP mode, we must process every batch to avoid desync
                    # If no samples pass threshold, use at least 1 sample
                    if not mask.any():
                        mask[0] = True

                    inputs_misclassified = inputs[mask]
                    labels_misclassified = labels[mask]

                    optimizer.zero_grad()

                    outputs_misclassified = self.model(inputs_misclassified)
                    loss = criterion(outputs_misclassified, labels_misclassified)
                    num_step+=len(outputs_misclassified)
                    samples_used+=len(outputs_misclassified)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    total_correct += (preds == labels).sum().item()
                    total_samples += labels.size(0)
                    progress_bar.set_postfix({"Loss": loss.item()})

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = total_correct/total_samples if total_samples > 0 else 0 
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time-epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

                self.model.eval()
                # Keep on GPU to avoid sync
                correct = torch.tensor(0, device=self.device)
                total = 0
                test_loss = torch.tensor(0.0, device=self.device)
                with torch.no_grad():
                    for batch in tqdm(self.test_loader, desc="Evaluating"):
                        inputs = batch[0].to(self.device)
                        labels = batch[1].to(self.device)
                        outputs = self.model(inputs)

                        batch_loss = criterion(outputs, labels)
                        test_loss += batch_loss.detach() * labels.size(0)

                        predictions = torch.argmax(outputs, dim=-1)
                        correct += (predictions == labels).sum()
                        total += labels.size(0)

                # Only sync at end
                accuracy = (correct / total).item()
                val_loss = (test_loss / total).item()
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
                scheduler.step(val_loss)
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)

            else:
                self.model.train()
                running_loss = 0.0
                correct = 0
                total = 0

                print(f"Epoch [{epoch+1}/{self.epochs}]")
                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")


                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()

                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    num_step+=len(outputs)
                    samples_used+=len(outputs)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    
                    outputs = self.model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = correct / total
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time - epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

                self.model.eval()
                test_correct = 0
                test_total = 0
                test_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(self.test_loader, desc="Evaluating"):
                        inputs = batch[0].to(self.device)
                        labels = batch[1].to(self.device)
                        outputs = self.model(inputs)

                        batch_loss = criterion(outputs, labels)
                        test_loss+=batch_loss.item()

                        predictions = torch.argmax(outputs, dim=-1)
                        test_correct += (predictions == labels).sum().item()
                        test_total += labels.size(0)

                accuracy = test_correct / test_total
                val_loss = test_loss / len(self.test_loader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
                scheduler.step(val_loss)
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)
            
            samples_used_per_epoch.append(samples_used)


        end_time = time.time()
        log_memory(start_time, end_time)
        print(num_step)

        total_wall_time = end_time - start_time
        print(f"\n✅ Total Wall Time for {self.epochs} epochs: {total_wall_time:.2f} seconds "
            f"({total_wall_time / 60:.2f} minutes)")


        plot_accuracy_time_multi(
        model_name=self.model_name,  
        accuracy=epoch_accuracies,
        time_per_epoch=time_per_epoch,  
        save_path=save_path,
        data_file=save_path
        )
        plot_accuracy_time_multi_test(
            model_name = self.model_name,
            accuracy=epoch_test_accuracies,
            time_per_epoch=time_per_epoch,
            samples_per_epoch=samples_used_per_epoch,
            threshold=self.threshold,
            save_path=save_path,
            data_file=save_path
        )

        return self.model, num_step
    
    def train_with_noisy_revision(self, start_revision, task, cls_num_list):

        save_path = self.save_path
        self.model.to(self.device)
        if task=='classification':
            criterion = nn.CrossEntropyLoss()
        elif 'longtail':
            train_sampler = None
            idx = self.epochs // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(self.device)
            criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda(self.device)
        # optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        #as per implementation LR=0.045, they use 16 GPU. https://discuss.pytorch.org/t/training-mobilenet-on-imagenet/174391/6 from this blog
        #we use the idea to divide the learning rate by the number of GPUs. 
        # optimizer = optim.RMSprop(self.model.parameters(), weight_decay=0.00004, momentum=0.9, lr=0.0028125)   
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
        epoch_losses = []
        epoch_accuracies = []
        epoch_test_accuracies = []
        epoch_test_losses = []
        time_per_epoch = []
        start_time = time.time()
        num_step = 0
        samples_used_per_epoch = []
        for epoch in range(self.epochs):
            # Set epoch for DistributedSampler (important for DDP)
            self._set_sampler_epoch(epoch)

            samples_used = 0
            if epoch < start_revision : 
                self.model.train()
                epoch_start_time = time.time()
                running_loss = 0.0
                correct = 0
                total_correct = 0
                total_samples = 0
                total = 0
                print(f"Epoch [{epoch+1}/{self.epochs}]")
                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")
                
                for batch_idx, (inputs, labels, _) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(inputs)
                        preds = torch.argmax(outputs, dim=1)
                        
                        if self.threshold == 0:
                            mask = preds != labels
                        else:
                            prob = torch.softmax(outputs, dim=1)
                            correct_class = prob[torch.arange(labels.size(0)), labels]
                            mask = correct_class < self.threshold

                    # In DDP mode, we must process every batch to avoid desync
                    # If no samples pass threshold, use at least 1 sample
                    if not mask.any():
                        mask[0] = True

                    inputs_misclassified = inputs[mask]
                    labels_misclassified = labels[mask]

                    optimizer.zero_grad()

                    outputs_misclassified = self.model(inputs_misclassified)
                    loss = criterion(outputs_misclassified, labels_misclassified)
                    num_step+=len(outputs_misclassified)
                    samples_used+=len(outputs_misclassified)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    total_correct += (preds == labels).sum().item()
                    total_samples += labels.size(0)
                    progress_bar.set_postfix({"Loss": loss.item()})

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = total_correct/total_samples if total_samples > 0 else 0 
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time-epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

                self.model.eval()
                # Keep on GPU to avoid sync
                correct = torch.tensor(0, device=self.device)
                total = 0
                test_loss = torch.tensor(0.0, device=self.device)
                with torch.no_grad():
                    for batch in tqdm(self.test_loader, desc="Evaluating"):
                        inputs = batch[0].to(self.device)
                        labels = batch[1].to(self.device)
                        outputs = self.model(inputs)

                        batch_loss = criterion(outputs, labels)
                        test_loss += batch_loss.detach() * labels.size(0)

                        predictions = torch.argmax(outputs, dim=-1)
                        correct += (predictions == labels).sum()
                        total += labels.size(0)

                # Only sync at end
                accuracy = (correct / total).item()
                val_loss = (test_loss / total).item()
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
                scheduler.step(val_loss)
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)

            else:
                self.model.train()
                running_loss = 0.0
                correct = 0
                total = 0

                print(f"Epoch [{epoch+1}/{self.epochs}]")
                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")


                for batch_idx, (inputs, labels, _) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()

                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    num_step+=len(outputs)
                    samples_used+=len(outputs)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    
                    outputs = self.model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = correct / total
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time - epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

                self.model.eval()
                test_correct = 0
                test_total = 0
                test_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(self.test_loader, desc="Evaluating"):
                        inputs = batch[0].to(self.device)
                        labels = batch[1].to(self.device)
                        outputs = self.model(inputs)

                        batch_loss = criterion(outputs, labels)
                        test_loss+=batch_loss.item()

                        predictions = torch.argmax(outputs, dim=-1)
                        test_correct += (predictions == labels).sum().item()
                        test_total += labels.size(0)

                accuracy = test_correct / test_total
                val_loss = test_loss / len(self.test_loader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
                scheduler.step(val_loss)
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)
            
            samples_used_per_epoch.append(samples_used)


        end_time = time.time()
        log_memory(start_time, end_time)
        print(num_step)

        total_wall_time = end_time - start_time
        print(f"\n Total Wall Time for {self.epochs} epochs: {total_wall_time:.2f} seconds "
            f"({total_wall_time / 60:.2f} minutes)")


        plot_accuracy_time_multi(
        model_name=self.model_name,  
        accuracy=epoch_accuracies,
        time_per_epoch=time_per_epoch,  
        save_path=save_path,
        data_file=save_path
        )
        plot_accuracy_time_multi_test(
            model_name = self.model_name,
            accuracy=epoch_test_accuracies,
            time_per_epoch=time_per_epoch,
            samples_per_epoch=samples_used_per_epoch,
            threshold=self.threshold,
            save_path=save_path,
            data_file=save_path
        )

        return self.model, num_step
    
    def train_with_alternative(self, start_revision, task, cls_num_list):

        save_path = self.save_path
        self.model.to(self.device)
        if task=='classification':
            criterion = nn.CrossEntropyLoss()
        elif 'longtail':
            train_sampler = None
            idx = self.epochs // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(self.device)
            criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda(self.device)
        # optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        #as per implementation LR=0.045, they use 16 GPU. https://discuss.pytorch.org/t/training-mobilenet-on-imagenet/174391/6 from this blog
        #we use the idea to divide the learning rate by the number of GPUs. 
        # optimizer = optim.RMSprop(self.model.parameters(), weight_decay=0.00004, momentum=0.9, lr=0.0028125)   
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
        epoch_losses = []
        epoch_accuracies = []
        epoch_test_accuracies = []
        epoch_test_losses = []
        time_per_epoch = []
        start_time = time.time()
        num_step = 0
        samples_used_per_epoch = []
        for epoch in range(self.epochs):
            # Set epoch for DistributedSampler (important for DDP)
            self._set_sampler_epoch(epoch)

            samples_used = 0
            if epoch < start_revision : 
                self.model.train()
                epoch_start_time = time.time()
                running_loss = 0.0
                correct = 0
                total_correct = 0
                total_samples = 0
                total = 0
                print(f"Epoch [{epoch+1}/{self.epochs}]")
                if epoch % 2 == 0:
                    misclassified_indices = []

                    for batch_idx, (inputs, labels) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                        inputs, labels = inputs.to(self.device), labels.to(self.device)

                        with torch.no_grad():
                            outputs = self.model(inputs)
                            preds = torch.argmax(outputs, dim=1)

                            if self.threshold == 0:
                                mask = preds != labels
                            else:
                                prob = torch.softmax(outputs, dim=1)
                                correct_class = prob[torch.arange(labels.size(0)), labels]
                                mask = correct_class < self.threshold

                        if mask.any():
                            base_idx = batch_idx * self.train_loader.batch_size
                            selected = mask.nonzero(as_tuple=True)[0] + base_idx
                            misclassified_indices.extend(selected.tolist())

                    cached_misclassified_indices = misclassified_indices

                # Use cached indices for training
                # In DDP mode, ensure all ranks process the same number of epochs
                if not cached_misclassified_indices:
                    print("No misclassified samples. Using first sample as fallback...")
                    cached_misclassified_indices = [0]

                subset = torch.utils.data.Subset(self.train_loader.dataset, cached_misclassified_indices)
                misclassified_loader = torch.utils.data.DataLoader(
                    subset, batch_size=self.train_loader.batch_size, shuffle=True, num_workers=2
                )

                for batch_idx, (inputs, labels) in tqdm(enumerate(misclassified_loader), total=len(misclassified_loader)):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    num_step += len(outputs)
                    samples_used += len(outputs)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    total_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
                    total_samples += labels.size(0)

                epoch_loss = running_loss / len(misclassified_loader)
                epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time - epoch_start_time)

                print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")


                self.model.eval()
                # Keep on GPU to avoid sync
                correct = torch.tensor(0, device=self.device)
                total = 0
                test_loss = torch.tensor(0.0, device=self.device)
                with torch.no_grad():
                    for batch in tqdm(self.test_loader, desc="Evaluating"):
                        inputs = batch[0].to(self.device)
                        labels = batch[1].to(self.device)
                        outputs = self.model(inputs)

                        batch_loss = criterion(outputs, labels)
                        test_loss += batch_loss.detach() * labels.size(0)

                        predictions = torch.argmax(outputs, dim=-1)
                        correct += (predictions == labels).sum()
                        total += labels.size(0)

                # Only sync at end
                accuracy = (correct / total).item()
                val_loss = (test_loss / total).item()
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
                scheduler.step(val_loss)
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)

            else:
                self.model.train()
                running_loss = 0.0
                correct = 0
                total = 0

                print(f"Epoch [{epoch+1}/{self.epochs}]")
                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")


                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()

                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    num_step+=len(outputs)
                    samples_used+=len(outputs)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    
                    outputs = self.model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = correct / total
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time - epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

                self.model.eval()
                test_correct = 0
                test_total = 0
                test_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(self.test_loader, desc="Evaluating"):
                        inputs = batch[0].to(self.device)
                        labels = batch[1].to(self.device)
                        outputs = self.model(inputs)

                        batch_loss = criterion(outputs, labels)
                        test_loss+=batch_loss.item()

                        predictions = torch.argmax(outputs, dim=-1)
                        test_correct += (predictions == labels).sum().item()
                        test_total += labels.size(0)

                accuracy = test_correct / test_total
                val_loss = test_loss / len(self.test_loader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
                scheduler.step(val_loss)
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)
            
            samples_used_per_epoch.append(samples_used)


        end_time = time.time()
        log_memory(start_time, end_time)
        print(num_step)

        total_wall_time = end_time - start_time
        print(f"\n Total Wall Time for {self.epochs} epochs: {total_wall_time:.2f} seconds "
            f"({total_wall_time / 60:.2f} minutes)")


        plot_accuracy_time_multi(
        model_name=self.model_name,  
        accuracy=epoch_accuracies,
        time_per_epoch=time_per_epoch,  
        save_path=save_path,
        data_file=save_path
        )
        plot_accuracy_time_multi_test(
            model_name = self.model_name,
            accuracy=epoch_test_accuracies,
            time_per_epoch=time_per_epoch,
            samples_per_epoch=samples_used_per_epoch,
            threshold=self.threshold,
            save_path=save_path,
            data_file=save_path
        )

        return self.model, num_step

    def train_with_noisy_random(self, start_revision, task):

        save_path = self.save_path
        self.model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

        epoch_losses = []
        epoch_accuracies = []
        epoch_test_accuracies = []
        epoch_test_losses = []
        time_per_epoch = []
        samples_used_per_epoch = []

        start_time = time.time()
        num_step = 0

        for epoch in range(self.epochs):
            # Set epoch for DistributedSampler (important for DDP)
            self._set_sampler_epoch(epoch)

            samples_used = 0
            print(f"Epoch [{epoch+1}/{self.epochs}]")

            if epoch < start_revision:
                self.model.train()
                epoch_start_time = time.time()
                running_loss = 0.0
                total_correct = 0
                total_samples = 0

                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")

                for batch_idx, (inputs, labels, _) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    with torch.no_grad():
                        outputs = self.model(inputs)
                        preds = torch.argmax(outputs, dim=1)

                        if self.threshold == 0:
                            mask = preds != labels
                        else:
                            prob = torch.softmax(outputs, dim=1)
                            correct_class = prob[torch.arange(labels.size(0)), labels]
                            mask = correct_class < self.threshold

                        num_to_select = mask.sum().item()

                    # In DDP mode, we must process every batch to avoid desync
                    # Use at least 1 sample to ensure gradient sync happens
                    if num_to_select == 0:
                        num_to_select = 1

                    # 🔁 Random sampling based on how many passed threshold
                    indices = torch.randperm(inputs.size(0))[:num_to_select]
                    inputs_sampled = inputs[indices]
                    labels_sampled = labels[indices]

                    optimizer.zero_grad()
                    outputs_sampled = self.model(inputs_sampled)
                    loss = criterion(outputs_sampled, labels_sampled)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    num_step += len(outputs_sampled)
                    samples_used += len(outputs_sampled)

                    # Stats on original batch
                    with torch.no_grad():
                        outputs = self.model(inputs)
                        preds = torch.argmax(outputs, dim=1)
                        total_correct += (preds == labels).sum().item()
                        total_samples += labels.size(0)

                    progress_bar.set_postfix({"Loss": loss.item()})

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time - epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

                # Evaluation
                self.model.eval()
                correct = 0
                total = 0
                test_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(self.test_loader, desc="Evaluating"):
                        inputs = batch[0].to(self.device)
                        labels = batch[1].to(self.device)
                        outputs = self.model(inputs)

                        batch_loss = criterion(outputs, labels)
                        test_loss += batch_loss.item()

                        predictions = torch.argmax(outputs, dim=-1)
                        correct += (predictions == labels).sum().item()
                        total += labels.size(0)

                accuracy = correct / total
                val_loss = test_loss / len(self.test_loader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
                scheduler.step(val_loss)
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)

            else:
                self.model.train()
                epoch_start_time = time.time()
                running_loss = 0.0
                correct = 0
                total = 0

                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")
                for batch_idx, (inputs, labels, _) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    num_step += len(outputs)
                    samples_used += len(outputs)

                    with torch.no_grad():
                        preds = torch.argmax(outputs, dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)

                    progress_bar.set_postfix({"Loss": loss.item()})

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = correct / total
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time - epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

                # Evaluation
                self.model.eval()
                test_correct = 0
                test_total = 0
                test_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(self.test_loader, desc="Evaluating"):
                        inputs = batch[0].to(self.device)
                        labels = batch[1].to(self.device)
                        outputs = self.model(inputs)

                        batch_loss = criterion(outputs, labels)
                        test_loss += batch_loss.item()

                        predictions = torch.argmax(outputs, dim=-1)
                        test_correct += (predictions == labels).sum().item()
                        test_total += labels.size(0)

                accuracy = test_correct / test_total
                val_loss = test_loss / len(self.test_loader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
                scheduler.step(val_loss)
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)

            samples_used_per_epoch.append(samples_used)

        end_time = time.time()
        log_memory(start_time, end_time)
        print(num_step)

        # Visualization
        plot_accuracy_time_multi(
            model_name=self.model_name,  
            accuracy=epoch_accuracies,
            time_per_epoch=time_per_epoch,  
            save_path=save_path,
            data_file=save_path
        )
        plot_accuracy_time_multi_test(
            model_name=self.model_name,
            accuracy=epoch_test_accuracies,
            time_per_epoch=time_per_epoch,
            samples_per_epoch=samples_used_per_epoch,
            threshold=self.threshold,
            save_path=save_path,
            data_file=save_path
        )

        return self.model, num_step
    
    def train_with_noisy_percentage(self, start_revision):
        save_path = self.save_path
        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

        epoch_losses = []
        epoch_accuracies = []
        epoch_test_accuracies = []
        epoch_test_losses = []
        time_per_epoch = []
        samples_used_per_epoch = []
        num_step = 0
        start_time = time.time()

        for epoch in range(self.epochs):
            samples_used = 0
            self.model.train()
            epoch_start_time = time.time()
            running_loss = 0.0
            correct = 0
            total = 0
            total_correct = 0
            total_samples = 0

            print(f"Epoch [{epoch+1}/{self.epochs}]")
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")

            if epoch < start_revision:
                decay_factor = 0.95 ** epoch  ##percentage to be sampled
                for batch_idx, (inputs, labels, _) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    batch_size = inputs.size(0)
                    selected_count = int(decay_factor * batch_size)

                    if selected_count == 0:
                        continue

                    selected_indices = torch.randperm(batch_size)[:selected_count]
                    inputs_selected = inputs[selected_indices]
                    labels_selected = labels[selected_indices]

                    optimizer.zero_grad()
                    outputs = self.model(inputs_selected)
                    loss = criterion(outputs, labels_selected)
                    num_step += selected_count
                    samples_used += selected_count
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    with torch.no_grad():
                        preds = torch.argmax(self.model(inputs), dim=1)
                        total_correct += (preds == labels).sum().item()
                        total_samples += labels.size(0)

                    progress_bar.set_postfix({"Loss": loss.item()})
            else:
                for batch_idx, (inputs, labels, _) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    num_step += inputs.size(0)
                    samples_used += inputs.size(0)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    with torch.no_grad():
                        preds = torch.argmax(outputs, dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)

            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = (
                total_correct / total_samples if epoch < start_revision else correct / total
            )
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)

            epoch_end_time = time.time()
            time_per_epoch.append(epoch_end_time - epoch_start_time)

            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

            self.model.eval()
            test_correct = 0
            test_total = 0
            test_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(self.test_loader, desc="Evaluating"):
                    inputs = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                    outputs = self.model(inputs)
                    batch_loss = criterion(outputs, labels)
                    test_loss += batch_loss.item()
                    predictions = torch.argmax(outputs, dim=-1)
                    test_correct += (predictions == labels).sum().item()
                    test_total += labels.size(0)

            accuracy = test_correct / test_total
            val_loss = test_loss / len(self.test_loader)
            print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
            scheduler.step(val_loss)
            epoch_test_accuracies.append(accuracy)
            epoch_test_losses.append(val_loss)
            samples_used_per_epoch.append(samples_used)

        end_time = time.time()
        log_memory(start_time, end_time)
        print(num_step)

        plot_accuracy_time_multi(
            model_name=self.model_name,
            accuracy=epoch_accuracies,
            time_per_epoch=time_per_epoch,
            save_path=save_path,
            data_file=save_path,
        )
        plot_accuracy_time_multi_test(
            model_name=self.model_name,
            accuracy=epoch_test_accuracies,
            time_per_epoch=time_per_epoch,
            samples_per_epoch=samples_used_per_epoch,
            threshold=self.threshold,
            save_path=save_path,
            data_file=save_path,
        )

        return self.model, num_step
    def train_with_stratified(self, start_revision, task, num_classes, cls_num_list=None):
        """
        Stratified Difficulty-Based Progressive Dropout (SDBPD)

        Similar to SMRD but with intelligent stratified selection:
        - Identifies hard samples in each batch
        - Selects hard samples independently from each class
        - Ensures tail classes get representation even when minority in batch

        Key advantage: Prevents majority class dominance in hard sample selection

        Args:
            start_revision: Epoch to stop dropout and use full dataset
            task: Task type (classification, segmentation, longtail)
            num_classes: Number of classes in the dataset
            cls_num_list: List of sample counts per class (for longtail task)
        """

        save_path = self.save_path
        self.model.to(self.device)

        if task=='classification':
            criterion = nn.CrossEntropyLoss()
        elif 'longtail':
            train_sampler = None
            idx = self.epochs // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(self.device)
            criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

        epoch_losses = []
        epoch_accuracies = []
        epoch_test_accuracies = []
        epoch_test_losses = []
        epoch_test_f1_macros = []  # Track F1 macro scores
        time_per_epoch = []
        samples_used_per_epoch = []

        start_time = time.time()
        num_step = 0

        for epoch in range(self.epochs):
            # Set epoch for DistributedSampler (important for DDP)
            self._set_sampler_epoch(epoch)

            samples_used = 0
            num_hard_samples = 0  # Track number of hard examples detected
            print(f"Epoch [{epoch+1}/{self.epochs}]")

            if epoch < start_revision:
                self.model.train()
                epoch_start_time = time.time()
                running_loss = 0.0
                total_correct = 0
                total_samples = 0

                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training (SDBPD)")

                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # ============================================================
                    # STEP 1: IDENTIFY ALL HARD SAMPLES IN BATCH
                    # ============================================================
                    with torch.no_grad():
                        outputs = self.model(inputs)

                        if self.threshold == 0:
                            # "Hard" = Misclassified
                            hard_mask = (torch.argmax(outputs, dim=1) != labels)
                        else:
                            # "Hard" = Confidence below threshold
                            prob = torch.softmax(outputs, dim=1)
                            correct_class_probs = prob[torch.arange(labels.size(0)), labels]
                            hard_mask = (correct_class_probs < self.threshold)

                        num_hard_samples += hard_mask.sum().item()  # Track hard samples

                    # ============================================================
                    # STEP 2: STRATIFIED SELECTION - Select hard samples per class
                    # ============================================================
                    kept_indices = []
                    unique_classes_in_batch = torch.unique(labels)

                    for class_id in unique_classes_in_batch:
                        # Find samples belonging to current class
                        in_this_class_mask = (labels == class_id)

                        # Find samples that are BOTH in this class AND are hard
                        selected_for_this_class_mask = in_this_class_mask & hard_mask

                        # Get tensor indices of these samples
                        indices_for_this_class = torch.where(selected_for_this_class_mask)[0]

                        # Add these indices to our list of samples to keep
                        kept_indices.append(indices_for_this_class)

                    # Concatenate index tensors from all classes
                    if kept_indices:
                        final_indices = torch.cat(kept_indices)
                    else:
                        # Empty tensor if no hard samples found
                        final_indices = torch.tensor([], dtype=torch.long, device=self.device)

                    # ============================================================
                    # STEP 3: SAFETY CHECK - Ensure at least one sample
                    # ============================================================
                    # In DDP mode, we must process every batch to avoid desync
                    num_to_select = final_indices.size(0)
                    if num_to_select == 0:
                        # Fallback to one random sample to keep training loop alive
                        final_indices = torch.randint(0, inputs.size(0), (1,), device=self.device)

                    # ============================================================
                    # STEP 4: CREATE NEW TRAINING BATCH
                    # ============================================================
                    inputs_sampled = inputs[final_indices]
                    labels_sampled = labels[final_indices]

                    # ============================================================
                    # TRAINING ON STRATIFIED HARD SAMPLES
                    # ============================================================
                    optimizer.zero_grad()
                    outputs_sampled = self.model(inputs_sampled)
                    loss = criterion(outputs_sampled, labels_sampled)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    num_step += len(outputs_sampled)
                    samples_used += len(outputs_sampled)

                    # Stats on original batch (for logging)
                    with torch.no_grad():
                        preds = torch.argmax(outputs, dim=1)
                        total_correct += (preds == labels).sum().item()
                        total_samples += labels.size(0)

                    progress_bar.set_postfix({"Loss": loss.item()})

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time - epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
                print(f"  📊 Hard examples detected: {num_hard_samples:,} ({100*num_hard_samples/(total_samples+1e-10):.1f}%)")
                print(f"  📊 Samples actually used: {samples_used:,} / {total_samples:,} ({100*samples_used/total_samples:.1f}%) [DROPOUT - SDBPD]")

                # Training F1 evaluation
                train_acc, train_f1, _ = self._evaluate_with_f1(criterion, num_classes, self.train_loader)
                print(f"Train F1 Macro: {train_f1:.4f}")

                # Test evaluation with F1 macro
                accuracy, f1_macro, val_loss = self._evaluate_with_f1(criterion, num_classes)
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, F1 Macro: {f1_macro:.4f}, Test Loss: {val_loss:.4f}")

                # Scheduler step
                scheduler.step()
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)
                epoch_test_f1_macros.append(f1_macro)

            else:
                # ============ Phase 2: Revision - Full Dataset ============
                print("Revision Phase: Using full dataset")
                self.model.train()
                epoch_start_time = time.time()
                running_loss = 0.0
                correct = 0
                total = 0

                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training (Full)")
                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    num_step += len(outputs)
                    samples_used += len(outputs)

                    with torch.no_grad():
                        preds = torch.argmax(outputs, dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)

                    if batch_idx % 10 == 0:
                        progress_bar.set_postfix({"Loss": loss.item()})

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = correct / total
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time - epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

                # Training F1 evaluation
                train_acc, train_f1, _ = self._evaluate_with_f1(criterion, num_classes, self.train_loader)
                print(f"Train F1 Macro: {train_f1:.4f}")

                # Test evaluation with F1 macro
                accuracy, f1_macro, val_loss = self._evaluate_with_f1(criterion, num_classes)
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, F1 Macro: {f1_macro:.4f}, Test Loss: {val_loss:.4f}")

                # Scheduler step
                scheduler.step()
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)
                epoch_test_f1_macros.append(f1_macro)

            samples_used_per_epoch.append(samples_used)

        end_time = time.time()
        log_memory(start_time, end_time)
        print(num_step)

        # Visualization
        plot_accuracy_time_multi(
            model_name=self.model_name,
            accuracy=epoch_accuracies,
            time_per_epoch=time_per_epoch,
            save_path=save_path,
            data_file=save_path
        )
        plot_accuracy_time_multi_test(
            model_name=self.model_name,
            accuracy=epoch_test_accuracies,
            time_per_epoch=time_per_epoch,
            samples_per_epoch=samples_used_per_epoch,
            threshold=self.threshold,
            save_path=save_path,
            data_file=save_path
        )
        plot_f1_macro_time(
            model_name=self.model_name,
            f1_scores=epoch_test_f1_macros,
            time_per_epoch=time_per_epoch,
            save_path=save_path,
            data_file=save_path
        )

        return self.model, num_step

    def train_with_adaptive_dropout(self, start_revision, task, num_classes, ema_alpha=0.1, epsilon=0.01, cls_num_list=None):
        """
        Adaptive Proportionality Dropout (APD)

        This method dynamically adjusts sampling proportions each epoch based on a blend of
        global class frequency and the model's measured difficulty for each class. Unlike
        SMRD-P which uses fixed class proportions, APD adapts to the model's learning progress.

        Key innovation: Adaptive proportions = global_proportions × (class_difficulty + epsilon)
        - Classes the model struggles with get higher sampling proportions
        - Uses Exponential Moving Average (EMA) to smooth difficulty estimates
        - Automatically balances between dataset distribution and learning difficulty

        Algorithm:
        1. Initialize:
           - Calculate global class proportions from dataset
           - Initialize difficulty tracker (all classes start at 0.5)
        2. For each epoch:
           - Track hard/total counts per class during training
           - Update difficulty: EMA(difficulty, measured_hard_rate)
           - Calculate adaptive proportions: global_prop × (difficulty + epsilon)
           - Use these proportions for sampling in next epoch
        3. Within each batch:
           - Determine budget (number of hard samples)
           - Allocate proportionally using adaptive proportions
           - Randomly sample from each class

        Args:
            start_revision: Epoch to stop APD and use full dataset
            task: Task type (classification, segmentation, longtail)
            num_classes: Number of classes in the dataset
            ema_alpha: EMA smoothing factor for difficulty tracking (default: 0.1)
                      Higher values = faster adaptation, lower = smoother
            epsilon: Small constant added to difficulty to prevent zero proportions (default: 0.01)
        """
        save_path = self.save_path
        self.model.to(self.device)

        # Check DDP status
        is_ddp = dist.is_available() and dist.is_initialized()
        is_main = not is_ddp or dist.get_rank() == 0

        # ==============================================================================
        # 1. ONE-TIME INITIALIZATION FOR APD
        # ==============================================================================
        if is_main:
            print("\n" + "="*60)
            print("APD: Adaptive Proportionality Dropout")
            print("="*60)
            print(f"Initializing APD with EMA Alpha={ema_alpha}, Epsilon={epsilon}")

        # Calculate global class proportions (once)
        if is_main:
            print("Calculating global class proportions...")

        dataset = self.train_loader.dataset
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
            if isinstance(targets, list):
                targets = torch.tensor(targets)
            elif isinstance(targets, torch.Tensor):
                targets = targets.cpu()
        else:
            # Fallback: iterate through dataset
            targets = []
            for idx in range(len(dataset)):
                sample = dataset[idx]
                label = sample[1] if isinstance(sample, tuple) else sample['label']
                if isinstance(label, torch.Tensor):
                    label = label.item()
                targets.append(label)
            targets = torch.tensor(targets)

        class_counts = torch.bincount(targets, minlength=num_classes)
        global_proportions = (class_counts.float() / class_counts.sum()).to(self.device)

        # Initialize the difficulty tracker (neutral start: 0.5 for all classes)
        class_difficulty_tracker = torch.full((num_classes,), 0.5, device=self.device)

        # For the first epoch, adaptive proportions are just the global ones
        adaptive_proportions = global_proportions.clone()

        if is_main:
            print(f"✓ Initialization complete")
            print(f"  Total samples: {class_counts.sum().item():,.0f}")
            print(f"  Initial difficulty: 0.5 for all classes")
            print(f"\nAPD will start from epoch 1 until epoch {start_revision}")
            print(f"After epoch {start_revision}, training will use full dataset\n")

        # Loss function selection
        if task == 'classification':
            criterion = nn.CrossEntropyLoss()
        elif task == 'longtail':
            train_sampler = None
            idx = self.epochs // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(self.device)
            criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda(self.device)
            if is_main:
                print("Using Focal Loss with class-balanced weights (APD also handles imbalance via adaptive proportions)")
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        # Create scheduler
        if self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=0)
            if is_main:
                print(f"Using CosineAnnealingLR scheduler (T_max={self.epochs}, eta_min=0)")
        else:  # "step"
            scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
            if is_main:
                print(f"Using StepLR scheduler (step_size=1, gamma=0.98)")

        epoch_losses = []
        epoch_accuracies = []
        epoch_test_accuracies = []
        epoch_test_losses = []
        epoch_test_f1_macros = []
        time_per_epoch = []
        samples_used_per_epoch = []

        start_time = time.time()
        num_step = 0

        for epoch in range(self.epochs):
            self._set_sampler_epoch(epoch)

            # ==============================================================================
            # 2. PER-EPOCH SETUP FOR APD
            # ==============================================================================
            # Reset counters to accumulate stats for this epoch
            epoch_hard_counts = torch.zeros(num_classes, device=self.device)
            epoch_total_counts = torch.zeros(num_classes, device=self.device)

            samples_used = 0
            num_hard_samples = 0

            if is_main:
                print(f"\nEpoch [{epoch+1}/{self.epochs}]")

            if epoch < start_revision:
                # ============================================================
                # APD Phase: Adaptive Proportional Dropout
                # ============================================================
                self.model.train()
                epoch_start_time = time.time()
                running_loss = 0.0
                total_correct = 0
                total_samples = 0

                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                                  desc="Training (APD)", disable=not is_main)

                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # STEP 1: Determine sampling budget (number of hard samples)
                    with torch.no_grad():
                        outputs = self.model(inputs)
                        preds = torch.argmax(outputs, dim=1)

                        if self.threshold == 0:
                            # Misclassification-based
                            mask = preds != labels
                        else:
                            # Confidence-based
                            prob = torch.softmax(outputs, dim=1)
                            correct_class_probs = prob[torch.arange(labels.size(0)), labels]
                            mask = correct_class_probs < self.threshold

                        budget = mask.sum().item()
                        num_hard_samples += budget

                    # ==============================================================================
                    # 3. PER-BATCH DATA ACCUMULATION FOR APD
                    # ==============================================================================
                    with torch.no_grad():
                        hard_labels = labels[mask]
                        # Count total & hard occurrences for each class in this batch
                        batch_total_counts = torch.bincount(labels, minlength=num_classes)
                        batch_hard_counts = torch.bincount(hard_labels, minlength=num_classes)
                        # Add to the epoch-level accumulators
                        epoch_total_counts += batch_total_counts
                        epoch_hard_counts += batch_hard_counts

                    # STEP 2: Proportional Sampling (uses adaptive proportions)
                    if budget == 0:
                        # Fallback: sample at least one to avoid DDP desync
                        indices = torch.randint(0, inputs.size(0), (1,), device=self.device)
                    else:
                        final_indices = []
                        unique_classes = labels.unique()

                        for class_idx in unique_classes:
                            # Use the ADAPTIVE proportion for this epoch
                            proportion = adaptive_proportions[class_idx]

                            # Calculate number of samples to select for this class
                            num_to_sample_for_class = round(budget * proportion.item())

                            if num_to_sample_for_class == 0:
                                continue

                            # Get indices of all samples belonging to this class in current batch
                            indices_of_class_in_batch = (labels == class_idx).nonzero(as_tuple=True)[0]

                            # Don't sample more than available
                            num_to_sample_for_class = min(num_to_sample_for_class, len(indices_of_class_in_batch))

                            # Randomly select from available samples for this class
                            perm = torch.randperm(len(indices_of_class_in_batch), device=self.device)
                            selected_local_indices = perm[:num_to_sample_for_class]

                            # Map back to original batch indices
                            selected_batch_indices = indices_of_class_in_batch[selected_local_indices]
                            final_indices.append(selected_batch_indices)

                        if not final_indices:
                            # Fallback if rounding resulted in zero samples
                            indices = torch.randint(0, inputs.size(0), (1,), device=self.device)
                        else:
                            indices = torch.cat(final_indices)

                        # STEP 3: Adjust to match budget exactly (handle rounding)
                        if len(indices) > budget:
                            # Too many: randomly drop some
                            indices = indices[torch.randperm(len(indices), device=self.device)[:budget]]
                        elif len(indices) < budget and budget <= len(inputs):
                            # Too few: add random samples
                            available_indices = torch.ones(len(inputs), dtype=torch.bool, device=self.device)
                            available_indices[indices] = False
                            add_indices = available_indices.nonzero(as_tuple=True)[0]

                            num_to_add = budget - len(indices)
                            if len(add_indices) > 0:
                                num_to_add = min(num_to_add, len(add_indices))
                                add_perm = torch.randperm(len(add_indices), device=self.device)[:num_to_add]
                                indices = torch.cat([indices, add_indices[add_perm]])

                        # Final safety check
                        if len(indices) == 0:
                            indices = torch.randint(0, inputs.size(0), (1,), device=self.device)

                    # STEP 4: Train on sampled subset
                    inputs_sampled = inputs[indices]
                    labels_sampled = labels[indices]

                    optimizer.zero_grad()
                    outputs_sampled = self.model(inputs_sampled)
                    loss = criterion(outputs_sampled, labels_sampled)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    num_step += len(outputs_sampled)
                    samples_used += len(outputs_sampled)

                    # Evaluate on full batch for accuracy tracking
                    with torch.no_grad():
                        outputs = self.model(inputs)
                        preds = torch.argmax(outputs, dim=1)
                        total_correct += (preds == labels).sum().item()
                        total_samples += labels.size(0)

                    progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time - epoch_start_time)

                if is_main:
                    print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
                    print(f"  📊 Hard examples detected: {num_hard_samples:,} ({100*num_hard_samples/total_samples:.1f}%)")
                    print(f"  📊 Samples actually used: {samples_used:,} / {total_samples:,} ({100*samples_used/total_samples:.1f}%) [DROPOUT - APD]")

                # ==============================================================================
                # 4. END-OF-EPOCH ADAPTIVE UPDATE STEP
                # ==============================================================================
                # Avoid division by zero for classes not seen in the epoch
                measured_difficulty = epoch_hard_counts / (epoch_total_counts + 1e-8)

                # Update the difficulty tracker using EMA
                class_difficulty_tracker = (1 - ema_alpha) * class_difficulty_tracker + ema_alpha * measured_difficulty

                # Calculate the new unnormalized adaptive weights for the NEXT epoch
                adaptive_weights = global_proportions * (class_difficulty_tracker + epsilon)

                # Normalize to get the final proportions
                adaptive_proportions = adaptive_weights / adaptive_weights.sum()

                if is_main:
                    print("  🔄 APD: Updated adaptive proportions for next epoch")
                    # Print top 5 "hardest" classes for analysis
                    top5_values, top5_indices = torch.topk(adaptive_proportions, min(5, num_classes))
                    print(f"  💡 Top 5 classes by sampling proportion:")
                    for i, idx in enumerate(top5_indices):
                        print(f"     Class {idx.item():3d}: Proportion={adaptive_proportions[idx]:.4f}, "
                              f"Difficulty={class_difficulty_tracker[idx]:.4f}")

                # Evaluation
                train_acc, train_f1, _ = self._evaluate_with_f1(criterion, num_classes, self.train_loader)
                if is_main:
                    print(f"Train F1 Macro: {train_f1:.4f}")

                accuracy, f1_macro, val_loss = self._evaluate_with_f1(criterion, num_classes)
                if is_main:
                    print(f"Test Accuracy: {accuracy:.4f}, F1 Macro: {f1_macro:.4f}, Test Loss: {val_loss:.4f}")

                scheduler.step()
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)
                epoch_test_f1_macros.append(f1_macro)

            else:
                # ============================================================
                # Revision Phase: Full Dataset
                # ============================================================
                if is_main:
                    print("Revision Phase: Using full dataset")

                self.model.train()
                epoch_start_time = time.time()
                running_loss = 0.0
                correct = 0
                total = 0

                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                                  desc="Training (Full)", disable=not is_main)

                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    num_step += len(outputs)
                    samples_used += len(outputs)

                    with torch.no_grad():
                        preds = torch.argmax(outputs, dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)

                    if is_main and batch_idx % 10 == 0:
                        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = correct / total
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time - epoch_start_time)

                if is_main:
                    print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

                # Evaluation
                train_acc, train_f1, _ = self._evaluate_with_f1(criterion, num_classes, self.train_loader)
                if is_main:
                    print(f"Train F1 Macro: {train_f1:.4f}")

                accuracy, f1_macro, val_loss = self._evaluate_with_f1(criterion, num_classes)
                if is_main:
                    print(f"Test Accuracy: {accuracy:.4f}, F1 Macro: {f1_macro:.4f}, Test Loss: {val_loss:.4f}")

                scheduler.step()
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)
                epoch_test_f1_macros.append(f1_macro)

            samples_used_per_epoch.append(samples_used)

        end_time = time.time()
        log_memory(start_time, end_time)

        if is_main:
            print(f"Total steps: {num_step}")
            print(f"\nTotal training time: {end_time - start_time:.2f}s")

        # Visualization (only on main process)
        if is_main:
            plot_accuracy_time_multi(
                model_name=self.model_name,
                accuracy=epoch_accuracies,
                time_per_epoch=time_per_epoch,
                save_path=save_path,
                data_file=save_path
            )
            plot_accuracy_time_multi_test(
                model_name=self.model_name,
                accuracy=epoch_test_accuracies,
                time_per_epoch=time_per_epoch,
                samples_per_epoch=samples_used_per_epoch,
                threshold=self.threshold,
                save_path=save_path,
                data_file=save_path
            )
            plot_f1_macro_time(
                model_name=self.model_name,
                f1_scores=epoch_test_f1_macros,
                time_per_epoch=time_per_epoch,
                save_path=save_path,
                data_file=save_path
            )

        return self.model, num_step
