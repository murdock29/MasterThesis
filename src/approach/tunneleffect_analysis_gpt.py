import random
import time
import torch
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from torchmetrics.functional.classification import binary_confusion_matrix
import torch.nn.functional as F
from src.loggers.exp_logger import ExperimentLogger
from src.approach.learning_approach import Learning_Appr
from torchvision.models import vgg19


import time
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vgg19
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR


class Appr(Learning_Appr):
    """Class implementing the finetuning baseline (fixed training loop)."""

    def __init__(self, model, device, nepochs=160, lr=0.1, lr_min=1e-4, lr_factor=3,
                 lr_patience=5, clipgrad=10000, momentum=0.9, wd=1e-4,
                 eval_on_train=False, logger=None):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor,
                                   lr_patience, clipgrad, momentum, wd,
                                   eval_on_train, logger)
        self._criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss as criterion

    def _get_optimizer(self):
        """Returns the optimizer (SGD as in the paper)."""
        return SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wd)

    def train(self, trn_loader, val_loader):
        """Contains the epochs loop with scheduler and proper lr updates."""
        best_model = self.model.get_copy()
        best_loss = np.inf

        # Create optimizer and scheduler (decay LR at epochs 80 and 120)
        self.optimizer = self._get_optimizer()
        scheduler = MultiStepLR(self.optimizer, milestones=[80, 120], gamma=0.1)

        # Loop epochs
        for e in range(self.nepochs):
            clock0 = time.time()
            self.train_epoch(trn_loader)
            clock1 = time.time()

            if self.eval_on_train:
                train_loss, train_acc = self.eval(trn_loader)
                clock2 = time.time()
                print(f'| Epoch {e + 1:3d}, time={clock1 - clock0:5.1f}s/{clock2 - clock1:5.1f}s '
                      f'| Train: loss={train_loss:.6f}, acc={train_acc * 100:5.1f}%', end='')
                if self.logger:
                    self.logger.log_scalar(iter=e + 1, name="loss", value=train_loss, group="train")
                    self.logger.log_scalar(iter=e + 1, name="acc", value=train_acc * 100, group="train")
            else:
                print(f'\n| Epoch {e + 1:3d}, time={clock1 - clock0:5.1f}s | Train: skip eval |', end='')

            # Validation
            clock3 = time.time()
            valid_loss, valid_acc = self.eval(val_loader)
            clock4 = time.time()

            current_lr = float(self.optimizer.param_groups[0]['lr'])
            print(f' Valid: time={clock4 - clock3:5.1f}s loss={valid_loss:.6f}, acc={valid_acc * 100:5.1f}%, lr={current_lr:.5e} |', end='')

            if self.logger:
                self.logger.log_scalar(iter=e + 1, name="lr", value=current_lr, group="train")

            # Save checkpoint every 10 epochs
            if (e + 1) % 10 == 0:
                path_save = f'./model/tunnel_effect_analysis/checkpoint_epoch_{e + 1}.pt'
                torch.save(self.model.state_dict(), path_save)
                print(f' Checkpoint saved at {path_save}', end='')

            # Scheduler step (updates optimizer LR when needed)
            scheduler.step()

            # Keep best model based on validation loss
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = self.model.get_copy()
                print(' *', end='')

            print('')  # Newline for epoch

        # Restore best model
        self.model.set_state_dict(best_model)

    def train_epoch(self, loader):
        """Runs a single training epoch."""
        self.model.train()

        for images, targets in loader:
            # Move data to device
            images = images.to(self.device)
            targets = targets.to(self.device)

            # If the targets are one-hot encoded, convert them to class indices
            if targets.ndim > 1 and targets.shape[1] > 1:
                targets_idx = torch.argmax(targets, dim=1)
            else:
                targets_idx = targets.view(-1).long()

            # Forward pass
            outputs = self.model(images)  # Raw logits
            loss = self._criterion(outputs, targets_idx)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.clipgrad is not None and self.clipgrad < 1e5:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clipgrad)

            self.optimizer.step()

    def eval(self, loader):
        """Evaluation loop."""
        with torch.no_grad():
            total_loss = 0.0
            total_correct = 0
            total_num = 0
            self.model.eval()

            for images, targets in loader:
                # Move data to device
                images = images.to(self.device)
                targets = targets.to(self.device)

                # If the targets are one-hot encoded, convert them to class indices
                if targets.ndim > 1 and targets.shape[1] > 1:
                    targets_idx = torch.argmax(targets, dim=1)
                else:
                    targets_idx = targets.view(-1).long()

                # Forward pass
                outputs = self.model(images)  # Raw logits
                loss = self._criterion(outputs, targets_idx)

                # Predictions and statistics
                predicted = torch.argmax(outputs, dim=1)
                total_loss += float(loss.item()) * images.size(0)
                total_correct += (predicted == targets_idx).sum().item()
                total_num += images.size(0)

            avg_loss = total_loss / total_num
            avg_acc = total_correct / total_num

        return avg_loss, avg_acc