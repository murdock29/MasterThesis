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


class Appr(Learning_Appr):
    """Class implementing the finetuning baseline"""

    def __init__(self, model, device, nepochs=160, lr=0.1, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=1e-4, eval_on_train=False,logger=None):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   eval_on_train, logger)
        self.test_model = vgg19(pretrained=False)

    def _get_optimizer(self):
        """Returns the optimizer"""
        return torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train(self, trn_loader, val_loader):
        """Contains the epochs loop"""
        lr = self.lr
        best_model = self.model.get_copy()

        self.optimizer = self._get_optimizer()

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(trn_loader)
            clock1 = time.time()
            if self.eval_on_train:
                train_loss, train_acc = self.eval(trn_loader)
                clock2 = time.time()
                print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, acc={:5.1f}%'.format(
                    e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                self.logger.log_scalar(iter=e + 1, name="loss", value=train_loss, group="train")
                self.logger.log_scalar(iter=e + 1, name="acc", value=100 * train_acc, group="train")
            else:
                print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

            # Valid
            clock3 = time.time()
            # valid_loss, valid_acc = self.eval(val_loader)
            clock4 = time.time()
            # print(' Valid: time={:5.1f}s loss={:.3f}, acc={:5.1f}%, dice={:5.1f}% |'.format(
            #     clock4 - clock3, valid_loss, 100 * valid_acc, 100 * valid_metric), end='')
            # self.logger.log_scalar(iter=e + 1, name="loss", value=valid_loss, group="valid")
            # self.logger.log_scalar(iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

            # Save checkpoint every 10 epochs
            if (e + 1) % 10 == 0:
                path_save = f'./model/tunnel_effect_analysis/checkpoint_epoch_{e + 1}.pt'
                torch.save(self.model.state_dict(), path_save)
                print(f'Checkpoint saved at {path_save}')

            # Adapt learning rate at epochs 80 and 120
            if (e+1) in [80, 120]:
                self.lr *= 0.1

            self.logger.log_scalar(iter=e + 1, name="lr", value=lr, group="train")
            print()
        self.model.set_state_dict(best_model)

    def train_epoch(self, loader):
        """Runs a single epoch"""
        self.model.train()
        for images, targets in loader:
            # Forward current model
            outputs = self.model(images.to(self.device))
            loss = self.criterion(outputs, torch.argmax(targets, dim=1).to(self.device))
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def eval(self, loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc, total_num = 0, 0, 0
            self.model.eval()
            for images, targets in loader:
                # Forward current model
                outputs = self.model(images.to(self.device))
                loss = self.criterion(outputs, torch.argmax(targets, dim=1).to(self.device))
                # Accuracy
                predicted = torch.argmax(outputs.data, 1)
                total_loss += loss.sum().item()
                total_num += len(targets)
                total_acc += (predicted.cpu() == torch.argmax(targets, dim=1)).sum().item()
        return total_loss / total_num, total_acc / total_num

    def criterion(self, outputs, targets):
        """Returns the loss value"""
        # preds = torch.nn.functional.sigmoid(outputs)
        return torch.nn.functional.cross_entropy(outputs, targets)
