import random
import time
import torch
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from torchmetrics.functional.classification import binary_confusion_matrix
import torch.nn.functional as F
from src.loggers.exp_logger import ExperimentLogger


class Learning_Appr:
    """Basic class for implementing learning approaches"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, eval_on_train=False, logger: ExperimentLogger = None):
        self.model = model
        self.device = device
        self.nepochs = nepochs
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.momentum = momentum
        self.wd = wd
        self.logger = logger
        self.eval_on_train = eval_on_train
        self.optimizer = None

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        return torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train(self, trn_loader, val_loader):
        """Contains the epochs loop"""
        lr = self.lr
        best_loss = np.inf
        patience = self.lr_patience
        best_model = self.model.get_copy()

        self.optimizer = self._get_optimizer()

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(trn_loader)
            clock1 = time.time()
            if self.eval_on_train:
                train_loss, train_acc, train_metric = self.eval(trn_loader)
                clock2 = time.time()
                print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, acc={:5.1f}%, dice={:5.1f}%|'.format(
                    e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc, 100 * train_metric), end='')
                self.logger.log_scalar(iter=e + 1, name="loss", value=train_loss, group="train")
                self.logger.log_scalar(iter=e + 1, name="acc", value=100 * train_acc, group="train")
                self.logger.log_scalar(iter=e + 1, name="dice", value=100 * train_metric, group="train")
            else:
                print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

            # Valid
            clock3 = time.time()
            valid_loss, valid_acc, valid_metric = self.eval(val_loader)
            clock4 = time.time()
            print(' Valid: time={:5.1f}s loss={:.3f}, acc={:5.1f}%, dice={:5.1f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc, 100 * valid_metric), end='')
            self.logger.log_scalar(iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(iter=e + 1, name="acc", value=100 * valid_acc, group="valid")
            self.logger.log_scalar(iter=e + 1, name="dice", value=100 * valid_metric, group="valid")

            # Adapt learning rate - patience scheme - early stopping regularization
            if valid_loss < best_loss:
                # if the loss goes down, keep it as the best model and end line with a star ( * )
                best_loss = valid_loss
                best_model = self.model.get_copy()
                patience = self.lr_patience
                print(' *', end='')
            else:
                # if the loss does not go down, decrease patience
                patience -= 1
                if patience <= 0:
                    # if it runs out of patience, reduce the learning rate
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        # if the lr decreases below minimum, stop the training session
                        print()
                        break
                    # reset patience and recover best model so far to continue training
                    patience = self.lr_patience
                    self.optimizer.param_groups[0]['lr'] = lr
                    self.model.set_state_dict(best_model)
            self.logger.log_scalar(iter=e + 1, name="patience", value=patience, group="train")
            self.logger.log_scalar(iter=e + 1, name="lr", value=lr, group="train")
            print()
        self.model.set_state_dict(best_model)

    def train_epoch(self, loader):
        """Runs a single epoch"""
        self.model.train()
        for images, targets in loader:
            # Forward current model
            outputs = self.model(images.to(self.device))
            loss = self.criterion(outputs, targets.to(self.device))
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def eval(self, loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc, total_metric, total_num, total_num_acc = 0, 0, 0, 0, 0
            self.model.eval()
            for images, targets in loader:
                # Forward current model
                outputs = self.model(images.to(self.device))
                loss = self.criterion(outputs, targets.to(self.device))
                # Accuracy
                preds = torch.nn.functional.sigmoid(outputs).round()
                acc = (preds == targets.to(self.device)).float()
                # Metric -- the base approach does not provide an extra metric, just accuracy
                # Log
                total_loss += loss.sum().item() * len(targets)
                total_acc += acc.sum().item()
                total_num += len(targets)
                total_num_acc += len(targets) * targets.size(dim=2) * targets.size(dim=3)
        return total_loss / total_num, total_acc / total_num, 0.0

    def predict(self, loader):  # TODO: fix this part
        """pass the patches of the test_set through the model with the stitching included"""
        with torch.no_grad():
            total_loss, total_acc, total_metric, total_num, total_num_acc = 0, 0, 0, 0, 0
            self.model.eval()
            for images, targets in loader:
                # Forward current model
                B, C, H, W = images.shape
                C = 1
                kernel_size = 572
                stride = 428
                patches = images.unfold(3, kernel_size, stride).unfold(2, kernel_size, stride).permute(0,1,2,3,5,4) # [B, C, nb_patches_h, nb_patches_w, kernel_size, kernel_size]
                patches = patches.flatten(2, 3) # [B, C, nb_patches_all, kernel_size, kernel_size]
                num_patches = patches.shape[2]
                patches = patches.permute(2, 0, 1, 3, 4)

                # perform the operations on each patch
                container = torch.zeros(num_patches, B, C, kernel_size, kernel_size)
                for patch in range(num_patches):
                    outputs = self.model(patches[patch].to(self.device))
                    container[patch] = outputs

                # outputs = self.model(patches.flatten(0, 1).to(self.device))
                # outputs = container.unflatten(0, (num_patches, B)).permute(0, 2, 1, 3, 4)
                outputs = container.permute(1, 2, 0, 3, 4)

                weight = torch.ones_like(outputs)
                # reshape output to match F.fold input
                patches = container.contiguous().view(B, C, -1, kernel_size * kernel_size) # [B, C, nb_patches_all, kernel_size*kernel_size]
                patches = patches.permute(0, 1, 3, 2) # [B, C, kernel_size*kernel_size, nb_patches_all]
                patches = patches.contiguous().view(B, C * kernel_size * kernel_size, -1) # [B, C*prod(kernel_size), L] as expected by Fold

                # https://pytorch.org/docs/stable/nn.html#torch.nn.Fold
                output = F.fold(
                    patches, output_size=(H, W), kernel_size=kernel_size, stride=stride)

                weight_mask = weight.contiguous().view(B, C, -1, kernel_size * kernel_size)
                weight_mask = weight_mask.permute(0, 1, 3, 2)
                weight_mask = weight_mask.contiguous().view(B, C * kernel_size * kernel_size, -1)
                w_mask = F.fold(
                    weight_mask, output_size=(H, W), kernel_size=kernel_size, stride=stride)
                output /= w_mask

                outputs = output
                preds = torch.nn.functional.sigmoid(outputs)
                loss = self.criterion(preds, targets, targets)

                # Accuracy
                acc = (torch.round(preds) == targets).sum()
                torch.cuda.empty_cache()
                # Log
                total_loss += loss.sum().item() * len(targets)
                total_acc += acc.item()
                total_num += len(targets)
                total_num_acc += len(targets) * targets.size(dim=2) * targets.size(dim=3) #height x width to get the pixel count
            return total_loss / total_num, total_acc / total_num_acc, 0.0

    def criterion(self, outputs, targets):
        """Returns the loss value"""
        #todo check if softmax needed
        # preds = torch.nn.functional.sigmoid(outputs)
        return torch.nn.functional.binary_cross_entropy(outputs, targets)
