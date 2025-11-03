import math
import torch
from torch.optim.lr_scheduler import LambdaLR


class WarmupCosineLR(LambdaLR):
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_factor=0.1, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_factor = warmup_factor
        super(WarmupCosineLR, self).__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, epoch):
        if epoch < self.warmup_epochs:
            return self.warmup_factor * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
