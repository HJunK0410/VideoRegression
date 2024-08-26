import os
import math
import torch
import random
import numpy as np

import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    
class ModelCheckpoint:
    def __init__(self, directory, max_checkpoints=3):
        self.directory = directory
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []

        if not os.path.exists(directory):
            os.makedirs(directory)

    def save_checkpoint(self, ckpt, ckpt_name, epoch, performance_metric):
        # Save the model state_dict with the performance metric in the filename
        filename = os.path.join(self.directory, f'{ckpt_name}_{epoch}_perf_{performance_metric:.4f}.pt')
        torch.save(ckpt, filename)
        self.checkpoints.append((filename, performance_metric))

        # Sort checkpoints by performance_metric in descending order (best performance first)
        self.checkpoints.sort(key=lambda x: x[1])  #, reverse=True) # ACTIVATE THIS WHEN BIGGER IS BETTER

        # Keep only the best `max_checkpoints` checkpoints
        if len(self.checkpoints) > self.max_checkpoints:
            # Remove the worst checkpoint (lowest performance_metric)
            worst_checkpoint = self.checkpoints.pop(-1)
            os.remove(worst_checkpoint[0])