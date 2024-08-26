import os
import random
import pickle
import logging
import numpy as np
import pandas as pd

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from dataset import *
from model.ConvLSTM import *
from trainer import Trainer
from utils import seed_everything

seed_everything(42)

# == arguments & parameters == #
input_dim = 1
hidden_dim = [16, 32]
kernel_size = [(1,1), (3,3)]
num_layers = [1, 3]
output_dim = 5
batch_size = [2, 4, 6, 8]
learning_rate = 1e-4
T_max = 30
eta_min = 1e-5
epochs = 50
patience = 5
log_interval = 10
device = 'cuda'
data_path = '/mnt/hdd1/hyunjun/speckle/frames'
finetune = True # If finetune, checkpoint will not be saved

# ============================ #
   
with open('./data/train.pickle', 'rb') as f:
    train_list_label = pickle.load(f)
f.close()

with open('./data/valid.pickle', 'rb') as f:
    valid_list_label = pickle.load(f)
f.close()

with open('./data/test.pickle', 'rb') as f:
    test_list_label = pickle.load(f)
f.close()

train_list = train_list_label[0]
val_list = valid_list_label[0]
test_list = test_list_label[0]

train_label = np.array(train_list_label[1])
val_label = np.array(valid_list_label[1])
test_label = np.array(test_list_label[1])

median = np.median(train_label, axis=0)
iqr = np.quantile(train_label, 0.75, axis=0) - np.quantile(train_label, 0.25, axis=0)

train_label = (train_label-median)/iqr
val_label = (val_label-median)/iqr
test_label = (test_label-median)/iqr

begin_frame, end_frame, skip_frame = 0, 4999, 500
selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

hd_col = []
ks_col = []
nl_col = []
bs_col = []
best_loss_col = []
for hd in hidden_dim:
    for ks in kernel_size:
        for nl in num_layers:
            for bs in batch_size:
                hd_col.append(str(hd))
                ks_col.append(str(ks))
                nl_col.append(str(nl))
                bs_col.append(str(bs))
                save_model_path = '/mnt/hdd1/hyunjun/speckle/convlstm/multi/weights/'
                save_pred_path = '/mnt/hdd1/hyunjun/speckle/convlstm/multi/predictions/'
                log_path = '/mnt/hdd1/hyunjun/speckle/convlstm/multi/log/'
                log_path += f'hd{hd}_ks{ks}_nl{nl}_bs{bs}'
                if not finetune:
                    for paths in [data_path, save_model_path, save_pred_path]:
                        os.makedirs(paths, exist_ok=True)
                else:
                    os.makedirs(log_path, exist_ok=True)
                # logger
                logging.basicConfig(level=logging.INFO, filemode='w', format="%(message)s")
                logger = logging.getLogger()
                logger.addHandler(logging.FileHandler(log_path + '/log.log'))
                logger.info(f'hidden dim: {hd}, kernel size: {ks}, num layers: {nl}, batch size: {bs}')

                params = {'batch_size': bs, 'shuffle': True, 'num_workers': 4, 'pin_memory': True}
                test_params = {'batch_size': bs, 'shuffle': False, 'num_workers': 4, 'pin_memory': True}

                transform = transforms.Compose([transforms.Resize([128, 128]),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.5], std=[0.5])])

                train_set, valid_set, test_set = CustomDataset(data_path, train_list, train_label, selected_frames, transform=transform), \
                                                CustomDataset(data_path, val_list, val_label, selected_frames, transform=transform), \
                                                CustomDataset(data_path, test_list, test_label, selected_frames, transform=transform)
                                                
                train_loader = DataLoader(train_set, **params)
                valid_loader = DataLoader(valid_set, **test_params)
                test_loader = DataLoader(test_set, **test_params)

                model = ConvLSTM(input_dim=input_dim, 
                                hidden_dim=hd, 
                                kernel_size=ks, 
                                num_layers = nl, 
                                batch_first=True, bias=True, 
                                output_dim=output_dim
                                ).to(device)

                criterion = nn.L1Loss()
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

                trainer = Trainer(finetune, train_loader, valid_loader, test_loader, 
                                model, optimizer, criterion, 
                                epochs, scheduler, patience, 
                                save_model_path, logger, log_interval, device)

                best_loss = trainer.train()
                best_loss_col.append(best_loss)
                
                logging.shutdown()
                
result = pd.DataFrame({'hidden dim': hd_col, 
                        'kernel size': ks_col, 
                        'num layers': nl_col, 
                        'batch size': bs_col, 
                        'best loss': best_loss_col})
result.to_csv('finetune_result/convlstm_multi.csv', index=None)