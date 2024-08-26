import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.optim as optim

def conv2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
    return outshape

class EncoderCNN(nn.Module):
    def __init__(self, img_x=128, img_y=128, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, output_dim=256):
        super(EncoderCNN, self).__init__()

        self.img_x = img_x
        self.img_y = img_y
        self.output_dim = output_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 32, 64, 128, 256
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # conv2D output shapes
        self.conv1_outshape = conv2D_output_size((self.img_x, self.img_y), self.pd1, self.k1, self.s1)  # Conv1 output shape
        self.conv2_outshape = conv2D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
        self.conv3_outshape = conv2D_output_size(self.conv2_outshape, self.pd3, self.k3, self.s3)
        self.conv4_outshape = conv2D_output_size(self.conv3_outshape, self.pd4, self.k4, self.s4)

        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.pd1),
            nn.BatchNorm2d(self.ch1, momentum=0.01),
            nn.ReLU(inplace=True),                      
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.pd2),
            nn.BatchNorm2d(self.ch2, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.pd3),
            nn.BatchNorm2d(self.ch3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=self.k4, stride=self.s4, padding=self.pd4),
            nn.BatchNorm2d(self.ch4, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.drop = nn.Dropout2d(self.drop_p)
        self.fc1 = nn.Linear(self.ch4 * self.conv4_outshape[0] * self.conv4_outshape[1], self.fc_hidden1)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, self.output_dim)
        
    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            x = self.conv1(x_3d[:, t, :, :, :])
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = x.view(x.size(0), -1)

            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)
            cnn_embed_seq.append(x)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)

        return cnn_embed_seq

class DecoderLSTM(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, num_layers=3, output_dim=5, drop_p=0.1):
        super(DecoderLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.drop_p = drop_p

        self.LSTM = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,        
            num_layers=self.num_layers,       
            batch_first=True,
        )

        self.fc1 = nn.Linear(self.hidden_dim, int(self.hidden_dim/2))
        self.fc2 = nn.Linear(int(self.hidden_dim/2), int(self.hidden_dim/4))
        self.proj = nn.Linear(int(self.hidden_dim/4), self.output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.drop_p)

    def forward(self, x):
        
        self.LSTM.flatten_parameters()
        LSTM_out, (h_n, h_c) = self.LSTM(x, None)  
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
        """ None represents zero initial hidden state. LSTM_out has shape=(batch, time_step, output_size) """

        x = self.relu(self.fc1(LSTM_out[:, -1, :]))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.proj(x)

        return x
    
class DecoderGRU(nn.Module) :
    def __init__(self, input_dim=256, hidden_dim=128, num_layers=3, output_dim=5, drop_p=0.1):
        super(DecoderGRU, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.drop_p = drop_p

        self.GRU = nn.GRU(
            input_size = self.input_dim,
            hidden_size = self.hidden_dim,
            num_layers = num_layers,
            batch_first = True,
        )

        self.fc1 = nn.Linear(self.hidden_dim, int(self.hidden_dim/2))
        self.fc2 = nn.Linear(int(self.hidden_dim/2), int(self.hidden_dim/4))
        self.proj = nn.Linear(int(self.hidden_dim/4), self.output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.drop_p)

    def forward(self, x) :
        self.GRU.flatten_parameters()
        GRU_out, h_n = self.GRU(x, None)
        
        x = self.relu(self.fc1(GRU_out[:, -1, :]))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc2(x)

        return x