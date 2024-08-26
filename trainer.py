import os 
import sys
import numpy as np
from tqdm import tqdm
import torch
import torchaudio 
import torch.nn as nn
from utils import ModelCheckpoint

class Trainer():
    def __init__(self, finetune, train_loader, valid_loader, test_loader, 
                 model, optimizer, criterion, 
                 epochs, scheduler, patience, 
                 best_model_path, logger, log_interval, device):
        self.finetune = finetune
        # data
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        # model
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.scheduler = scheduler
        self.patience = patience
        self.best_model_path = best_model_path
        # supplementary
        self.logger = logger
        self.log_interval = log_interval
        self.device = device
        if not self.finetune: 
            if isinstance(model, list):
                self.encoder_ckpt = ModelCheckpoint(best_model_path, 1)
                self.decoder_ckpt = ModelCheckpoint(best_model_path, 1)
            else:
                self.model_ckpt = ModelCheckpoint(best_model_path, 1)
            
    def train(self):        
        losses = []
        best_loss = np.inf
        for epoch in range(1, self.epochs+1):
            if isinstance(self.model, list):
                # CNN-LSTM, CNN-GRU
                encoder, decoder = self.model
                encoder.train()
                decoder.train()
            else:
                # ConvLSTM
                self.model.train()
            
            N_count = 0
            for batch_idx, (X, y) in enumerate(self.train_loader):
                X, y = X.to(self.device), y.to(self.device)

                N_count += X.size(0)

                self.optimizer.zero_grad()
                if isinstance(self.model, list):
                    output = decoder(encoder(X))
                else:
                    output = self.model(X)

                loss = self.criterion(output.squeeze(), y.squeeze())
                losses.append(loss.item())

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

                # if ((batch_idx + 1) % self.log_interval == 0) or (batch_idx==len(self.train_loader)):
                #     self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tMAE: {:.6f}'.format(
                #         epoch, N_count, len(self.train_loader.dataset), 100. * (batch_idx + 1) / len(self.train_loader), np.mean(losses)))
                
            loss_val = self.valid_step()
            self.logger.info(f'Epoch [{epoch}/{self.epochs}] Train Loss {np.mean(losses):.6f} Validation Loss {loss_val:.6f}')
            
            if loss_val <= best_loss:
                best_loss = loss_val
            
            if not self.finetune:
                if isinstance(self.model, list):
                    self.encoder_ckpt.save_checkpoint(encoder.state_dict(), 'encoder', epoch, loss_val)
                    self.decoder_ckpt.save_checkpoint(decoder.state_dict(), 'decoder', epoch, loss_val)
                else:
                    self.model_ckpt.save_checkpoint(self.model.state_dict(), 'model', epoch, loss_val)
            else: 
                pass
                
        return best_loss
                
    def valid_step(self):
        if isinstance(self.model, list):
            encoder, decoder = self.model
            encoder.eval()
            decoder.eval()
        else:
            self.model.eval()

        losses = []
        with torch.no_grad():
            for X, y in self.valid_loader:
                X, y = X.to(self.device), y.to(self.device)

                if isinstance(self.model, list):
                    output = decoder(encoder(X))
                else:
                    output = self.model(X)
                    
                loss = self.criterion(output.squeeze(), y.squeeze())
                losses.append(loss.item())
        
        return np.mean(losses)

    def predict(self): 
        true = []
        pred = []
        if isinstance(self.model, list):
            encoder, decoder = self.model
            encoder.load_state_dict(torch.load(os.path.join(self.best_model_path, os.listdir(self.best_model_path)[0])))
            encoder.load_state_dict(torch.load(os.path.join(self.best_model_path, os.listdir(self.best_model_path)[0])))
            encoder.eval()
            decoder.eval()
        else:
            self.model.load_state_dict(torch.load(os.path.join(self.best_model_path, os.listdir(self.best_model_path)[0])))
            self.model.eval()
            
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)

                if isinstance(self.model, list):
                    output = decoder(encoder(X))
                else:
                    output = self.model(X)
                
                pred.extend(output.cpu().data.squeeze().numpy())
                true.extend(y.cpu().data.squeeze().numpy())
        
        return true, pred