import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path, folders, labels, frames, transform=None):
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        X = []
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, 'f_{:05}.jpg'.format(i))).convert('L')

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        folder = self.folders[index]

        X = self.read_images(self.data_path, folder, self.transform)
        y = torch.FloatTensor(np.array([self.labels[index]]))
        return X, y