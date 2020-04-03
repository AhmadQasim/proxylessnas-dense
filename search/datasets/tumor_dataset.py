import torch
from torch.utils import data
import os
import numpy as np


class TumorDataset(data.Dataset):

    def __init__(self, ids, save_path):
        self.ids = ids
        self.save_path = save_path

    def __len__(self):
        return len(self.ids)

    def normalize(self, y):
        return (y - torch.mean(y)) / torch.std(y)

    def __getitem__(self, index):
        id = self.ids[index]

        data = np.load(os.path.join(self.save_path, id + '.npz'))
        X = torch.Tensor(data['y'])
        y = torch.Tensor(data['x'][:, :, 61, 0])

        y = self.normalize(y)
        y = torch.unsqueeze(y, 0)

        return X, y
