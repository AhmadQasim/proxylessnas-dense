import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import itertools

PATH = './dataset/tumor_simul'
TARGET = './dataset/tiny_tumor_simul'

# data = np.load(os.path.join(PATH, "{}_{}_{}.npz".format(0, 5, 1)))

z = 61
patient_id = [i for i in range(1)]
timesteps = [i for i in range(1, 20)]
parameters = [i for i in range(0, 2)]

a = ["{}_{}_{}".format(x[0], x[1], x[2]) for x in itertools.product(patient_id, parameters, timesteps)]

print(a)

exit(1)

X = []
y = []

for i in patient_id:
    for j in parameters:
        for k in timesteps:
            data = np.load(os.path.join(PATH, "{}_{}_{}.npz".format(i, j, k)))
            X.append(data['x'][:, :, 61, 0])
            y.append(data['y'])

X = torch.Tensor(X)
y = torch.Tensor(y)

X = (X - torch.mean(X))/torch.std(X)

print(X.shape, y.shape)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=1)

for batch_ndx, sample in enumerate(loader):
    print(sample.X.shape)
    print(sample.tgt.shape)
