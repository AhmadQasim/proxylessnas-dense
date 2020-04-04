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
timesteps = [i for i in range(1, 21)]
parameters = [i for i in range(0, 100)]

# a = ["{}_{}_{}".format(x[0], x[1], x[2]) for x in itertools.product(patient_id, parameters, timesteps)]


X = []
y = []

sum_1 = 0
sum_2 = 0
N = 0

for i in patient_id:
    for j in parameters:
        for k in timesteps:
            print("Reading: ", "{}_{}_{}.npz".format(i, j, k))
            data = np.load(os.path.join(PATH, "{}_{}_{}.npz".format(i, j, k)))
            data = data['x'][:, :, :, 0]
            sum = np.sum(data)

            sum_1 += sum
            sum_2 += np.square(sum)
            N += np.prod(data.shape)
            print(sum_1, sum_2, N)

mean = sum_1 / N

print("Mean: ", mean)
print("Standard Deviation: ", np.sqrt((sum_2 / N) - np.square(mean)))
