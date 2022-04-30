from scipy.io import loadmat
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


dreamer_ecg = loadmat('dataset/DREAMER/DREAMER_ecg.mat',squeeze_me=True)['ecg']
dreamer_label = loadmat('dataset/DREAMER/DREAMER_label.mat',squeeze_me=True)['label']


## get heart rate variability stat
n = len(dreamer_ecg)
hrv_stat = np.zeros((n,4))
for t in range(n):
    x = dreamer_ecg[t]
    x_max = np.min([np.max(x,0),[2500,2500]],0)
    x_min = np.max([np.min(x,0),[1800,1800]],0)
    th = x_max - 2/5 * (x_max - x_min)
    peak1 = []
    peak2 = []
    for i in range(1,x.shape[0]-1):
        if (x[i,0] > x[i-1,0] and x[i,0] > x[i+1,0] and x[i,0] > th[0]):
            peak1.append(i/256)
        if (x[i,1] > x[i-1,1] and x[i,1] > x[i+1,1] and x[i,1] > th[1]):
            peak2.append(i/256)
    hrv1 = np.array([peak1[i]-peak1[i-1] for i in range(1,len(peak1))])
    hrv1 = hrv1[(hrv1>0.2) & (hrv1<1.5)]
    hrv2 = np.array([peak2[i]-peak2[i-1] for i in range(1,len(peak2))])
    hrv2 = hrv2[(hrv2>0.2) & (hrv2<1.5)]
    hrv_stat[t] = np.mean(hrv1),np.std(hrv1),np.mean(hrv2),np.std(hrv2)
    # print(t,np.mean(hrv1),np.std(hrv1),np.mean(hrv2),np.std(hrv2))
    dreamer_ecg[t] = (x-np.mean(x))/np.std(x)

np.save("dataset/DREAMER/DREAMER_hrv.npy",hrv_stat)
np.save("dataset/DREAMER/DREAMER_ecg.npy",dreamer_ecg)
np.save("dataset/DREAMER/DREAMER_label.npy",dreamer_label)
