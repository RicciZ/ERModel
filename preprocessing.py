from scipy.io import loadmat
import numpy as np
import torch
import torch.nn as nn


dreamer_ecg = loadmat('dataset/DREAMER/DREAMER_ecg.mat',squeeze_me=True)['ecg']
dreamer_label = loadmat('dataset/DREAMER/DREAMER_label.mat',squeeze_me=True)['label']


## get heart rate variability stat
n = len(dreamer_ecg)
ibi_stat = np.zeros((n,4))
dreamer_ibi = []

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
    ibi1 = [peak1[i]-peak1[i-1] for i in range(1,len(peak1))]
    ibi2 = [peak2[i]-peak2[i-1] for i in range(1,len(peak2))]
    l = min(len(ibi1), len(ibi2))
    ibi1 = [ibi1[i] if 0.5<ibi1[i]<1.2 else (ibi1[i-1]+ibi1[i]+ibi1[(i+1)%l])/3 for i in range(l)]
    ibi2 = [ibi2[i] if 0.5<ibi2[i]<1.2 else (ibi2[i-1]+ibi2[i]+ibi2[(i+1)%l])/3 for i in range(l)]
    ibi_stat[t] = np.mean(ibi1),np.std(ibi1),np.mean(ibi2),np.std(ibi2)
    dreamer_ibi.append(np.array([ibi1,ibi2]).T)
    dreamer_ecg[t] = (x-np.mean(x))/np.std(x)

np.save("dataset/DREAMER/DREAMER_ibi_stat.npy",ibi_stat)
np.save("dataset/DREAMER/DREAMER_ibi.npy",dreamer_ibi)
np.save("dataset/DREAMER/DREAMER_ecg.npy",dreamer_ecg)
np.save("dataset/DREAMER/DREAMER_label.npy",dreamer_label)
