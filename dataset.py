import torch
import numpy as np
from torch.utils.data import Dataset

class DREAMER(Dataset):
    def __init__(self, trunc_len):
        self.ecg = np.load("dataset/DREAMER/DREAMER_ecg.npy",allow_pickle=True)
        self.label = np.load("dataset/DREAMER/DREAMER_label.npy",allow_pickle=True)
        self.hrv = np.load("dataset/DREAMER/DREAMER_hrv.npy")
        self.trunc_len = trunc_len

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        y_label = torch.from_numpy(self.label[idx])
        x_ecg = torch.from_numpy(self.ecg[idx])
        x_hrv = torch.from_numpy(self.hrv[idx])
        if x_ecg.shape[0] < self.trunc_len:
            # cat
            x_ecg = torch.cat((x_ecg,torch.zeros(self.trunc_len - x_ecg.shape[0],2)),0)
        else:
            # truncate
            s = np.random.randint(x_ecg.shape[0]-self.trunc_len+1)
            x_ecg = x_ecg[s:s+self.trunc_len]
        return (x_ecg, y_label)

