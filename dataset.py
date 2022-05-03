import torch
import numpy as np
from torch.utils.data import Dataset

class DREAMER(Dataset):
    def __init__(self, trunc_len, use_hrv=False):
        self.ecg = np.load("dataset/DREAMER/DREAMER_ecg.npy", allow_pickle=True)
        self.label = np.load("dataset/DREAMER/DREAMER_label.npy", allow_pickle=True)
        self.hrv = np.load("dataset/DREAMER/DREAMER_hrv.npy")
        self.trunc_len = trunc_len
        self.use_hrv = use_hrv

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        y_label = torch.from_numpy(self.label[idx])
        x_ecg = torch.from_numpy(self.ecg[idx])
        x_hrv = torch.from_numpy(self.hrv[idx]) # (mean(hrv1),std(hrv1),mean(hrv2),std(hrv2))
        if x_ecg.shape[0] < self.trunc_len:
            # cat
            x_ecg = torch.cat((x_ecg, torch.zeros(self.trunc_len - x_ecg.shape[0],2)),0)
        else:
            # truncate
            s = np.random.randint(x_ecg.shape[0]-self.trunc_len+1)
            x_ecg = x_ecg[s:s+self.trunc_len]
        if self.use_hrv:
            x_ecg = torch.cat((x_ecg, torch.tensor([[x_hrv[0],x_hrv[2]], [x_hrv[1],x_hrv[3]]])),0)
        return (x_ecg, y_label)

