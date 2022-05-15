import torch
import numpy as np
from torch.utils.data import Dataset

class DREAMER(Dataset):
    def __init__(self, trunc_len):
        # self.ecg = np.load("dataset/DREAMER/DREAMER_ecg.npy", allow_pickle=True)
        self.label = np.load("dataset/DREAMER/DREAMER_label.npy", allow_pickle=True)
        self.ibi = np.load("dataset/DREAMER/DREAMER_ibi.npy", allow_pickle=True)
        self.ibi_stat = np.load("dataset/DREAMER/DREAMER_ibi_stat.npy")
        self.trunc_len = trunc_len

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        y_label = torch.from_numpy(self.label[idx])
        # x_ecg = torch.from_numpy(self.ecg[idx])
        x_ibi = torch.from_numpy(self.ibi[idx])
        x_ibi_stat = torch.from_numpy(self.ibi_stat[idx]) # (mean(ibi1),std(ibi1),mean(ibi2),std(ibi2))
        if x_ibi.shape[0] < self.trunc_len:
            # cat
            x_ibi = torch.cat((x_ibi, torch.zeros(self.trunc_len - x_ibi.shape[0],2)),0)
        else:
            # truncate
            s = np.random.randint(x_ibi.shape[0]-self.trunc_len+1)
            x_ibi = x_ibi[s:s+self.trunc_len]
        x_ibi = torch.cat((x_ibi, torch.tensor([[x_ibi_stat[0],x_ibi_stat[2]], [x_ibi_stat[1],x_ibi_stat[3]]])),0)
        return (x_ibi, y_label)

