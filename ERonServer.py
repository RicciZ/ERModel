import numpy as np
import pandas as pd
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class ERModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device, use_hrv,
            dropout_cnn = 0.5, dropout_lstm = 0.8, kernel_size_pool = 5, stride_pool = 4):
        super(ERModel, self).__init__()
        self.device = device
        self.use_hrv = use_hrv
        # CNN stream
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.dropout_cnn = nn.Dropout(p=dropout_cnn)
        self.dropout_lstm = nn.Dropout(p=dropout_lstm)
        self.pool = nn.AvgPool1d(kernel_size=kernel_size_pool, stride=stride_pool)
        # BLSTM stream
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # batch x time_seq x features
        self.blstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size*2 + 64*int((input_size-kernel_size_pool)/4+1), 256)
        self.fc2 = nn.Linear(256, 32)
        if self.use_hrv:
            self.fc3 = nn.Linear(36, num_classes)
        else:
            self.fc3 = nn.Linear(32, num_classes)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(32)


    def forward(self, x):
        if self.use_hrv:
            x_hrv = x[:,:,-2:]
            x = x[:,:,:-2]
            # print(x_hrv.shape,x.shape)
        # BLSTM stream
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.blstm(x, (h0, c0))
        out = self.dropout_lstm(out[:, -1, :])

        # CNN stream
        x = F.relu(self.conv1(x))
        x = self.dropout_cnn(x)
        x = F.relu(self.conv2(x))
        x = self.dropout_cnn(x)
        x = F.relu(self.conv3(x))
        x = self.dropout_cnn(x)
        x = F.relu(self.conv4(x))
        x = self.dropout_cnn(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        
        # concatenate and fc
        x = torch.cat((x,out),1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # print(x.shape)
        x = torch.cat((x, x_hrv.reshape(x_hrv.shape[0], -1)),1)
        # print(x.shape)
        x = self.fc3(x)

        return x


class ERonServer:
    def __init__(self):
        # init model
        num_classes = 3
        self.input_size = 60000
        hidden_size = 512
        num_layers = 2
        lr = 0.001
        use_hrv = True
        device = torch.device("cpu")
        self.model = ERModel(self.input_size, hidden_size, num_layers, num_classes, device, use_hrv).to(device)
        self.load_checkpoint(os.path.join('checkpoint/my_checkpoint.pth.tar'))
    

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename, map_location="cpu")
        self.model.load_state_dict(checkpoint['state_dict'])


    def get_emo(self, filename):
        ecg = pd.read_csv(filename, index_col=False)
        x = np.array([i for i in ecg['信号强度'] if i != 0])
        x = (x-np.mean(x))/np.std(x)
        x = torch.tensor([x,x])
        x = torch.cat((x, torch.zeros(2, self.input_size - x.shape[1])),1)
        ibi = np.array([i/256 for i in ecg['IBI'] if i != 0])
        mean = np.mean(ibi)
        std = np.std(ibi)
        x = torch.cat((x, torch.tensor([[mean,std], [mean,std]])),1).unsqueeze(0).float()
        res = self.model(x)
        res = res[0].detach().numpy()
        emo = self.map_score_emo(res)

        return emo

    
    def map_score_emo(self, score):
        # map 3 scores to specific emotion: 
        # calmness 322, happiness 433, fear 244, sadness 133, anger 144, excitement 333
        
        # val positive or negative
        # aro bored or excited
        # dom without control or empowered
        emotion = np.array([[3.17, 2.26, 2.09], [4.39, 3.44, 3.65], [2.26, 3.67, 3.67], 
                        [1.52, 3.00, 3.96], [1.85, 3.09, 3.37], [3.44, 3.53, 3.39]])
        out = np.zeros(6)
        for i in range(len(emotion)):
            temp = (score - emotion[i])**2
            out[i] = 100*(2*temp[0] + temp[1] + temp[2])
        out = max(out)-out+1
        out /= sum(out)
        
        return out

    
    def map_emo_tag(self, emo):
        tag = None
        return tag


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Emotion Recognition on Server')
    parser.add_argument('--ecg_file', type=str, default=None, help='the path of the input ecg file')
    args = parser.parse_args()
    
    model = ERonServer()
    emo = model.get_emo(args.ecg_file)

    # test example
    # emo = model.get_emo("user_ecg/2022_5_18_13_49.csv")

    print(emo)

