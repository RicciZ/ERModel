import torch
import torch.nn as nn
import torch.nn.functional as F

class ERModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_cnn = 0.5, dropout_lstm = 0.8,
                    kernel_size_pool = 5, stride_pool = 4):
        super(ERModel, self).__init__()
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
        self.fc3 = nn.Linear(32, num_classes)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(32)


    def forward(self, x):
        # BLSTM stream
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
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
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        return x

# test the dim of the model
# model = ERModel(input_size, hidden_size, num_layers, num_classes).to(device)
# x = torch.randn(32, 2, 10000).to(device)
# print(model(x).shape)
# exit()
