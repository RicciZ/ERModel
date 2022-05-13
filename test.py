import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DREAMER
from ERModel import ERModel


def load_checkpoint(filename):
    print("=> Loading checkpoint")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


class Metric(object):
    def __init__(self):
        self.count = 0
        self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0
    
    def update(self, y, y_):
        y = (y >= 3).long()
        y_ = (y_ >= 3).long()
        self.tp += sum(y_[y == 1] == 1)
        self.tn += sum(y_[y == 0] == 0)
        self.fp += sum(y_[y == 0] == 1)
        self.fn += sum(y_[y == 1] == 0)
        self.count += len(y)
    
    def printinfo(self):
        print(f"tp={self.tp}, tn={self.tn}, fp={self.fp}, fn={self.fn}, count={self.count}")

    def f1(self):
        precision = self.tp/(self.tp + self.fp + 0.001)
        recall = self.tp/(self.tp + self.fn + 0.001)
        return 2*precision*recall / (precision + recall + 0.001)
    
    def acc(self):
        return self.tp/self.count


def check_accuracy(loader, model, train=True):
    if train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on testing data")

    metric_val = Metric()
    metric_aro = Metric()
    metric_dom = Metric()
    mse_sum = 0
    num_samples = 0
    model.eval() # set model to evaluation mode

    # don't compute the gradient in this step
    with torch.no_grad(): 
        for x, y in loader:
            x = x.permute(0,2,1).float().to(device)
            y = y.float().to(device)
            y_hat = model(x)
            metric_val.update(y[:,0], y_hat[:,0])
            metric_aro.update(y[:,1], y_hat[:,1])
            metric_dom.update(y[:,2], y_hat[:,2])
            mse_sum += torch.sum(torch.mean((y_hat-y)**2,1))
            num_samples += y_hat.shape[0]
            # metric_val.printinfo()
            # metric_aro.printinfo()
            # metric_dom.printinfo()
            
        print(f"accuracy: {(metric_val.acc(), metric_aro.acc(), metric_dom.acc())}")
        print(f"f1 score: {(metric_val.f1(), metric_aro.f1(), metric_dom.f1())}")
        print(f"Got mean mse {mse_sum/num_samples:.2f}")

    model.train() # set back to train mode

if __name__ == "__main__":
    # model
    num_classes = 3
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    input_size = 60000
    hidden_size = 512
    num_layers = 2
    lr = 0.001
    use_hrv = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ERModel(input_size, hidden_size, num_layers, num_classes, device, use_hrv).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    

    # dataset
    batch_size = 32
    dataset = DREAMER(input_size, use_hrv)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [380, 34])
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


    # load parameters for model
    load_checkpoint(os.path.join('checkpoint/my_checkpoint.pth.tar'))


    # check accuracy
    check_accuracy(test_loader, model, train=False)




