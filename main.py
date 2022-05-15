import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from dataset import DREAMER
from ERModel import ERModel


# Load data
# test model using MNIST
# train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
# test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

def collate_fn(data):
    x_ecg = [i[0] for i in data]
    y_label = torch.stack([i[1] for i in data])
    x_ecg.sort(key=lambda x: len(x), reverse=True)
    x_ecg = torch.nn.utils.rnn.pad_sequence(x_ecg, batch_first=True, padding_value=0)
    return x_ecg, y_label

def save_checkpoint(state, filename):
    print("=> Saving checkpoint")
    torch.save(state, filename)

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
        return (self.tp+self.tn)/self.count
    
    
def train(args):
    # init network
    model = ERModel(args.input_size, args.hidden_size, args.num_layers, num_classes, device).to(device)

    # loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # load model
    # if args.load_model:
    #     load_checkpoint(os.path.join('checkpoint', args.exp_name, 'my_checkpoint.pth.tar'))

    # train
    for epoch in range(args.num_epochs):
        losses = []
        
        if epoch % 10 == 0:
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_checkpoint(checkpoint, os.path.join('checkpoint/my_checkpoint.pth.tar'))

        for batch_idx, (data, targets) in enumerate(train_loader):
            # transfer data to gpu
            data = data.permute(0,2,1).float().to(device)
            targets = targets.float().to(device)

            # forward
            y_hat = model(data)
            loss = criterion(y_hat, targets)
            losses.append(loss.item())

            # backward
            optimizer.zero_grad() # set gradient to 0 for each batch
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

        if epoch % 10 == 0:
            mean_loss = sum(losses)/len(losses)
            print(f"Loss at epoch {epoch} was {mean_loss:.5f}")
            res_train.append(mean_loss)

    return model

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
        print(f"mean mse: {mse_sum/num_samples:.2f}")
    res_test["accuracy"] = (metric_val.acc(), metric_aro.acc(), metric_dom.acc())
    res_test["f1_score"] = (metric_val.f1(), metric_aro.f1(), metric_dom.f1())
    res_test["mean_mse"] = mse_sum/num_samples


    model.train() # set back to train mode


if __name__ == "__main__":
    # Parameters
    num_classes = 3
    
    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training settings
    parser = argparse.ArgumentParser(description='Emotion Recognition')
    parser.add_argument('--exp_name', type=str, default=None, help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--input_size', type=int, default=500, help='input size')
    parser.add_argument('--num_layers', type=int, default=2, help='num of layers for blstm')
    parser.add_argument('--hidden_size', type=int, default=512, help='hidden size for blstm')
    parser.add_argument('--num_epochs', type=int, default=2000, help='number of epochs')
    # parser.add_argument('--load_model', type=bool, default=False, help='load model or not')
    parser.add_argument('--lr', type=int, default=0.001, help='learning rate')\

    args = parser.parse_args()
    args.exp_name = '_'.join([f'{k}[{v}]' for k, v in args.__dict__.items() if v != None])

    # init dataset
    dataset = DREAMER(args.input_size)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [380, 34])
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)
    # test the dataset
    # for batch_idx, (data, targets) in enumerate(train_loader):
    #     print("test the dataset loading process")
    #     print(data.shape)
    #     print(targets.shape)
    #     exit()
    
    # train
    res_train = []
    model = train(args)
    
    # test and output the results
    filename = "results/" + args.exp_name + ".txt"
    with open(filename, "w") as f:
        f.write(str(res_train)+"\n")
        res_test = {}
        check_accuracy(train_loader, model, train=True)
        f.write(str(res_test)+"\n")
        check_accuracy(test_loader, model, train=False)
        f.write(str(res_test)+"\n")


