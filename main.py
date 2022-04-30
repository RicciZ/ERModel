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

batch_size = 32

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

def save_checkpoint(state, filename="checkpoint/my_checkpoint2.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def dataset_init(trunc_len):
    dataset = DREAMER(trunc_len)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [380, 34])
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    # test the dataset
    # for batch_idx, (data, targets) in enumerate(train_loader):
    #     print("test the dataset loading process")
    #     print(data.shape)
    #     print(targets.shape)
    #     exit()


def train(args):
    # init dataset
    dataset_init(args.input_size)

    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init network
    model = ERModel(args.input_size, args.hidden_size, args.num_layers, num_classes).to(device)

    # loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # load model
    if args.load_model:
        load_checkpoint(torch.load("checkpoint/my_checkpoint.pth.tar"))

    # train
    for epoch in range(args.num_epochs):
        losses = []
        
        if epoch % 10 == 0:
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_checkpoint(checkpoint)

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

    return model

# check accuracy on training
def check_accuracy(loader, model, train=True):
    if train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on testing data")

    mse_sum = 0
    num_samples = 0
    model.eval() # set model to evaluation mode

    # don't compute the gradient in this step
    with torch.no_grad(): 
        for x, y in loader:
            # x = x.to(device).squeeze(1)
            x = x.permute(0,2,1).float().to(device)
            y = y.float().to(device)
            y_hat = model(x)
            mse_sum += torch.sum(torch.mean((y_hat-y)**2,1))
            num_samples += y_hat.size(0)

        print(f"Got mean mse {mse_sum/num_samples:.2f}")

    model.train() # set back to train mode


if __name__ == "__main__":
    # Parameters
    num_classes = 3

    # Training settings
    parser = argparse.ArgumentParser(description='Emotion Recognition')
    parser.add_argument('--exp_name', type=str, default=None, help='Name of the experiment')
    parser.add_argument('--input_size', type=int, default=60000, help='input size')
    parser.add_argument('--num_layers', type=int, default=2, help='num of layers for blstm')
    parser.add_argument('--hidden_size', type=int, default=512, help='hidden size for blstm')
    parser.add_argument('--num_epochs', type=int, default=2000, help='number of epochs')
    parser.add_argument('--load_model', type=bool, default=False, help='load model or not')
    parser.add_argument('--lr', type=int, default=0.001, help='learning rate')

    args = parser.parse_args()

    model = train(args)

    check_accuracy(train_loader, model, train=True)
    check_accuracy(test_loader, model, train=False)


