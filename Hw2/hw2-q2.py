#!/usr/bin/env python

# Deep Learning Homework 2

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
import numpy as np

import utils

class CNN(nn.Module):
    def __init__(self, dropout_prob, no_maxpool=False):
        super(CNN, self).__init__()
        self.no_maxpool = no_maxpool
        if not no_maxpool:
            
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
            self.relu = nn.ReLU()
            self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0)
            self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
             
        else:
            
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0)
            
                    
        self.fc1_input_features = 16 * 6 * 6  
        
        
        self.fc1 = nn.Linear(self.fc1_input_features, 320)
        self.fc1_relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(320, 120)
        self.fc2_relu = nn.ReLU()
        self.fc3 = nn.Linear(120, 10)
        
    def forward(self, x, no_maxpool=False):
        
        x = self.conv1(x)
        x = self.relu(x)
        
        if not no_maxpool and hasattr(self, 'maxpool1'):
           
            x = self.maxpool1(x)

        
        x = self.conv2(x)
        x = self.relu(x)

        if not no_maxpool and hasattr(self, 'maxpool2'):
            
            x = self.maxpool2(x)

        
        x = x.view(-1, self.fc1_input_features)
        x = self.fc1(x)
        x = self.fc1_relu(x)
        
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.fc2_relu(x)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)



def train_batch(X, y, model, optimizer, criterion, **kwargs):
    optimizer.zero_grad()
    X = torch.reshape(X, (X.size()[0], 1, 28, 28))
    out = model(X, no_maxpool=kwargs.get('no_maxpool', False))
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, X):
    scores = model(X)
    predicted_labels = scores.argmax(dim=-1)
    return predicted_labels

def evaluate(model, X, y):
    model.eval()
    y_hat = predict(model, X)
    n_correct = (y == y_hat).sum().item()
    n_possible = float(y.shape[0])
    model.train()
    return n_correct / n_possible

def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')

def get_number_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    '''return 0'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=8, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.01,
                        help="""Learning rate for parameter updates""")
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-dropout', type=float, default=0.7)
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-no_maxpool', action='store_true')
    
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    data = utils.load_oct_data()
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True)
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    # Configuration based on Q2.1 or Q2.2
    if not opt.no_maxpool:
        # Q2.1 Configuration
        model = CNN(opt.dropout, no_maxpool=False)
    else:
        # Q2.2 Configuration
        model = CNN(opt.dropout, no_maxpool=True)
    
    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay
    )
    
    # get a loss criterion
    criterion = nn.NLLLoss()
    
    # training loop
    epochs = np.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_accs = []
    train_losses = []
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(X_batch, y_batch, model, optimizer, criterion, no_maxpool=opt.no_maxpool)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        valid_accs.append(evaluate(model, dev_X.reshape(-1, 1, 28, 28), dev_y))  # Reshape dev_X
        print('Valid acc: %.4f' % (valid_accs[-1]))

    print('Final Test acc: %.4f' % (evaluate(model, test_X.reshape(-1, 1, 28, 28), test_y)))  # Reshape test_X
    # plot
    config = "{}-{}-{}-{}-{}".format(opt.learning_rate, opt.dropout, opt.l2_decay, opt.optimizer, opt.no_maxpool)

    plot(epochs, train_mean_losses, ylabel='Loss', name='CNN-training-loss-{}'.format(config))
    plot(epochs, valid_accs, ylabel='Accuracy', name='CNN-validation-accuracy-{}'.format(config))
    
    print('Number of trainable parameters: ', get_number_trainable_params(model))

if __name__ == '__main__':
    main()
