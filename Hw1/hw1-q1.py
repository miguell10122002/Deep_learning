#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        y_hat = self.predict(x_i.reshape(1,-1))

       
        if y_hat != y_i:
            self.W[y_i] += x_i
            self.W[y_hat] -= x_i

class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        
        scores = np.dot(self.W, x_i)

       
        exp_scores = np.exp(scores)
        prob = exp_scores / np.sum(exp_scores)
        
        
        
       
                
        prob[y_i] -= 1
        self.W = self.W - learning_rate * np.outer(prob, x_i)


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size): 
        self.hidden_size = hidden_size
        self.W1 = np.random.normal(0.1,0.1, size=(hidden_size, n_features))
        self.b1 = np.zeros((hidden_size,))
        self.W2 = np.random.normal(0.1,0.1, size=(n_classes, hidden_size))
        self.b2 = np.zeros((n_classes,))

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        
        return np.exp(x - np.max(x)) / sum(np.exp(x - np.max(x)))

    def predict(self, X, train = False):
        self.hidden_input = np.dot(self.W1, X.T) + self.b1.reshape(-1, 1)
        self.hidden_output = self.relu(self.hidden_input)
        self.output_scores = np.dot(self.W2, self.hidden_output) + self.b2.reshape(-1, 1)
        self.output_probs = self.softmax(self.output_scores)
        return self.output_probs if train == True else self.output_probs.argmax(axis = 0)

    def update_weights(self, x_i, y_i, learning_rate):
        m = x_i.shape[0]
        
        
        self.predict(x_i, train=True)

       
        doutput = self.output_probs.copy()
        doutput[y_i, np.arange(m)] -= 1

        dW2 = np.dot(doutput, self.hidden_output.T) / m
        db2 = np.sum(doutput, axis=1, keepdims=True) / m

        
        dhidden = np.dot(self.W2.T, doutput)
        dhidden[self.hidden_input <= 0] = 0 

        dW1 = np.dot(dhidden, x_i) / m
        db1 = np.sum(dhidden, axis=1, keepdims=True) / m

        
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2.squeeze()
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1.squeeze()
        
    def train_epoch(self, X, y, learning_rate=0.001):
        m = X.shape[0]
       
       

        total_loss = 0.0
        for i in range(m):
            x_i = X[i]
            y_i = y[i]
            x_i = x_i.reshape(1, -1)

            output_probs = self.predict(x_i, train=True)
            loss = -np.log(output_probs[y_i, 0])
            total_loss += loss

            self.update_weights(x_i, y_i, learning_rate)

        return total_loss / m
    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible
    
    
        


def plot(epochs, train_accs, val_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    plt.show()

def plot_loss(epochs, loss):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_oct_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    valid_accs = []
    train_accs = []
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs)
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss)


if __name__ == '__main__':
    main()
