# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YXBeCI5bRj_7SixvRG82ZEYt28Y1UBrw
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import argparse
import time
import copy
import os





# input id
id_ = 201676585

# setup training parameters
parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args(args=[])


# judge cuda is available or not
use_cuda = not args.no_cuda and torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cpu")

torch.manual_seed(args.seed)
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

############################################################################
################    don't change the below code    #####################
############################################################################
train_set = torchvision.datasets.FashionMNIST(root='data', train=True, download=True,
                                              transform=transforms.Compose([transforms.ToTensor()]))
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

test_set = torchvision.datasets.FashionMNIST(root='data', train=False, download=True,
                                             transform=transforms.Compose([transforms.ToTensor()]))
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)


# define fully connected network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output


##############################################################################
#############    end of "don't change the below code"   ######################
##############################################################################

# generate adversarial data, you can define your adversarial method
def adv_attack(model, X, y, device):
    
    ################################################################################################
    ## Note: below is the place you need to edit to implement your own attack algorithm
    ################################################################################################
    #LPGDattack
    random_start=True
    epsilon=0.1
    k=3
    a=0.01
    loss_fn = nn.CrossEntropyLoss()
    if random_start:
            X0 = X + np.random.uniform(-epsilon, epsilon,
                X.shape).astype('float32')
    else:
            X0 = np.copy(X)

    for i in range(k):
            X_adv = Variable(X.data,requires_grad=True)
            y_adv= Variable(y)
            with torch.enable_grad():
              scores = model(X_adv)
              loss = loss_fn(scores, y)
            # loss.requires_grad_(True)
            loss.backward()
            grad = X_adv.grad.numpy()
            X0 += a * np.sign(grad)
            X0 = np.clip(X0, X - epsilon, X + epsilon)
            X0 = np.clip(X0, 0, 1) # ensure valid pixel range

    ################################################################################################
    ## end of attack method
    ################################################################################################

    return X0

# train function, you can use adversarial training
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), 28 * 28)

        # use adverserial data to train the defense model
        #adv_data = adv_attack(model, data, target, device=device)
        param = {
            'batch_size': 128,
            'test_batch_size': 100,
            'num_epochs': 10,
            'delay': 10,
            'learning_rate': 1e-3,
            'weight_decay': 5e-4,
        }
        criterion = nn.CrossEntropyLoss()
        adversary=adv_attack(model,data,target,device)
        # print(adversary)
        if epoch + 1 > param['delay']:
            # use predicted label to prevent label leaking
            y_pred =torch.from_numpy(np.argmax(model(Variable(data)).data.cpu().numpy(), axis=1))
            # If adversarial training, need a snapshot of
            # the model at each batch to compute grad, so
            # as not to mess up with the optimization step
            model_cp = copy.deepcopy(model)
            for p in model_cp.parameters():
                p.requires_grad = False
            model_cp.eval()

            adversary.model = model_cp
            # print(model_cp)
            X_adv = adversary.perturb(data.numpy(), y_pred)
            x_adv = torch.from_numpy(X_adv)
            x_adv_var = Variable(x_adv)
            loss_adv = criterion(Net(x_adv_var), target)
            loss = (loss + loss_adv) / 2

        # clear gradients
        optimizer.zero_grad()
        # compute loss
        loss = F.nll_loss(model(adversary), target)
        # get gradients and updat
        loss.backward()
        optimizer.step()


# predict function
def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), 28 * 28)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def eval_adv_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), 28 * 28)
            adv_data = adv_attack(model, data, target, device=device)
            output = model(adv_data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


# main function, train the dataset and print train loss, test loss for each epoch
def train_model():
    model = Net().to(device)

    ################################################################################################
    ## Note: below is the place you need to edit to implement your own training algorithm
    ##       You can also edit the functions such as train(...).
    ################################################################################################
    param = {
        'batch_size': 128,
        'test_batch_size': 100,
        'num_epochs': 10,
        'delay': 10,
        'learning_rate': 1e-3,
        'weight_decay': 5e-4,
    }
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=param['learning_rate'],
                                    weight_decay=param['weight_decay'])
    for epoch in range(param['num_epochs']):
        start_time = time.time()
        print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))
        # training
        train(args, model, device, train_loader, optimizer, epoch)

        # get trnloss and testloss
        trnloss, trnacc = eval_test(model, device, train_loader)
        advloss, advacc = eval_adv_test(model, device, train_loader)

        # print trnloss and testloss
        #print('Epoch ' + str(epoch) + ': ' + str(int(time.time() - start_time)) + 's', end=', ')
        #print('trn_loss: {:.4f}, trn_acc: {:.2f}%'.format(trnloss, 100. * trnacc), end=', ')
        #print('adv_loss: {:.4f}, adv_acc: {:.2f}%'.format(advloss, 100. * advacc))

    adv_tstloss, adv_tstacc = eval_adv_test(model, device, test_loader)
    print('Your estimated attack ability, by applying your attack method on your own trained model, is: {:.4f}'.format(
        1 / adv_tstacc))
    print('Your estimated defence ability, by evaluating your own defence model over your attack, is: {:.4f}'.format(
        adv_tstacc))
    ################################################################################################
    ## end of training method
    ################################################################################################

    # save the model
    torch.save(model.state_dict(), str(id_) + '.pt')
    return model


# compute perturbation distance
def p_distance(model, train_loader, device):
    p = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), 28 * 28)
        data_ = copy.deepcopy(data.data)
        adv_data = adv_attack(model, data, target, device=device)
        p.append(torch.norm(data_ - adv_data, float('inf')))
    print('epsilon p: ', max(p))


################################################################################################
## Note: below is for testing/debugging purpose, please comment them out in the submission file
################################################################################################

#Comment out the following command when you do not want to re-train the model
#In that case, it will load a pre-trained model you saved in train_model()
#do not want to re-train model
#model = train_model()
# Call adv_attack() method on a pre-trained model'
# adv_attack(model,train_loader,test_loader,device)
# the robustness of the model is evaluated against the infinite-norm distance measure
# important: MAKE SURE the infinite-norm distance (epsilon p) less than 0.11 !!!
#p_distance(model, train_loader, device)
#print(p_distance)
#torch.save(model.state_dict(),'201676585.pt')

