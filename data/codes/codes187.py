
# First of all, the following code piece imports a few packages that are needed.
############################################################################
### Written by Gaojie Jin and updated by Xiaowei Huang, 2021
###
### For a 2-nd year undergraduate student competition on 
### the robustness of deep neural networks, where a student 
### needs to develop 
### 1. an attack algorithm, and 
### 2. an adversarial training algorithm
###
### The score is based on both algorithms. 
############################################################################


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

# %% [markdown]
# The below line of code defines the student number. By replacing it with your own
# student number, it will automatically output the file 𝑖.pt once you trained a model.

# %%
# input id
id_ = 1000

# %%
# setup training parameters
parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=18, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.015, metavar='LR',
                    help='learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')


args = parser.parse_args(args=[]) 

# judge cuda is available or not
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#device = torch.device("cpu")

torch.manual_seed(args.seed)
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# %%
############################################################################
################    don't change the below code    #####################
############################################################################

train_set = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

test_set = torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

# define fully connected network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
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

# %%
#generate adversarial data, you can define your adversarial method

class GenerateNet(nn.Module):
    def __init__(self):
        super(GenerateNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 56*56)
        self.fc2 = nn.Linear(56*56, 32*32)
        self.fc3 = nn.Linear(32*32, 28*28)
        self.fc4 = nn.Linear(28*28, 28*28)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        output = F.tanh( self.fc4(x))  #使分布在-1-1之间
        return output

def adv_attack(model, X, y, device,method = None):
    X_adv = Variable(X.data)
    
    ################################################################################################
    ## Note: below is the place you need to edit to implement your own attack algorithm
    ################################################################################################
    
    #random_noise = model(X.data) 
    random_noise = torch.FloatTensor(28*28).uniform_(-1, 1).to(device)
    #X_adv = Variable(X_adv.data + 0.1*model(random_noise) )
    #X_adv = Variable(X_adv.data + random_noise)
    if method == 'detach':
        #X_adv = model(X_adv).detach()
        X_adv = Variable(X_adv.data + 0.12*model(random_noise) )
    else:
        #X_adv = model(X_adv)
        X_adv = Variable(X_adv.data + 0.12*model(random_noise) )
    ################################################################################################
    ## end of attack method
    ################################################################################################
    
    return X_adv

# %%
#train function, you can use adversarial training
def train(args, model,gen_model, device, train_loader, optimizer,gen_optimizer, epoch):
    gen_model =   gen_model
    model.train()
    gen_model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0),28*28)
        adv_data = adv_attack(gen_model, data, target, device=device,method='detach') 
        #clear gradients
        optimizer.zero_grad() 
        
        #compute loss
        loss_real = F.nll_loss(model(data), target)
        loss_fake = F.nll_loss(model(adv_data), target)
        
        loss = (0.6*loss_real + 0.4*loss_fake)
        #get gradients and update
        loss.backward()
        optimizer.step()


        gen_optimizer.zero_grad()
        #use adverserial data to train the defense model
        adv_data = adv_attack(gen_model, data, target, device=device)  
        #compute loss
        #这里不用detach 因为生成网络在前面
        
        gen_loss = F.nll_loss(model(adv_data), target)
        #clear gradients
        #optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()

        '''
        #repeat again loss
        optimizer.zero_grad()
        loss = F.nll_loss(model(adv_data), target)
        loss.backward()
        optimizer.step()
        '''

        
#predict function
def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0),28*28)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def eval_adv_test(model,gen_model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0),28*28)
            adv_data = adv_attack(gen_model, data, target, device=device)
            output = model(adv_data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

# %%
#main function, train the dataset and print train loss, test loss for each epoch
def train_model():
    model = Net().to(device)
    gen_model = GenerateNet().to(device)
    ################################################################################################
    ## Note: below is the place you need to edit to implement your own training algorithm
    ##       You can also edit the functions such as train(...). 
    ################################################################################################
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    gen_optimizer = optim.SGD(gen_model.parameters(), lr=args.lr )
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        #training
        train(args, model= model, gen_model=gen_model,device= device, train_loader= train_loader,optimizer= optimizer, gen_optimizer=gen_optimizer,epoch= epoch)
        
        #get trnloss and testloss
        trnloss, trnacc = eval_test(model, device, train_loader)
        advloss, advacc = eval_adv_test(model,gen_model, device, train_loader)
        
        #print trnloss and testloss
        print('Epoch '+str(epoch)+': '+str(int(time.time()-start_time))+'s', end=', ')
        print('trn_loss: {:.4f}, trn_acc: {:.2f}%'.format(trnloss, 100. * trnacc), end=', ')
        print('adv_loss: {:.4f}, adv_acc: {:.2f}%'.format(advloss, 100. * advacc))
        
    adv_tstloss, adv_tstacc = eval_adv_test(model,gen_model, device, train_loader)
    print('Your estimated attack ability, by applying your attack method on your own trained model, is: {:.4f}'.format(1/adv_tstacc))
    print('Your estimated defence ability, by evaluating your own defence model over your attack, is: {:.4f}'.format(adv_tstacc))
    ################################################################################################
    ## end of training method
    ################################################################################################
    
    #save the model
    torch.save(model.state_dict(), str(id_)+'.pt')
    return model,gen_model

#compute perturbation distance
def p_distance(model,gen_model, train_loader, device):
    p = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0),28*28)
        data_ = copy.deepcopy(data.data)
        adv_data = adv_attack(gen_model, data, target, device=device)
        p.append(torch.norm(data_-adv_data, float('inf')))
    print('epsilon p: ',max(p))

    
################################################################################################
## Note: below is for testing/debugging purpose, please comment them out in the submission file
################################################################################################
    
#Comment out the following command when you do not want to re-train the model
#In that case, it will load a pre-trained model you saved in train_model()
model,gen_#model = train_model()

#Call adv_attack() method on a pre-trained model'
#the robustness of the model is evaluated against the infinite-norm distance measure
#important: MAKE SURE the infinite-norm distance (epsilon p) less than 0.11 !!!
p_distance(model,gen_model, train_loader, device)


