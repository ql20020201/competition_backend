import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn. functional as F
from torch.utils.data import Dataset , DataLoader
import torch.optim as optim
import torchvision
from torch._six import container_abcs
from torchvision import transforms
from torch.autograd import Variable
import argparse
import time
import copy

id_ = 201677485

def parse_args():
    parser = argparse. ArgumentParser(description ='PyTorch MNIST Training')
    parser.add_argument ('--batch_size', type=int, default =256 , metavar='N', help='input batch size for training (default:128)')
    parser.add_argument ('--test_batch_size', type=int, default =256 , metavar='N', help='input batch size for testing (default:128)')
    parser.add_argument ('--epochs', type=int, default =100, metavar='N' , help='number of epochs to train')
    parser.add_argument ('--lr', type=float , default =0.1 , metavar='LR', help='learning rate')
    parser.add_argument ('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument ('--seed', type=int, default =1, metavar='S', help='random seed (default: 1)')
    args = parser.parse_args(args =[])
    return args

class Net(nn.Module):
    def __init__(self):
        super(Net , self).__init__ ()
        self.fc1 = nn.Linear (28*28 , 128)
        self.fc2 = nn.Linear (128 , 64)
        self.fc3 = nn.Linear (64, 32)
        self.fc4 = nn.Linear (32, 10)

    def forward(self , x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = F. log_softmax (x, dim =1)
        return output

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, container_abcs.Iterable):
        for elem in x:
            zero_gradients(elem)    

def adv_attack(model , X, y, device, batch_idx=1):
    X_adv = Variable(X.data)
    ############################################ Note: below is
    # the place you need to edit to implement your own attack
    # algorithm
    ############################################
    # random_noise = torch. FloatTensor (* X_adv.shape).uniform_ (-0.1,
    # 0.1).to(device)
    # X_adv = Variable(X_adv.data + random_noise)

    epsilon = 0.11
    x_adv = X.detach() + 0.003 * torch.randn(X.shape).cuda().detach()

    for _ in range(10):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_ce = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss_ce, [x_adv])[0]
        x_adv = x_adv.detach() + 0.025 * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, X - epsilon), X + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    X_adv = x_adv
    ############################################ end of attack
    # method
    ############################################
    return X_adv

def eval_test (model , device , test_loader ):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad ():
        for data , target in test_loader :
            data , target = data.to(device), target.to(device)
            data = data.view(data.size (0) ,28*28)
            output = model(data)
            test_loss += F.nll_loss(output , target , size_average =
            False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item ()
    test_loss /= len( test_loader .dataset)
    test_accuracy = correct / len( test_loader .dataset)
    return test_loss , test_accuracy

def eval_adv_test (model , device , test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad ():
        for data, target in test_loader :
            data , target = data.to(device), target.to(device)
            data = data.view(data.size (0) ,28*28)
            adv_data = adv_attack (model , data , target , device=
            device)
            output = model(adv_data)
            test_loss += F.nll_loss(output , target , size_average =
            False).item ()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item ()
        test_loss /= len( test_loader .dataset)
        test_accuracy = correct / len( test_loader .dataset)
    return test_loss , test_accuracy


def train(args , model , device , train_loader , optimizer , epoch):
    model.train ()
    for batch_idx , (data , target) in enumerate(train_loader):
        data , target = data.to(device), target.to(device)
        data = data.view(data.size (0) ,28*28)
        #use adverserial data to train the defense model
        model.train()

        kl = nn.KLDivLoss(reduction='none')

        adv_data = adv_attack(model , data , target , device=device)
        adv_data = Variable(torch.clamp(adv_data, 0.0, 1.0), requires_grad=False)

        optimizer.zero_grad()

        logits = model(data)

        logits_adv = model(adv_data)

        adv_probs = F.softmax(logits_adv, dim=1)

        tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

        new_tar = torch.where(tmp1[:, -1] == target, tmp1[:, -2], tmp1[:, -1])

        loss_adv = F.cross_entropy(logits_adv, target) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_tar)

        nat_probs = F.softmax(logits, dim=1)

        true_probs = torch.gather(nat_probs, 1, (target.unsqueeze(1)).long()).squeeze()

        loss_robust = (1.0 / len(data)) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    
        loss = loss_adv + 6.0 * loss_robust

        #compute loss

        #get gradients and update
        loss.backward ()
        optimizer.step ()

#main function , train the dataset and print train loss , test loss for each epoch
def train_model ():
    model = Net().to(device)
    #
    ####################################################################
    ## Note: below is the place you need to edit to implement
    # your own training algorithm
    ## You can also edit the functions such as train(...).
    #
    ####################################################################

    # optimizer = optim.SGD(model. parameters (), lr=args.lr)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                              weight_decay=0.0002,
                              momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                            milestones=[int(args.epochs * 0.5),
                                        int(args.epochs * 0.75),
                                        int(args.epochs * 0.9)],
                                    gamma=0.1)
    for epoch in range(1, args.epochs + 1):
        start_time = time.time ()
        #training
        train(args, model, device, train_loader , optimizer , epoch)
        # model.load_state_dict(torch.load('1000.pt'))

        #get trnloss and testloss
        trnloss , trnacc = eval_test (model , device , train_loader)
        advloss , advacc = eval_adv_test (model , device , train_loader)

        #print trnloss and testloss
        print('Epoch '+str(epoch)+': '+str(int(time.time ()-
        start_time ))+'s', end=', ')
        print('trn_loss: {:.4f}, trn_acc: {:.2f}%'.format(trnloss
        , 100. * trnacc), end=', ')
        print('adv_loss: {:.4f}, adv_acc: {:.2f}%'.format(advloss
        , 100. * advacc))

        adv_tstloss , adv_tstacc = eval_adv_test (model, device, test_loader)
        print('Your estimated attack ability , by applying your attack method on your own trained model , is: {:.4f}'.format(1/
        adv_tstacc))
        print('Your estimated defence ability , by evaluating your own defence model over your attack , is: {:.4f}'.format(
        adv_tstacc ))
        scheduler.step()
        ############################################
        ## end of training method
        ############################################
        
        #save the model
        torch.save(model. state_dict (), str(id_)+'.pt')
    return model

def p_distance (model , train_loader , device):
    p = []
    for batch_idx , (data , target) in enumerate( train_loader ):
        data , target = data.to(device), target.to(device)
        data = data.view(data.size (0) ,28*28)
        data_ = copy.deepcopy(data.data)
        adv_data = adv_attack (model , data , target , device=device)
        p.append(torch.norm(data_ -adv_data , float('inf')))
        print('epsilon p: ',max(p))
        

if __name__ == '__main__':
    args = parse_args()
    use_cuda = not args.no_cuda and torch.cuda. is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_set = torchvision.datasets. FashionMNIST (root='data', train=True , download =True , transform = transforms.Compose ([ transforms.ToTensor ()]))
    train_loader = DataLoader (train_set , batch_size =args.batch_size, shuffle=True)
    test_set = torchvision .datasets. FashionMNIST (root='data', train=False , download=True , transform = transforms.Compose([transforms . ToTensor()]))
    test_loader = DataLoader (test_set , batch_size =args.batch_size, shuffle=True)

    model = train_model()
    # p_distance(model, train_loader, device)