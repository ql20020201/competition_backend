default_app_config = 'submittion.apps.SubmittionConfig'

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import argparse


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
device = torch.device("cuda" if use_cuda else "cpu")
# device = torch.device("cpu")

torch.manual_seed(args.seed)
test_set = torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)


import torch.nn as nn
import torch.nn.functional as F

import importlib

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

# 评估函数
def eval_adv_test(model, device, test_loader, adv_attack_method):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device) # 将数据移动到设定的设备上。
            data = data.view(data.size(0),28*28)
            adv_data = adv_attack_method(model, data, target, device=device)  # 生成对抗样本。 Get adversarial samples
            output = model(adv_data) # 对对抗样本进行预测。 Predict adversarial samples.
            test_loss += F.nll_loss(output, target, size_average=False).item() # 计算损失。
            pred = output.max(1, keepdim=True)[1] # 获取预测结果。 Get prediction results.
            correct += pred.eq(target.view_as(pred)).sum().item() # 计算正确预测的数量。 Calculate the number of correct predictions.
    test_loss /= len(test_loader.dataset)  # 计算平均损失
    test_accuracy = correct / len(test_loader.dataset) # 计算准确率。 Calculate accuracy.
    return test_loss, test_accuracy

# 评估模型
def evaluate_all_models(model_file, attack_method, test_loader, device):

    model = Net().to(device)   # 加载模型并移动到设备。
    model.load_state_dict(torch.load(model_file))  # 加载模型权重。

    adv_attack = attack_method
    ls, acc = eval_adv_test(model, device, test_loader, adv_attack)  # 调用评估函数。 Call evaluation function
     
    del model
    return 1/acc, acc

def attack(atkpy, defmodel):
    atk = importlib.import_module('data.codes.%s' % (atkpy))  # 动态加载攻击脚本
    attack_score, defence_score = evaluate_all_models('data/models/%s.pt' % defmodel, atk.adv_attack, test_loader, device) # 动态加载攻击脚本
    return attack_score, defence_score


class caled:
    cal = False

calculated = caled()