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
from absl import app, flags

# input id
id_ = 201676661
FLAGS = flags.FLAGS
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
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')

args = parser.parse_args(args=[])

# judge cuda is available or not
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# device = torch.device("cpu")
print(f"device: {device}")

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
    X_adv = Variable(X.data)
    ################################################################################################
    ## Note: below is the place you need to edit to implement your own attack algorithm
    ################################################################################################
    X = Variable(X.data, requires_grad=True)

    def clip_eta(eta, norm, eps):
        """
        PyTorch implementation of the clip_eta in utils_tf.
        :param eta: Tensor
        :param norm: np.inf, 1, or 2
        :param eps: float
        """
        if norm not in [np.inf, 1, 2]:
            raise ValueError("norm must be np.inf, 1, or 2.")

        avoid_zero_div = torch.tensor(1e-12, dtype=eta.dtype, device=eta.device)
        reduc_ind = list(range(1, len(eta.size())))
        if norm == np.inf:
            eta = torch.clamp(eta, -eps, eps)
        else:
            if norm == 1:
                raise NotImplementedError("L1 clip is not implemented.")
                norm = torch.max(
                    avoid_zero_div, torch.sum(torch.abs(eta), dim=reduc_ind, keepdim=True)
                )
            elif norm == 2:
                norm = torch.sqrt(
                    torch.max(
                        avoid_zero_div, torch.sum(eta ** 2, dim=reduc_ind, keepdim=True)
                    )
                )
            factor = torch.min(
                torch.tensor(1.0, dtype=eta.dtype, device=eta.device), eps / norm
            )
            eta *= factor
        return eta

    def optimize_linear(grad, eps, norm=np.inf):
        """
        Solves for the optimal input to a linear function under a norm constraint.
        Optimal_perturbation = argmax_{eta, ||eta||_{norm} < eps} dot(eta, grad)
        :param grad: Tensor, shape (N, d_1, ...). Batch of gradients
        :param eps: float. Scalar specifying size of constraint region
        :param norm: np.inf, 1, or 2. Order of norm constraint.
        :returns: Tensor, shape (N, d_1, ...). Optimal perturbation
        """

        red_ind = list(range(1, len(grad.size())))
        avoid_zero_div = torch.tensor(1e-12, dtype=grad.dtype, device=grad.device)
        if norm == np.inf:
            # Take sign of gradient
            optimal_perturbation = torch.sign(grad)
        elif norm == 1:
            abs_grad = torch.abs(grad)
            sign = torch.sign(grad)
            red_ind = list(range(1, len(grad.size())))
            abs_grad = torch.abs(grad)
            ori_shape = [1] * len(grad.size())
            ori_shape[0] = grad.size(0)

            max_abs_grad, _ = torch.max(abs_grad.view(grad.size(0), -1), 1)
            max_mask = abs_grad.eq(max_abs_grad.view(ori_shape)).to(torch.float)
            num_ties = max_mask
            for red_scalar in red_ind:
                num_ties = torch.sum(num_ties, red_scalar, keepdim=True)
            optimal_perturbation = sign * max_mask / num_ties
            # TODO integrate below to a test file
            # check that the optimal perturbations have been correctly computed
            opt_pert_norm = optimal_perturbation.abs().sum(dim=red_ind)
            assert torch.all(opt_pert_norm == torch.ones_like(opt_pert_norm))
        elif norm == 2:
            square = torch.max(avoid_zero_div, torch.sum(grad ** 2, red_ind, keepdim=True))
            optimal_perturbation = grad / torch.sqrt(square)
            # TODO integrate below to a test file
            # check that the optimal perturbations have been correctly computed
            opt_pert_norm = (
                optimal_perturbation.pow(2).sum(dim=red_ind, keepdim=True).sqrt()
            )
            one_mask = (square <= avoid_zero_div).to(torch.float) * opt_pert_norm + (
                    square > avoid_zero_div
            ).to(torch.float)
            assert torch.allclose(opt_pert_norm, one_mask, rtol=1e-05, atol=1e-08)
        else:
            raise NotImplementedError(
                "Only L-inf, L1 and L2 norms are " "currently implemented."
            )

        # Scale perturbation to be the solution for the norm=eps rather than
        # norm=1 problem
        scaled_perturbation = eps * optimal_perturbation
        return scaled_perturbation

    def fast_gradient_method(
            model_fn,
            x,
            eps,
            norm,
            clip_min=None,
            clip_max=None,
            y=None,
            targeted=False,
            sanity_checks=False,
    ):
        """
        PyTorch implementation of the Fast Gradient Method.
        :param model_fn: a callable that takes an input tensor and returns the model logits.
        :param x: input tensor.
        :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
        :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
        :param clip_min: (optional) float. Minimum float value for adversarial example components.
        :param clip_max: (optional) float. Maximum float value for adversarial example components.
        :param y: (optional) Tensor with true labels. If targeted is true, then provide the
                  target label. Otherwise, only provide this parameter if you'd like to use true
                  labels when crafting adversarial samples. Otherwise, model predictions are used
                  as labels to avoid the "label leaking" effect (explained in this paper:
                  https://arxiv.org/abs/1611.01236). Default is None.
        :param targeted: (optional) bool. Is the attack targeted or untargeted?
                  Untargeted, the default, will try to make the label incorrect.
                  Targeted will instead try to move in the direction of being more like y.
        :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
                  memory or for unit tests that intentionally pass strange input)
        :return: a tensor for the adversarial example
        """
        if norm not in [np.inf, 1, 2]:
            raise ValueError(
                "Norm order must be either np.inf, 1, or 2, got {} instead.".format(norm)
            )
        if eps < 0:
            raise ValueError(
                "eps must be greater than or equal to 0, got {} instead".format(eps)
            )
        if eps == 0:
            return x
        if clip_min is not None and clip_max is not None:
            if clip_min > clip_max:
                raise ValueError(
                    "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                        clip_min, clip_max
                    )
                )

        asserts = []

        # If a data range was specified, check that the input was in that range
        if clip_min is not None:
            assert_ge = torch.all(
                torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
            )
            asserts.append(assert_ge)

        if clip_max is not None:
            assert_le = torch.all(
                torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
            )
            asserts.append(assert_le)

        # x needs to be a leaf variable, of floating point type and have requires_grad being True for
        # its grad to be computed and stored properly in a backward call
        x = x.clone().detach().to(torch.float).requires_grad_(True)
        if y is None:
            # Using model predictions as ground truth to avoid label leaking
            _, y = torch.max(model_fn(x), 1)

        # Compute loss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(model_fn(x), y)
        # If attack is targeted, minimize loss of target label rather than maximize loss of correct label
        if targeted:
            loss = -loss

        # Define gradient of loss wrt input
        loss.backward()
        optimal_perturbation = optimize_linear(x.grad, eps, norm)

        # Add perturbation to original example to obtain adversarial example
        adv_x = x + optimal_perturbation

        # If clipping is needed, reset all values outside of [clip_min, clip_max]
        if (clip_min is not None) or (clip_max is not None):
            if clip_min is None or clip_max is None:
                raise ValueError(
                    "One of clip_min and clip_max is None but we don't currently support one-sided clipping"
                )
            adv_x = torch.clamp(adv_x, clip_min, clip_max)

        if sanity_checks:
            assert np.all(asserts)
        return adv_x

    def projected_gradient_descent(
            model_fn,
            x,
            eps,
            eps_iter,
            nb_iter,
            norm,
            clip_min=None,
            clip_max=None,
            y=None,
            targeted=False,
            rand_init=True,
            rand_minmax=None,
            sanity_checks=True,
    ):
        """
        This class implements either the Basic Iterative Method
        (Kurakin et al. 2016) when rand_init is set to False. or the
        Madry et al. (2017) method if rand_init is set to True.
        Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
        Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
        :param model_fn: a callable that takes an input tensor and returns the model logits.
        :param x: input tensor.
        :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
        :param eps_iter: step size for each attack iteration
        :param nb_iter: Number of attack iterations.
        :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
        :param clip_min: (optional) float. Minimum float value for adversarial example components.
        :param clip_max: (optional) float. Maximum float value for adversarial example components.
        :param y: (optional) Tensor with true labels. If targeted is true, then provide the
                  target label. Otherwise, only provide this parameter if you'd like to use true
                  labels when crafting adversarial samples. Otherwise, model predictions are used
                  as labels to avoid the "label leaking" effect (explained in this paper:
                  https://arxiv.org/abs/1611.01236). Default is None.
        :param targeted: (optional) bool. Is the attack targeted or untargeted?
                  Untargeted, the default, will try to make the label incorrect.
                  Targeted will instead try to move in the direction of being more like y.
        :param rand_init: (optional) bool. Whether to start the attack from a randomly perturbed x.
        :param rand_minmax: (optional) bool. Support of the continuous uniform distribution from
                  which the random perturbation on x was drawn. Effective only when rand_init is
                  True. Default equals to eps.
        :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
                  memory or for unit tests that intentionally pass strange input)
        :return: a tensor for the adversarial example
        """
        if norm == 1:
            raise NotImplementedError(
                "It's not clear that FGM is a good inner loop"
                " step for PGD when norm=1, because norm=1 FGM "
                " changes only one pixel at a time. We need "
                " to rigorously test a strong norm=1 PGD "
                "before enabling this feature."
            )
        if norm not in [np.inf, 2]:
            raise ValueError("Norm order must be either np.inf or 2.")
        if eps < 0:
            raise ValueError(
                "eps must be greater than or equal to 0, got {} instead".format(eps)
            )
        if eps == 0:
            return x
        if eps_iter < 0:
            raise ValueError(
                "eps_iter must be greater than or equal to 0, got {} instead".format(
                    eps_iter
                )
            )
        if eps_iter == 0:
            return x

        assert eps_iter <= eps, (eps_iter, eps)
        if clip_min is not None and clip_max is not None:
            if clip_min > clip_max:
                raise ValueError(
                    "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                        clip_min, clip_max
                    )
                )

        asserts = []

        # If a data range was specified, check that the input was in that range
        if clip_min is not None:
            assert_ge = torch.all(
                torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
            )
            asserts.append(assert_ge)

        if clip_max is not None:
            assert_le = torch.all(
                torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
            )
            asserts.append(assert_le)

        # Initialize loop variables
        if rand_init:
            if rand_minmax is None:
                rand_minmax = eps
            eta = torch.zeros_like(x).uniform_(-rand_minmax, rand_minmax)
        else:
            eta = torch.zeros_like(x)

        # Clip eta
        eta = clip_eta(eta, norm, eps)
        adv_x = x + eta
        if clip_min is not None or clip_max is not None:
            adv_x = torch.clamp(adv_x, clip_min, clip_max)

        if y is None:
            # Using model predictions as ground truth to avoid label leaking
            _, y = torch.max(model_fn(x), 1)

        i = 0
        while i < nb_iter:
            adv_x = fast_gradient_method(
                model_fn,
                adv_x,
                eps_iter,
                norm,
                clip_min=clip_min,
                clip_max=clip_max,
                y=y,
                targeted=targeted,
            )

            # Clipping perturbation eta to norm norm ball
            eta = adv_x - x
            eta = clip_eta(eta, norm, eps)
            adv_x = x + eta

            # Redo the clipping.
            # FGM already did it, but subtracting and re-adding eta can add some
            # small numerical error.
            if clip_min is not None or clip_max is not None:
                adv_x = torch.clamp(adv_x, clip_min, clip_max)
            i += 1

        asserts.append(eps_iter <= eps)
        if norm == np.inf and clip_min is not None:
            # TODO necessary to cast clip_min and clip_max to x.dtype?
            asserts.append(eps + clip_min <= clip_max)

        if sanity_checks:
            assert np.all(asserts)
        return adv_x
    # random_noise = torch.FloatTensor(*X_adv.shape).uniform_(-0.1, 0.1).to(device)
    # X_adv = Variable(X_adv.data + random_noise)
    with torch.enable_grad():
        x_pgd = projected_gradient_descent(model, x=X, eps=0.1, eps_iter=0.01, nb_iter=100, norm=np.inf)
        X_adv = Variable(x_pgd.data).to(device)
        # X_adv = Variable(X_adv.data + x_pgd.data)

    ################################################################################################
    ## end of attack method
    ################################################################################################

    return X_adv


# train function, you can use adversarial training
# optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0002)
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), 28 * 28)

        # use adverserial data to train the defense model
        # if torch.cuda.is_available():
        #     data, target = data.cuda(), target.cuda()
        adv_data = adv_attack(model, data, target, device=device)

        # clear gradients
        optimizer.zero_grad()

        # compute loss
        loss = F.nll_loss(model(adv_data), target)
        # loss = F.nll_loss(model(data), target)

        # get gradients and update
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

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # training
        train(args, model, device, train_loader, optimizer, epoch)

        # get trnloss and testloss
        trnloss, trnacc = eval_test(model, device, train_loader)
        advloss, advacc = eval_adv_test(model, device, train_loader)

        # print trnloss and testloss
        print('Epoch ' + str(epoch) + ': ' + str(int(time.time() - start_time)) + 's', end=', ')
        print('trn_loss: {:.4f}, trn_acc: {:.2f}%'.format(trnloss, 100. * trnacc), end=', ')
        print('adv_loss: {:.4f}, adv_acc: {:.2f}%'.format(advloss, 100. * advacc))

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

# Comment out the following command when you do not want to re-train the model
# In that case, it will load a pre-trained model you saved in train_model()
# model = train_model()

# Call adv_attack() method on a pre-trained model'
# the robustness of the model is evaluated against the infinite-norm distance measure
# important: MAKE SURE the infinite-norm distance (epsilon p) less than 0.11 !!!
# p_distance(model, train_loader, device)