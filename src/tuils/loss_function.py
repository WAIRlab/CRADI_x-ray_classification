import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.sum(dim=1).mean()

class BinaryEntropyLoss_weight(nn.Module):
    def __init__(self, weight=None, size_average=True, is_weight=True):
        super(BinaryEntropyLoss_weight, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.is_weight = is_weight
        self.class_num = np.array([[1375, 1285, 1701, 654, 2116, 6836, 7766, 5777, 6585, 5198, 9700, 18, 161, 37951, 3253, 12191, 694, 3099, 12787, 4558, 579, 13083, 1783, 484, 604]])
        self.class_num = np.power((1-self.class_num/74082), 2)
        # print(target.shape)

    def forward(self, input, target):

        self.weight = torch.cuda.FloatTensor(self.class_num.repeat(target.shape[0], axis=0))

        loss = F.binary_cross_entropy(input, target, self.weight, self.size_average)

        return loss

class FocalLossSigmoid(nn.Module):
    '''
    sigmoid version focal loss
    '''

    def __init__(self, alpha=0.25, gamma=2, size_average=False):
        super(FocalLossSigmoid, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = inputs
        alpha_mask = self.alpha * targets
        loss_pos = -1. * torch.pow(1 - P, self.gamma) * torch.log(P) * targets * alpha_mask
        loss_neg = -1. * torch.pow(1 - P, self.gamma) * torch.log(1 - P) * (1 - targets) * (1 - alpha_mask)
        batch_loss = loss_neg + loss_pos
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def f1_loss(logits, labels):
    __small_value=1e-6
    batch_size = logits.size()[0]
    p = logits
    l = labels
    num_pos = torch.sum(p, 1) + __small_value
    num_pos_hat = torch.sum(l, 1) + __small_value
    # print(num_pos)
    tp = torch.sum(l * p, 1)
    precise = tp / num_pos
    recall = tp / num_pos_hat
    # print(precise, recall)
    fs = precise * recall / (precise + recall + __small_value)
    loss = fs.sum() / batch_size
    return (1 - loss)

class F1_loss(nn.Module):

    def __init__(self):
        super(F1_loss, self).__init__()

    def forward(self, input, target):
        
        f1loss = f1_loss(input, target)
        return f1loss