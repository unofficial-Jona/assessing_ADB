import torch
import torch.nn.functional as F
from torch import nn

from pdb import set_trace


class SetCriterion(nn.Module):
    def __init__(self, reduction='mean', weights=None):
        super().__init__()
        self.criterion_CE = nn.CrossEntropyLoss(reduction=reduction)
        self.criterion_MSE = nn.MSELoss(reduction=reduction)
        self.criterion_KL = nn.KLDivLoss(reduction='batchmean')
        self.criterion_Multilabel = nn.BCEWithLogitsLoss(pos_weight=weights)

    def forward(self, outputs, targets, type):
        if type == 'CE':
            loss = self.criterion_CE(outputs, targets)
        elif type == 'MSE':
            outputs = F.softmax(outputs, dim=1)
            targets = F.softmax(targets, dim=1)
            loss = self.criterion_MSE(outputs, targets)
        elif type == 'KL':
            outputs = F.softmax(outputs, dim=1)
            targets = F.softmax(targets, dim=1)
            loss1 = self.criterion_KL(outputs, targets)
            loss2 = self.criterion_KL(targets, outputs)
            loss = loss1 + loss2
        ### My modifications ###
        elif type == 'KL_new':
            outputs = torch.sigmoid(outputs)
            targets = torch.sigmoid(targets)
            loss1 = self.criterion_KL(torch.log(outputs), targets)
            loss2 = self.criterion_KL(torch.log(targets), outputs)
            loss = loss1 + loss2
        elif type == 'ML':
            loss = self.criterion_Multilabel(outputs, targets)
        return loss


if __name__ == '__main__':
    x_input = torch.randn(128, 5, 1)
    y_input = torch.randn(128, 5, 1)
    x = SetCriterion()

    y = x(x_input, y_input, 'KL_new')
    print(y)
    
    y = x(x_input, y_input, 'KL')
    print(y)
    
    y = x(x_input, y_input,  'ML')
    print(y)
