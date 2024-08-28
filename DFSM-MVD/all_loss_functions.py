from torch.autograd import Function
import torch
import  torch.nn as nn
import torch.nn.functional as F

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

class CodeTGContrastiveLoss(nn.Module):
    def __init__(self,projection_dim=100):
        super().__init__()
        self.temperature=0.05

    def forward(self,CTembeddings,CGenbeddings):

        # Calculating the Loss
        
        logits = (CTembeddings @ CGenbeddings.T) / self.temperature
        codegraphs_similarity = CGenbeddings @ CGenbeddings.T
        codetexts_similarity = CTembeddings @ CTembeddings.T

        targets = F.softmax(
            (codegraphs_similarity + codetexts_similarity) / 2 * self.temperature, dim=-1
        )
        codetexts_loss = cross_entropy(logits, targets, reduction='none')
        codegraphs_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (codegraphs_loss + codetexts_loss) / 2.0  # shape: (batch_size)
        return loss.mean()

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


class SIMSE(nn.Module):

    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1 - x2, 2)
        summed = torch.sum(power)
        sqrt = summed ** (0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=torch.tensor([1.0, 1.2])):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input = input.to(device)
        target = target.to(device)

        logpt = nn.functional.log_softmax(input, dim=1)
        pt = torch.exp(logpt)

        alpha = self.alpha[target].unsqueeze(dim=1)
        alpha = alpha.to(device)

        logpt = alpha * (1 - pt) ** self.gamma * logpt

        loss = nn.functional.nll_loss(logpt, target, reduction='sum')#sum

        return loss