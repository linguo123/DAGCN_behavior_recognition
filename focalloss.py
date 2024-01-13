# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1, alpha=0.5):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        # print(pred,true)
        loss = self.loss_fcn(pred, true)

        g_loss = torch.abs(loss)

        # print(loss.shape,true,'sssss')
        #print(loss.shape)
        p_t = torch.exp(-loss)

        loss *= (self.alpha * torch.sigmoid(1/(1.000001 - p_t) ))  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        # pred = pred.mean(1)
        # pred_prob = torch.sigmoid(pred)  # prob from logits
        # p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        # alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        # modulating_factor = (1.0 - p_t) ** self.gamma
        # loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss