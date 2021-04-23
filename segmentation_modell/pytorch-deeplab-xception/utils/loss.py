import torch
import torch.nn as nn
import torch.nn.functional as F
class ACLoss(nn.Module):
    def __init__(self):
        super(ACLoss,self).__init__()
    def forward(self,logit,target,weight=10):
        '''
        y_true, y_pred: tensor of shape (B, C, H, W), where y_true[:,:,region_in_contour] == 1, y_true[:,:,region_out_contour] == 0.
        weight: scalar, length term weight.
        '''
        N,C,H,W = logit.shape
        target = target.unsqueeze(1)
        onehot = torch.cuda.FloatTensor(N,C,H,W).zero_()
        y_true = onehot.scatter_(1,target,1)
        y_pred = logit
        # length term
        delta_r = y_pred[:,3,1:,:] - y_pred[:,3,:-1,:] # horizontal gradient (B, C, H-1, W) 
        delta_c = y_pred[:,3,:,1:] - y_pred[:,3,:,:-1] # vertical gradient   (B, C, H,   W-1)
        
        delta_r    = delta_r[:,1:,:-2]**2  # (B, C, H-2, W-2)
        delta_c    = delta_c[:,:-2,1:]**2  # (B, C, H-2, W-2)
        delta_pred = torch.abs(delta_r + delta_c) 

        epsilon = 1e-8 # where is a parameter to avoid square root is zero in practice.
        length = torch.mean(torch.sqrt(delta_pred + epsilon)) # eq.(11) in the paper, mean is used instead of sum.
        #print('length',length)
        # region term
        c_in  = torch.ones_like(y_pred)
        c_out = torch.zeros_like(y_pred)

        region_in  = torch.mean( y_pred[:,3,:,:]     * (y_true[:,3,:,:] - c_in[:,3,:,:] )**2 ) # equ.(12) in the paper, mean is used instead of sum.
        region_out = torch.mean( (1-y_pred[:,3,:,:]) * (y_true[:,3,:,:] - c_out[:,3,:,:])**2 ) 
        region = region_in + region_out
        
        loss =  weight*length + region
        return loss
class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'segshape':
            return self.SegShapeLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss
    
    def SegShapeLoss(self, logit, target):
        n, c, h, w = logit.size()
        
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        criterion1 = ACLoss()
        if self.cuda:
            criterion = criterion.cuda()
            criterion1 = criterion1.cuda()
        segloss = criterion(logit, target.long())
        if self.batch_average:
            segloss /= n
        logit = F.softmax(logit,1)
        shapeloss = criterion1(logit,target.long())
        a = 0.4
        loss = (1-a)*segloss+a*shapeloss
        return loss

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




