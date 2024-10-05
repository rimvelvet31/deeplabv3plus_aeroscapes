import torch
import torch.nn.functional as F
import numpy as np
import time


class Metrics():
    def __init__(self, pred, gt, loss_fn, eps=1e-10, num_classes=2):
        # Use softmax to classify prediction
        self.pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
        self.gt = gt
        self.loss_fn = loss_fn
        self.eps = eps 
        self.num_classes = num_classes
        self.pred_ = pred
    
    # Flatten the input tensor to 1D
    def to_contiguous(self, inp):
        return inp.contiguous().view(-1)

    def loss(self): 
        return self.loss_fn(self.pred_, torch.argmax(self.gt, dim = 1))
        # return self.loss_fn(self.pred_, self.gt)

    def PixelAcc(self):
        with torch.no_grad():
            match = torch.eq(self.pred, self.gt).int()
        return float(match.sum()) / float(match.numel())

    def mIOU(self):
        with torch.no_grad():
            self.gt = torch.argmax(self.gt, dim=1)

            pred, gt = self.to_contiguous(self.pred), self.to_contiguous(self.gt)

            iou_per_class = []

            for c in range(self.num_classes):
                match_pred = pred == c
                match_gt = gt == c

                if match_gt.long().sum().item() == 0: 
                    iou_per_class.append(np.nan)   
                else:
                    intersect = torch.logical_and(match_pred, match_gt).sum().float().item()
                    union = torch.logical_or(match_pred, match_gt).sum().float().item()
                    iou = (intersect + self.eps) / (union + self.eps)
                    iou_per_class.append(iou)

            return np.nanmean(iou_per_class)
    

# Calculate training time
def timer(start_time = None): 
    return time.time() if start_time == None else time.time() - start_time