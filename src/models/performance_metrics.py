# from torchmetrics.functional import dice_score as pt_dice_score
from torchmetrics.functional import dice as pt_dice_score

def dice_score(preds, target, average_method = 'macro'):
    result = pt_dice_score(preds, target, average=average_method, num_classes=preds.size()[1])
    # result = pt_dice_score(preds, target, average=average_method, num_classes=preds.size()[1], zero_division=1)
    #result = pt_dice_score(preds, target, average=average_method, num_classes=preds.size()[1], zero_division=1, ignore_index=0)
    result = result.cpu().detach().numpy()
    return result

def dice_score_2Dmasks(preds, target):
    #result = pt_dice_score(preds, target)
    result = pt_dice_score(preds, target, ignore_index=0) # we consider background as non-class
    result = result.cpu().detach().numpy()
    return result

def dice_score_class(preds, target, average_method = 'none'):
    x = dice_score(preds = preds, target = target, average_method = average_method)
    return x