# from torchmetrics.functional import dice_score as pt_dice_score
from torchmetrics.functional import dice as pt_dice_score

def dice_score(preds, target, average_method = 'macro'):
    result = pt_dice_score(preds, target, average=average_method, num_classes=preds.size()[1])
    result = result.cpu().detach().numpy()
    return result