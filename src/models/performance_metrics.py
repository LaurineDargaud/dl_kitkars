# from torchmetrics.functional import dice_score as pt_dice_score
from torchmetrics.functional import dice as pt_dice_score

def dice_score(preds, target, average_method = 'macro'):
    result = pt_dice_score(preds, target, average=average_method, num_classes=preds.size()[1])
    result = result.cpu().detach().numpy()
    return result


def dice_score_class(preds, target, average_method = 'none', mdmc_average= 'samplewise'):
    x = dice_score(preds = preds, target = target, average_method = average_method, mdmc_average= mdmc_average)
    return x