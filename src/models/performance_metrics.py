# from torchmetrics.functional import dice_score as pt_dice_score
from torchmetrics.functional import dice as pt_dice_score

def dice_score(preds, target, average_method = 'macro', mdmc_average='samplewise'):
    result = pt_dice_score(preds, target, average=average_method, num_classes=preds.size()[1], mdmc_average=mdmc_average)
    result = result.cpu().detach().numpy()
    return result


def dice_score_class(preds, target, average_method = 'none', mdmc_average= 'samplewise'):
    x = dice_score(preds, target, average_method = average_method, mdmc_average= mdmc_average)
    #import pdb; pdb.set_trace()
    return x