from torchmetrics.functional import dice_score as pt_dice_score

def dice_score(preds, target):
    result = pt_dice_score(preds, target)
    result = result.cpu().detach().numpy()
    return result