_MASK_COLORS_ ={
    0:'#141414', #black
    1:'#e31b1b', #red
    2:'#d46408', #orange
    3:'#f2cf07', #yellow
    4:'#1cd113', #green
    5:'#07d9af', #turquoise
    6:'#0e1ded', #blue
    7:'#9816d9', #purple
    8:'#d60bae', #pink
}

import numpy as np

def hex_to_rgb(h):
    """ Hex format color to RGB list. Ex: '#e31b1b' -> np.array([227, 27, 27]) """
    if '#' in h:
        h = h[1:]
    return np.array(list(int(h[i:i+2], 16) for i in (0, 2, 4)))

def mask_to_rgb(aMask):
    aMask = aMask.astype(int)
    mask_rgb = np.zeros((len(aMask), aMask.shape[1], 3))
    for i in range(len(aMask)):
        for j in range (aMask.shape[1]):
            mask_rgb[i,j] = hex_to_rgb(_MASK_COLORS_[aMask[i,j]])
    return mask_rgb.astype(int)