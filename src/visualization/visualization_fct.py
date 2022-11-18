_MASK_COLORS_ ={
    0:'#141414', #black #background
    1:'#e31b1b', #red #front_door
    2:'#d46408', #orange #back_door
    3:'#f2cf07', #yellow #front_side
    4:'#1cd113', #green #back_side
    5:'#07d9af', #turquoise #front_bumper 
    6:'#0e1ded', #blue #bonnet #capot
    7:'#9816d9', #purple #back_bumper
    8:'#d60bae', #pink #trunk #coffre
}

_MASK_NAMES_ ={
    0:'background', #black #background
    1:'front_door', #red #front_door
    2:'back_door', #orange #back_door
    3:'front_side', #yellow #front_side
    4:'back_side', #green #back_side
    5:'front_bumper ', #turquoise #front_bumper 
    6:'bonnet', #blue #bonnet #capot
    7:'back_bumper', #purple #back_bumper
    8:'trunk', #pink #trunk #coffre
}

def get_mask_names():
    return _MASK_NAMES_

def get_mask_colors():
    return _MASK_COLORS_

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