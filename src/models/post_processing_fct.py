import cv2
import numpy as np

__MORPH_OPERATIONS__ = {
    0: [], # do nothing on background,
    1: [(cv2.MORPH_OPEN, cv2.MORPH_ELLIPSE, 3)],
    2: [(cv2.MORPH_OPEN, cv2.MORPH_RECT, 3)],
    3: [(cv2.MORPH_OPEN, cv2.MORPH_ELLIPSE, 3),(cv2.MORPH_CLOSE, cv2.MORPH_ELLIPSE, 3)],
    4: [(cv2.MORPH_OPEN, cv2.MORPH_CROSS, 3),(cv2.MORPH_CLOSE, cv2.MORPH_CROSS, 3)],
    5: [(cv2.MORPH_OPEN, cv2.MORPH_ELLIPSE, 3)],
    6: [(cv2.MORPH_OPEN, cv2.MORPH_ELLIPSE, 3)],
    7: [(cv2.MORPH_OPEN, cv2.MORPH_ELLIPSE, 3),(cv2.MORPH_CLOSE, cv2.MORPH_ELLIPSE, 3)],
    8: [(cv2.MORPH_OPEN, cv2.MORPH_ELLIPSE, 3)]
}

def apply_morphology(aInitMask, aListOfOps):
    # ex: aListOfMorphOps = [ (cv2.MORPH_OPEN, cv2.MORPH_ELLIPSE, 3) ]
    processed_mask = aInitMask.astype(float)
    for morphType, kernelType, kernelSize in aListOfOps:
        kernel = cv2.getStructuringElement(kernelType,(kernelSize,kernelSize))
        processed_mask = cv2.morphologyEx(processed_mask, morphType, kernel)
    return processed_mask


def post_processing(aMask, dice_confidence_dict):
    predicted_mask = []
    for aClassIndex, aListOfOperations in __MORPH_OPERATIONS__.items():
        # get binary mask for given class
        predicted_binary_mask = aMask == aClassIndex
        # apply morphological operations
        processed_binary_mask = apply_morphology(predicted_binary_mask, aListOfOperations)
        # append list with processed binary mask
        predicted_mask.append(processed_binary_mask.astype(int)*(dice_confidence_dict[aClassIndex]))
    predicted_mask = np.array(predicted_mask)
    # argmax to merge all masks
    predicted_mask = np.argmax(predicted_mask, axis=0)
    return predicted_mask