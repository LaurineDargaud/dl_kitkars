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

def get_morph_operations():
    return __MORPH_OPERATIONS__

def combine_processed_binary_masks(aMultiBinaryMask, confidence_per_class=[0, 1, 6, 3, 5, 2, 7, 4, 8]):
    recombined_img = np.zeros(aMultiBinaryMask.shape[1:], dtype=int)
    confidence_per_class = np.array(confidence_per_class)

    for i in range (recombined_img.shape[0]):
        for j in range (recombined_img.shape[1]):
            all_pixel_classes = aMultiBinaryMask[:,i,j]
            if np.sum(all_pixel_classes) == 0:
                pixel_value = 0
            elif np.sum(all_pixel_classes) == 1:
                pixel_value = np.argwhere(all_pixel_classes==1)[0][0]
            else:
                potential_classes = np.argwhere(all_pixel_classes==1).flatten()
                best_class = sorted(potential_classes, key=lambda x:np.argwhere(confidence_per_class==x)[0][0])[0]
                pixel_value = best_class
            recombined_img[i,j]=pixel_value
    
    return recombined_img

def apply_morphology(aInitMask, aListOfOps):
    # ex: aListOfMorphOps = [ (cv2.MORPH_OPEN, cv2.MORPH_ELLIPSE, 3) ]
    processed_mask = aInitMask.astype(float)
    for morphType, kernelType, kernelSize in aListOfOps:
        kernel = cv2.getStructuringElement(kernelType,(kernelSize,kernelSize))
        processed_mask = cv2.morphologyEx(processed_mask, morphType, kernel)
    return processed_mask


def post_processing(aMask):
    predicted_mask = []
    for aClassIndex, aListOfOperations in __MORPH_OPERATIONS__.items():
        # get binary mask for given class
        predicted_binary_mask = aMask == aClassIndex
        # apply morphological operations
        processed_binary_mask = apply_morphology(predicted_binary_mask, aListOfOperations)
        # append list with processed binary mask
        predicted_mask.append(processed_binary_mask.astype(int))
    predicted_mask = np.array(predicted_mask)
    return combine_processed_binary_masks(predicted_mask)