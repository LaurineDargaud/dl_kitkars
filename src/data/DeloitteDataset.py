import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from torchvision.transforms.functional import crop, resize, hflip, perspective
from torchvision.transforms import RandomResizedCrop, RandomPerspective, Normalize

class DeloitteDataset(Dataset):
    def __init__(self, data_list, transform_img=None, transform_mask=None, transform_both=None, feature_extractor=None, doNormalize=True):
        # set list of paths
        self.data_list = data_list
        # set rgb img transformation
        self.transform_img = transform_img
        # set both transformations
        self.transform_both = transform_both
        # set mask transformation
        if transform_mask == None:
            # default mask transformation
            self.transform_mask = ToTensor()
        else:
            self.transform_mask = transform_mask
        # set feature extractor
        self.feature_extractor = feature_extractor
        # set normalization bool
        self.doNormalize = doNormalize

    def __len__(self):
        return len(self.data_list)
    
    def apply_transformation(self, aKeyword, correspondingParams, aRGBImg, aMask):
        output_rgb_img = aRGBImg
        output_mask = aMask
        if aKeyword == 'crop_resize':
            # get parameters for both img and mask
            transformer = RandomResizedCrop((aRGBImg.size(-2),aRGBImg.size(-1)))
            parameters = transformer.get_params(aRGBImg, **correspondingParams)
            parameters = dict(zip(['top','left','height','width'], parameters))
            # apply crop
            output_rgb_img = crop(aRGBImg, **parameters)
            output_mask = crop(aMask, **parameters)
            # resize to original size
            output_rgb_img = resize(output_rgb_img, (aRGBImg.size(-2),aRGBImg.size(-1)))
            output_mask = resize(output_mask, (aRGBImg.size(-2),aRGBImg.size(-1)))
        
        elif aKeyword == 'random_hflip':
            # get probability
            p = correspondingParams['p']
            # get random number
            s = np.random.binomial(1, p)
            if s == 1:
                # do the h flip
                output_rgb_img = hflip(aRGBImg)
                output_mask = hflip(aMask)
        
        elif aKeyword == 'random_perspective':
            # get parameters for both img and mask
            transformer = RandomPerspective()
            startpoints , endpoints = transformer.get_params(aRGBImg.size(-1), aRGBImg.size(-2), **correspondingParams)
            # apply perspective
            output_rgb_img = perspective(aRGBImg, startpoints , endpoints )
            output_mask = perspective(aMask, startpoints, endpoints )
        
        return output_rgb_img, output_mask

    def get_rawImg(self, idx):
        aNumpyFilePath = self.data_list[idx]
        # load npy array
        numpy_array = np.load(aNumpyFilePath)
        rgb_img = (numpy_array[:3]*255)
        rgb_img = torch.Tensor(rgb_img).type(torch.uint8)
        return rgb_img

    def __getitem__(self, idx):
        aNumpyFilePath = self.data_list[idx]
        # load npy array
        numpy_array = np.load(aNumpyFilePath)
        # get RGB image
        # rgb_img = (np.transpose(numpy_array[:3], (1, 2, 0))*255).astype(float)
        rgb_img = (numpy_array[:3]*255)
        # get grayscale maske
        mask_img = numpy_array[3]
        
        # get distinct classes list in image from mask
        # distinct_classes_list = list(set(mask_img.flatten()))
        # get filename
        # filename = aNumpyFilePath.stem
        # to tensor
        rgb_img = torch.Tensor(rgb_img).type(torch.uint8)
        if self.transform_img != None:
            rgb_img = self.transform_img(rgb_img).type(torch.float)
        else:
            rgb_img = rgb_img.type(torch.float)
            
        mask_img = self.transform_mask(mask_img).type(torch.int)
            
        # apply transformations to both if provided
        if self.transform_both != None:
            for aTransformation, correspondingParams in self.transform_both.items():
                rgb_img, mask_img = self.apply_transformation(aTransformation, correspondingParams, rgb_img, mask_img)
        
        # transpose
        # rgb_img = rgb_img.view((rgb_img.size(2), rgb_img.size(0), rgb_img.size(1)))
        
        # normalization
        if self.doNormalize:
            rgb_img = Normalize((127.5,127.5,127.5), (127.5,127.5,127.5) ).forward(rgb_img)
        
        # apply feature extraction if exists
        if self.feature_extractor != None:
            # before torch.Size([3, 256, 256]) torch.Size([1, 256, 256])
            features = self.feature_extractor(rgb_img, mask_img[0])
            rgb_features = torch.FloatTensor(features['pixel_values'][0])
            mask_features = torch.IntTensor(features['labels'][0])
            mask_features = mask_features.view((1,mask_features.size(0), mask_features.size(1)))
            # after torch.Size([3, 512, 512]) torch.Size([1, 512, 512])
            return rgb_features, mask_img
        else:
            return rgb_img, mask_img

from pathlib import Path
import glob2
import numpy as np
import random

 # clear dataset from the fake images
def processing(path_list, seed_random, ratio=0.1, duplicate=2):
    np.random.seed(seed_random)
    real_data = []
    synt_data = []
    for aPath in path_list:
        if not ("DOOR" in aPath.stem or "OPEL" in aPath.stem):
            real_data.append(aPath)
        else:
            synt_data.append(aPath)
        stop = int(len(synt_data)*ratio)
        random.shuffle(synt_data)
        synt_slice = synt_data[slice(stop)]
        together = (real_data + synt_slice)*duplicate
    return together

def split_dataset(
    aPath, aTestTXTFilenamesPath, 
    train_ratio=0.85, valid_ratio=0.15, seed_random=42, 
    transform_img=None, transform_mask=None, transform_both=None, test_only_transform=None, 
    data_real=False, 
    feature_extractor=None,
    synthetic_data_ratio=0.1, train_valid_duplicate=2):
    """_summary_

    :param aPath: path to folder that contains all npy data files
    :type aPath: str
    :param aTestTXTFilenamesPath: path to txt file with test set filenames (in references/test_set_ids.txt)
    :type aTestTXTFilenamesPath: str
    :param train_ratio: percentage of training data, defaults to 0.85
    :type train_ratio: float, optional
    :param valid_ratio: percentage of validation data, defaults to 0.15
    :type valid_ratio: float, optional
    :param seed_random: random seed to use, defaults to 42
    :type seed_random: int, optional
    :return: train_dataset, valid_dataset, test_dataset
    :rtype: 3 DeloitteDataset objects
    """
    # set random np seed
    np.random.seed(seed_random)
    
    #building training, validation and test sets
    assert train_ratio+valid_ratio == 1.0
    
    # get test filenames list
    test_ids = np.loadtxt(aTestTXTFilenamesPath, dtype=str)
    test_filenames = np.array([f.split('.')[0] for f in test_ids])
    
    # get all data
    all_paths = [ Path(p).absolute() for p in glob2.glob(aPath + '/*') ]
    
    # get filepaths lists
    train_valid_data_list, test_data_list = [], []
    for aPath in all_paths:
        # get filename
        filename = aPath.stem
        if filename in test_filenames:
            test_data_list.append(aPath)
        else:
            train_valid_data_list.append(aPath)
    
    # get test dataset
    test_dataset = DeloitteDataset(test_data_list, transform_img=test_only_transform, feature_extractor=feature_extractor)

    # get train and valid datasets
    if data_real:
        train_valid_data_list = np.array(
            processing(train_valid_data_list,  seed_random=seed_random, ratio=synthetic_data_ratio, duplicate=train_valid_duplicate)
        )
    else:
        train_valid_data_list = np.array(train_valid_data_list)
    permutation = np.random.permutation(len(train_valid_data_list))
    
    train_indices = permutation[:int(train_ratio*len(train_valid_data_list))]
    valid_indices = permutation[int(train_ratio*len(train_valid_data_list)):]
    train_dataset = DeloitteDataset(list(train_valid_data_list[train_indices]), transform_img=transform_img, transform_mask=transform_mask, transform_both=transform_both, feature_extractor=feature_extractor)
    valid_dataset = DeloitteDataset(list(train_valid_data_list[valid_indices]), transform_img=transform_img, transform_mask=transform_mask, transform_both=transform_both, feature_extractor=feature_extractor)
        
    return train_dataset, valid_dataset, test_dataset
