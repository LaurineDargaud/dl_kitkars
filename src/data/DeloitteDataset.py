import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class DeloitteDataset(Dataset):
    def __init__(self, data_list, transform=ToTensor(), transform_mask=ToTensor()):
        self.data_list = data_list
        self.transform = transform
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        aNumpyFilePath = self.data_list[idx]
        # load npy array
        numpy_array = np.load(aNumpyFilePath)
        # get RGB image
        rgb_img = (np.transpose(numpy_array[:3], (1, 2, 0))*255).astype(float)
        # get grayscale maske
        mask_img = numpy_array[3]
        # get distinct classes list in image from mask
        # distinct_classes_list = list(set(mask_img.flatten()))
        # get filename
        # filename = aNumpyFilePath.stem
        # to tensor
        rgb_img = self.transform(rgb_img).type(torch.float)
        mask_img = self.transform_mask(mask_img).type(torch.int)
        # return {'image':rgb_img, 'mask':mask_img, 'distinct_classes':distinct_classes_list, 'filename':filename, 'path':aNumpyFilePath}
        return rgb_img, mask_img

from pathlib import Path
import glob2
import numpy as np

def split_dataset(aPath, aTestTXTFilenamesPath, train_ratio=0.85, valid_ratio=0.15, seed_random=42, transform=None, data_real=False):
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
    if transform == None:
        test_dataset = DeloitteDataset(test_data_list)
    else:
        test_dataset = DeloitteDataset(test_data_list, transform)
    
    # clear dataset from the fake images
    def processing(path_list):
        real_data = []
        for aPath in path_list:
            if not ("DOOR" in aPath.stem or "OPEL" in aPath.stem):
                real_data.append(aPath)
        return real_data

    # get train and valid datasets
    if data_real:
        train_valid_data_list = np.array(processing(train_valid_data_list))
    else:
        train_valid_data_list = np.array(train_valid_data_list)
    permutation = np.random.permutation(len(train_valid_data_list))
    train_indices = permutation[:int(train_ratio*len(train_valid_data_list))]
    valid_indices = permutation[int(train_ratio*len(train_valid_data_list)):]
    
    if transform == None:
        train_dataset = DeloitteDataset(list(train_valid_data_list[train_indices]))
        valid_dataset = DeloitteDataset(list(train_valid_data_list[valid_indices]))
    else:
        train_dataset = DeloitteDataset(list(train_valid_data_list[train_indices]), transform)
        valid_dataset = DeloitteDataset(list(train_valid_data_list[valid_indices]), transform)

    
    return train_dataset, valid_dataset, test_dataset
