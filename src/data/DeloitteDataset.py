from dataclasses import dataclass

@dataclass
class Car:
    distinct_classes: list
    mask: list
    rgb_image: list
    filename: str
    npy_file_path: str
    is_in_test_set: bool
    
from torch.utils.data import Dataset

class DeloitteDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        aCar = self.data_list[idx]
        return (aCar.rgb_image, aCar.mask)

from pathlib import Path
import glob2
import numpy as np

def get_all_data(aPath, test_set_list):
    """Get all files as Car from given folder

    :param aPath: folder path (here: data/raw/carset_data/clean_data)
    :type aPath: str
    :param test_set_list: list of test set's filenames (provided by Deloitte)
    :type test_set_list: list
    :return: data in given folder as Car instances
    :rtype: list of Car
    """
    all_paths = [ Path(p).absolute() for p in glob2.glob(aPath + '/*') ]
    all_data = []
    
    for aNumpyFilePath in all_paths:
        # load npy array
        numpy_array = np.load(aNumpyFilePath)
        # get RGB image
        rgb_img = (np.transpose(numpy_array[:3], (1, 2, 0))*255).astype(int)
        # get grayscale maske
        mask_img = numpy_array[3]
        # get distinct classes list in image from mask
        distinct_classes_list = list(set(mask_img.flatten()))
        # get filename
        filename = aNumpyFilePath.stem
        # is it in test set?
        is_in_test = filename in test_set_list
        # add object
        all_data.append(
            Car(
                distinct_classes=distinct_classes_list,
                mask=mask_img,
                rgb_image=rgb_img,
                filename=filename,
                npy_file_path=aNumpyFilePath,
                is_in_test_set=is_in_test
            )
        )
        
    return all_data

def split_dataset(aPath, aTestTXTFilenamesPath, train_ratio=0.85, valid_ratio=0.15, seed_random=42):
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
    test_ids = np.loadtxt(aTestTXTFilenamesPath, delimiter='\n', dtype=str)
    test_filenames = np.array([f.split('.')[0] for f in test_ids])
    
    # get all data
    all_data = get_all_data(aPath, test_filenames)
    
    # get test dataset
    test_dataset = DeloitteDataset([c for c in all_data if c.is_in_test_set])
    
    # get train and valid datasets
    train_valid_data_list = np.array([c for c in all_data if not c.is_in_test_set])
    permutation = np.random.permutation(len(train_valid_data_list))
    train_indices = permutation[:train_ratio*len(train_valid_data_list)]
    valid_indices = permutation[train_ratio*len(train_valid_data_list):]
    
    train_dataset = DeloitteDataset(list(train_valid_data_list[train_indices]))
    valid_dataset = DeloitteDataset(list(train_valid_data_list[valid_indices]))
    
    return train_dataset, valid_dataset, test_dataset