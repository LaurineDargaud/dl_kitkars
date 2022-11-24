# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import hydra
from tqdm import tqdm
import numpy as np

from src.data.DeloitteDataset import split_dataset

from torchvision.transforms import ColorJitter

from torchvision import transforms

from src.visualization.visualization_fct import mask_to_rgb

import torch

import wandb

@click.command()
@hydra.main(version_base=None, config_path='../models/conf', config_name="config_unet")
def main(cfg, max_render=100):
    """ See some samples from data after data augmentation
    """
    log_wandb = cfg.log_wandb
    
    logger = logging.getLogger(__name__)
    logger.info(f'render data augmentation visualisation')
    
    # WANDB LOG
    if log_wandb:
        logger.info('setting wandb logging system')
        wandb.init(
            project="unet-visualisation", 
            entity="kitkars"
        )
    else:
        logger.info('NO WANDB LOG')
    
    # Define image transformations
    transformations_img = transforms.Compose([
        transforms.ToTensor(),
        #ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ColorJitter(brightness=0.1),
        transforms.Normalize(0.0, 1.0),
    ])
    
    # Define transformations to apply to both img and mask
    transformations_both = {
        'crop_resize': {
            'scale':(0.08, 1.0),
            'ratio':(0.75, 1.3333333333333333)
        }
    }
    
    # Load Datasets
    logger.info(f'loading datasets')
    train_dataset, valid_dataset, testing_dataset = split_dataset(
        cfg.data_paths.clean_data, 
        cfg.data_paths.test_set_filenames,
        transform_img=transformations_img,
        transform_both=transformations_both,
        data_real=cfg.data_augmentation.data_real,
        synthetic_data_ratio=cfg.data_augmentation.synthetic_data_ratio,
        train_valid_duplicate=cfg.data_augmentation.nb_train_valid_duplicate
    )
    
    all_datasets = {
        'train':train_dataset, 'valid': valid_dataset, 'test': testing_dataset
    }
    
    test_dataset = all_datasets['train']

    # wandb log
    if log_wandb:
        
        wandb.log({
            "transformations_img": str(transformations_img),
            "transformations_both": str(transformations_both)
        })
        
        logger.info(f'creating wandb table for visualization')
        
        # create a wandb.Table() with corresponding columns
        columns=["filename", "RGB augmented image", "real mask"]
        test_table = wandb.Table(columns=columns)
        
        random_idx = list(range(len(test_dataset)))
        np.random.permutation(random_idx)
        
        for i in tqdm(random_idx[:max_render]):            
            rgb_image, mask_img = test_dataset[i]
            
            #rgb_image = rgb_image.type(torch.int).cpu().detach().numpy()
            rgb_image = rgb_image.cpu().detach().numpy()
            rgb_image = np.transpose(rgb_image, (1, 2, 0))
            mask_img = mask_img.cpu().detach().numpy()[0]
            mask_img = mask_to_rgb(mask_img)
            
            filename = test_dataset.data_list[i].name
            test_table.add_data(
                filename, 
                wandb.Image(rgb_image), 
                wandb.Image(mask_img)
            )
        
        wandb.log({"test_table": test_table})
    
    logger.info('FINISHED visualisation')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
