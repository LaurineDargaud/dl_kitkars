# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.data.DeloitteDataset import split_dataset
from torch.utils.data import DataLoader

import hydra

@click.command()
@hydra.main(config_name="config_unet.yaml")
def main(cfg):
    """ Fine tuning our U-Net pretrained model - baseline
    """
    logger = logging.getLogger(__name__)
    logger.info('finetune UNet pretrained model')
    
    # Load Datasets
    train_dataset, valid_dataset, test_dataset = split_dataset(cfg.data_paths.clean_data)
    
    # Get dataloaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=False)
    
    
    
    
    
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
