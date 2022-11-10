# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import hydra
from tqdm import tqdm
import numpy as np

from src.data.DeloitteDataset import split_dataset

from src.models.unet import UNet

from src.models.performance_metrics import dice_score

import torch
from torch.utils.data import DataLoader
from torch import nn

import wandb

@click.command()
@hydra.main(version_base=None, config_path='conf', config_name="config_unet")
def main(cfg):
    """ Predict test set with finetuned U-Net model
    """
    logger = logging.getLogger(__name__)
    logger.info('predict test set with finetuned U-Net model')
    
    cuda, name, log_wandb = cfg.cuda, cfg.name_best, cfg.log_wandb
    
    # WANDB LOG
    if log_wandb:
        logger.info('setting wandb logging system')
        wandb.init(
            project="unet-predictions", 
            entity="kitkars", 
            name=name,
            config={
                "pt_name":f'unet_finetuned_{name}.pt',
                "batch_size": cfg.hyperparameters.batch_size
            }
        )
    else:
        logger.info('NO WANDB LOG')
    
    # Set torch device
    device = torch.device(f'cuda:{cuda}')
    
    # Load Datasets
    logger.info('loading test set')
    _, _, test_dataset = split_dataset(
        cfg.data_paths.clean_data, 
        cfg.data_paths.test_set_filenames
    )
    
    batch_size=cfg.hyperparameters.batch_size
    
    # Get dataloaders
    logger.info('creating test dataloader')
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        num_workers=cfg.hyperparameters.num_workers,
        shuffle=False,
        drop_last=False
    )
    
    # Load model
    logger.info('load U-net pretrained model')
    model = UNet(n_channels=3, n_classes=cfg.unet_parameters.nb_output_channels)
    model.load_state_dict(torch.load(cfg.model_paths.models+f'unet_finetuned_{name}.pt'))
    model = model.to(device)
    
    # Set loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Make predictions    
    test_dice_scores_batches = []
    test_loss = 0
    all_predictions = []
    all_dice_scores = []
    
    with torch.no_grad():
        model.eval()
        for rgb_img, mask_img in tqdm(test_loader):
            rgb_img, mask_img = rgb_img.to(device), mask_img.to(device)
            output = model(rgb_img)
            loss = loss_fn(
                output.flatten(start_dim=2, end_dim=len(output.size())-1), 
                mask_img.flatten(start_dim=1, end_dim=len(mask_img.size())-1).type(torch.long)
            )
            test_loss += loss  

            predictions = output.flatten(start_dim=2, end_dim=len(output.size())-1).softmax(1)

            # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
            test_dice_scores_batches.append(
                dice_score(
                    predictions, 
                    mask_img.flatten(start_dim=1, end_dim=len(mask_img.size())-1)
                ) * len(rgb_img)
            )
            
            # Save output mask
            for i in range(len(output)):
                all_predictions.append(output[i].cpu().detach().numpy())
                all_dice_scores.append(
                    dice_score(
                        output[i].flatten(start_dim=2, end_dim=len(output[i].size())-1).softmax(1),
                        mask_img[i].flatten(start_dim=1, end_dim=len(mask_img[i].size())-1)
                    )
                )
    
    # Get performance metrics
    test_dice_score = np.sum(test_dice_scores_batches) / len(test_dataset)
    print(f"Test DICE score: {test_dice_score}")
    
    # wandb log
    if log_wandb:
        wandb.log({
            "test_dice_score": test_dice_score,
            "test_loss": test_loss.cpu().detach().numpy() / len(test_dataset)
        })
        
        # create a wandb.Table() with corresponding columns
        columns=["id", "filename", "RGB image", "real mask", "prediction", "DICE score"]
        test_table = wandb.Table(columns=columns)
        
        for i in range (len(test_dataset)):
            rgb_image, mask_img = test_dataset[i]
            logit_prediction = all_predictions[i]
            predicted_mask_img = np.argmax(logit_prediction, axis=0)
            filename = test_dataset.data_list[i]
            test_table.add_data(
                i, 
                filename, 
                wandb.Image(rgb_image), 
                wandb.Image(mask_img), 
                wandb.Image(predicted_mask_img),
                all_dice_scores[i]
            )
        
        wandb.log({"test_table": test_table})

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
