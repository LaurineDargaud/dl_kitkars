# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import hydra
from tqdm import tqdm
import numpy as np

from src.data.DeloitteDataset import split_dataset

from src.visualization.visualization_fct import get_mask_names

from autoencoder import AutoEncoder

from torchvision import transforms

from src.models.performance_metrics import dice_score

from src.visualization.visualization_fct import mask_to_rgb

from src.models.finetune_autoencoder import resize_logits

import torch
from torch.utils.data import DataLoader
from torch import nn

import wandb

@click.command()
@hydra.main(version_base=None, config_path='conf', config_name="config_autoencoder")
def main(cfg):
    """ Predict test set with finetuned Autoencoder model
    """
    cuda, name, log_wandb, dataset_to_predict = cfg.cuda, cfg.name_best, cfg.log_wandb, cfg.predict_dataset
    
    logger = logging.getLogger(__name__)
    logger.info(f'predict {dataset_to_predict} set with finetuned Autoencoder model')

    
    # WANDB LOG
    if log_wandb:
        logger.info('setting wandb logging system')
        wandb.init(
            project="autoencoder-predictions", 
            entity="kitkars", 
            name=f'{name} ({dataset_to_predict})',
            config={
                "pt_name":f'autoencoder_finetuned_{name}.pt',
                "batch_size": cfg.hyperparameters.batch_size
            }
        )
    else:
        logger.info('NO WANDB LOG')
    
    # Set torch device
    device = torch.device(f'cuda:{cuda}')
    
    # Define image transformations
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
    ])
    
    # Load Datasets
    
    logger.info(f'loading {dataset_to_predict} set')
    train_dataset, valid_dataset, testing_dataset = split_dataset(
        cfg.data_paths.clean_data, 
        cfg.data_paths.test_set_filenames,
        transform=transformations
    )
    
    all_datasets = {
        'train':train_dataset, 'valid': valid_dataset, 'test': testing_dataset
    }
    
    test_dataset = all_datasets[dataset_to_predict]
    
    # Load real img dataset - without extractor, for visualisation
    train_raw_dataset, valid_raw_dataset, testing_raw_dataset = split_dataset(
        cfg.data_paths.clean_data, 
        cfg.data_paths.test_set_filenames,
        transform=transformations
    )
    
    all_raw_datasets = {
        'train':train_raw_dataset, 'valid': valid_raw_dataset, 'test': testing_raw_dataset
    }
    
    test_raw_dataset = all_raw_datasets[dataset_to_predict]
    
    batch_size=cfg.hyperparameters.batch_size
    
    # Get dataloaders
    logger.info(f'creating {dataset_to_predict} dataloader')
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        num_workers=cfg.hyperparameters.num_workers,
        shuffle=False,
        drop_last=False
    )
    
    # Load model
    logger.info('load AutoEncoder finetuned model')
    model = AutoEncoder(
            n_channels = cfg.autoencoder_parameters.nb_input_channels,
            n_classes = cfg.autoencoder_parameters.nb_output_channels
    )
    model.load_state_dict(torch.load(cfg.model_paths.models+f'Autoencoder_{name}.pt'))
    model = model.to(device)
    
    # Set loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Make predictions    
    test_dice_scores_batches = []
    test_loss = 0
    all_predictions = []
    all_dice_scores = []
    
    logger.info(f'running {dataset_to_predict} predictions')
    
    with torch.no_grad():
        model.eval()
        for rgb_img, mask_img in tqdm(test_loader):
            rgb_img, mask_img = rgb_img.to(device), mask_img.to(device)
            output = model(rgb_img)['x_hat']
            
            loss = loss_fn(
                output.flatten(start_dim=2, end_dim=len(output.size())-1), 
                mask_img.flatten(start_dim=1, end_dim=len(mask_img.size())-1).type(torch.long)
            )
            test_loss += loss  

            predictions = output.flatten(start_dim=2, end_dim=len(output.size())-1)

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
                    float(dice_score(
                        output[i].flatten(start_dim=1, end_dim=len(output.size())-2).unsqueeze(0),
                        mask_img[i].flatten(start_dim=0, end_dim=len(mask_img.size())-2).unsqueeze(0)
                    ))
                )
    
    # Get performance metrics
    test_dice_score = np.sum(test_dice_scores_batches) / len(test_dataset)
    print(f"Test DICE score: {test_dice_score}")
    print(f"[Check] Avg Test DICE score: {np.mean(all_dice_scores)}")
    
    # wandb log
    if log_wandb:
        
        wandb.log({
            "test_dice_score": test_dice_score,
            "test_loss": test_loss.cpu().detach().numpy() / len(test_dataset)
        })
        
        logger.info(f'creating wandb table for predictions visualization')
        
        # create a wandb.Table() with corresponding columns
        columns=["id", "filename", "RGB image", "features", "real mask", "prediction", "DICE score float", "DICE score"]
        test_table = wandb.Table(columns=columns)
        
        for i in tqdm(range((len(test_raw_dataset)))):            
            rgb_image, mask_img = test_raw_dataset[i]
            rgb_features, _ = test_dataset[i]
            
            rgb_image = rgb_image.type(torch.int).cpu().detach().numpy()
            rgb_image = np.transpose(rgb_image, (1, 2, 0))
            mask_img = mask_img.cpu().detach().numpy()[0]
            mask_img = mask_to_rgb(mask_img)
            rgb_features = rgb_features.type(torch.int).cpu().detach().numpy()
            rgb_features = np.transpose(rgb_features, (1, 2, 0))
            
            logit_prediction = all_predictions[i]
            predicted_mask_img = mask_to_rgb(np.argmax(logit_prediction, axis=0))
            
            filename = test_dataset.data_list[i].name
            test_table.add_data(
                i, 
                filename, 
                wandb.Image(rgb_image), 
                wandb.Image(rgb_features), 
                wandb.Image(mask_img), 
                wandb.Image(predicted_mask_img),
                all_dice_scores[i],
                round(all_dice_scores[i]*100,3)
            )
        
        wandb.log({"test_table": test_table})
    
    logger.info('FINISHED predictions')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
