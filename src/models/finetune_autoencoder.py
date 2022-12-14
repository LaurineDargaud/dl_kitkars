# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from torchvision.transforms import ColorJitter, Grayscale

import hydra
from tqdm import tqdm
import numpy as np

from src.data.DeloitteDataset import split_dataset

from autoencoder import AutoEncoder

from src.models.performance_metrics import dice_score

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torch import nn
from torchvision import transforms
import torch.optim as optim

import wandb

@click.command()
@hydra.main(version_base=None, config_path='conf', config_name="config_autoencoder2")
def main(cfg):
    """ Training our AutoEncoder model - baseline
    """
    logger = logging.getLogger(__name__)
    logger.info('finetune AutoEncoder pretrained model')
    
    cuda, name, log_wandb = cfg.cuda, cfg.name, cfg.log_wandb
    
     # Define image transformations
    transformations_img = transforms.Compose(
        [ColorJitter(brightness=(0.7,1.3), contrast=(0.7,1.3), saturation=(0.7,1.3), hue=(-0.5,0.5))]
    )
    #transformations_img = None

    # Define transformations to apply to both img and mask
    transformations_both = {
        'crop_resize': {
            'scale':(0.3, 0.9),
            'ratio':(1.0,1.0)
        },
        'random_hflip':{'p':0.5},
        'random_perspective':{'distortion_scale': 0.5 }
    }
    # transformations_both = None

    # WANDB LOG
    if log_wandb:
        logger.info('setting wandb logging system')
        wandb.init(
            project="autoencoder-finetuning", 
            entity="kitkars", 
            name=name,
            config={
                "pt_name":f'AutoEncoder_{name}.pt',
                "learning_rate": cfg.hyperparameters.learning_rate,
                "epochs": cfg.hyperparameters.num_epochs,
                "batch_size": cfg.hyperparameters.batch_size,
                "weight_decay": cfg.hyperparameters.weight_decay,
                "data_real_processing": cfg.data_augmentation.data_real,
                "ratio_synthetic_data": cfg.data_augmentation.synthetic_data_ratio,
                "nb_duplicate": cfg.data_augmentation.nb_train_valid_duplicate,
                "gamma_exponential_scheduler": cfg.hyperparameters.gamma,
                #"eta_min_cosine_scheduler": cfg.hyperparameters.eta_min,
                "transformations_img": str(transformations_img),
                "transformations_both": str(transformations_both)
            }
        )
    else:
        logger.info('NO WANDB LOG')
    
    # Set torch device
    device = torch.device(f'cuda:{cuda}')
    
    # Load Datasets
    logger.info('loading datasets')
    train_dataset, valid_dataset, _ = split_dataset(
        cfg.data_paths.clean_data, 
        cfg.data_paths.test_set_filenames,
        transform_img=transformations_img,
        transform_both=transformations_both,
        data_real=cfg.data_augmentation.data_real,
        synthetic_data_ratio=cfg.data_augmentation.synthetic_data_ratio,
        train_valid_duplicate=cfg.data_augmentation.nb_train_valid_duplicate
    )
    
    print('Size of training set:', len(train_dataset))
    print('Size of validation set:', len(valid_dataset))
    
    batch_size=cfg.hyperparameters.batch_size
    
    # Get dataloaders
    logger.info('creating dataloaders')
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=cfg.hyperparameters.num_workers,
        shuffle=True,
        drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        num_workers=cfg.hyperparameters.num_workers,
        shuffle=True,
        drop_last=False
    )

    
    # Load model
    logger.info('load AutoEncoder model')
    model = AutoEncoder(
        n_channels=cfg.autoencoder_parameters.nb_input_channels, 
        n_classes=cfg.autoencoder_parameters.nb_output_channels
        )
    model = model.to(device)
    
    # Test the forward pass with dummy data
    logger.info('testing with dummy data')
    out = model(torch.randn(10, 3, 256, 256, device=device))['x_hat']
    assert out.size() == (10, cfg.autoencoder_parameters.nb_output_channels, 256, 256)
    
    # Set optimizer and scheduler
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr = cfg.hyperparameters.learning_rate, 
        weight_decay = cfg.hyperparameters.weight_decay,
        eps=1e-6
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.hyperparameters.T_max)
    #scheduler  = ExponentialLR(optimizer, gamma=cfg.hyperparameters.gamma)
    
    # Training loop
    num_epochs = cfg.hyperparameters.num_epochs
    validation_every_steps = cfg.hyperparameters.validation_every_steps

    step = 0
    cur_loss = 0

    model.train()

    train_dice_scores, valid_dice_score = [], []

    max_valid_dice_score, best_model = None, None
            
    for epoch in tqdm(range(num_epochs)):
        
        train_dice_scores_batches = []
        cur_loss = 0
        model.train()
        
        for rgb_img, mask_img in train_loader:
            # rgb_img, mask_img = aDictionary['image'], aDictionary['mask']
            rgb_img, mask_img = rgb_img.to(device), mask_img.to(device)
            
            # Forward pass, compute gradients, perform one training step.
            optimizer.zero_grad()
            
            output = model(rgb_img)['x_hat']
            
            batch_loss = loss_fn(
                output.flatten(start_dim=2, end_dim=len(output.size())-1), 
                mask_img.flatten(start_dim=1, end_dim=len(mask_img.size())-1).type(torch.long)
                )
            batch_loss.backward()
            optimizer.step()
            
            cur_loss += batch_loss  
        
            # Increment step counter
            step += 1
            
            # Compute DICE score.
            predictions = output.flatten(start_dim=2, end_dim=len(output.size())-1)
            train_dice_scores_batches.append(
                dice_score(
                    predictions, 
                    mask_img.flatten(start_dim=1, end_dim=len(mask_img.size())-1)
                )
            )
            
            if step % validation_every_steps == 0:
                
                # Append average training DICE score to list.
                train_dice_scores.append(np.mean(train_dice_scores_batches))
                
                train_dice_scores_batches = []
            
                # Compute DICE scores on validation set.
                valid_dice_scores_batches = []
                valid_loss = 0
                with torch.no_grad():
                    model.eval()
                    for rgb_img, mask_img in valid_loader:
                        rgb_img, mask_img = rgb_img.to(device), mask_img.to(device)
                        output = model(rgb_img)['x_hat']
                        loss = loss_fn(
                            output.flatten(start_dim=2, end_dim=len(output.size())-1), 
                            mask_img.flatten(start_dim=1, end_dim=len(mask_img.size())-1).type(torch.long)
                        )
                        valid_loss += loss  

                        predictions = output.flatten(start_dim=2, end_dim=len(output.size())-1)

                        # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
                        valid_dice_scores_batches.append(
                            dice_score(
                                predictions, 
                                mask_img.flatten(start_dim=1, end_dim=len(mask_img.size())-1)
                            ) * len(rgb_img)
                        )

                        # Keep the best model
                        if (max_valid_dice_score == None) or (valid_dice_scores_batches[-1] > max_valid_dice_score):
                            max_valid_dice_score = valid_dice_scores_batches[-1]
                            best_model = model.state_dict()

                    model.train()
                    
                # Append average validation DICE score to list.
                valid_dice_score.append(np.sum(valid_dice_scores_batches) / len(valid_dataset))
        
                print(f"Step {step:<5}   training DICE score: {train_dice_scores[-1]}")
                print(f"             valid DICE score: {valid_dice_score[-1]}")
                
                # wandb log
                if log_wandb:
                    wandb.log({
                        "valid_dice_score": valid_dice_score[-1],
                        "valid_loss": valid_loss.cpu().detach().numpy() / len(valid_dataset),
                        "training_dice_score": train_dice_scores[-1],
                    })
        scheduler.step()
        if log_wandb:            
            wandb.log({
                "training_loss": cur_loss.cpu().detach().numpy() / len(train_dataset),
                "learning_rate": scheduler.get_last_lr()[0]
            })
        
        if ((name == 'expFinal') and (epoch % 100 == 0)):
            # for final experiment, save intermediate models every 100 epochs
            logger.info(f'intermediate saving, epoch {epoch}')
            torch.save(model.state_dict(), cfg.model_paths.models+f'AutoEncoder_{name}_epoch{epoch}.pt')     
        
    logger.info('FINISHED training')
    
    # Save model
    model.load_state_dict(best_model)
    torch.save(model.state_dict(), cfg.model_paths.models+f'AutoEncoder_{name}.pt')     
    wandb.watch(model) #optional

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
