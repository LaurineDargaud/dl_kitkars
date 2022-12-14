# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from torchvision.transforms import ColorJitter
from transformers import SegformerFeatureExtractor
from transformers import SegformerForSemanticSegmentation

import hydra
from tqdm import tqdm
import numpy as np

from src.data.DeloitteDataset import split_dataset

from src.models.performance_metrics import dice_score

from src.visualization.visualization_fct import get_mask_names

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torch import nn
from torchvision import transforms
import torch.optim as optim

import wandb

def resize_logits(logits, size=(256,256)):
    # logits shape (batch_size, num_labels, height/4, width/4)
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=size, # (height, width)
        mode='bilinear',
        align_corners=False
    )
    return upsampled_logits

@click.command()
@hydra.main(version_base=None, config_path='conf', config_name="config_segformer4e")
def main(cfg):
    """ Fine tuning a SegFormer pretrained model
    """
    logger = logging.getLogger(__name__)
    logger.info('finetune SegFormer pretrained model')
    
    cuda, name, log_wandb = cfg.cuda, cfg.name, cfg.log_wandb
    pretrained_model_name = cfg.segformer_parameters.pretrained_name

    # Define image transformations
    # transformations_img = None
    transformations_img = transforms.Compose(
        [ColorJitter(brightness=(0.7,1.3), contrast=(0.7,1.3), saturation=(0.7,1.3), hue=(-0.5,0.5))]
    )

    # Define transformations to apply to both img and mask
    # transformations_both = None
    transformations_both = {
        'crop_resize': {
            'scale':(0.3, 0.9),
            'ratio':(1.0,1.0)
        },
        'random_hflip':{'p':0.5},
        'random_perspective':{'distortion_scale': 0.5 }
    }
    
    # WANDB LOG
    if log_wandb:
        logger.info('setting wandb logging system')
        wandb.init(
            project="segformer-finetuning", 
            entity="kitkars", 
            name=name,
            config={
                "pretrained_model_name":pretrained_model_name,
                "learning_rate": cfg.hyperparameters.learning_rate,
                "epochs": cfg.hyperparameters.num_epochs,
                "batch_size": cfg.hyperparameters.batch_size,
                "weight_decay": cfg.hyperparameters.weight_decay,
                #"finetuned_parameters": cfg.segformer_parameters.to_finetune,
                "data_real_processing": cfg.data_augmentation.data_real,
                "ratio_synthetic_data": cfg.data_augmentation.synthetic_data_ratio,
                "nb_duplicate": cfg.data_augmentation.nb_train_valid_duplicate,
                "gamma_exponential_scheduler": cfg.hyperparameters.gamma,
                "transformations_img": str(transformations_img),
                "transformations_both": str(transformations_both)
            }
        )
    else:
        logger.info('NO WANDB LOG')
    
    # Set torch device
    device = torch.device(f'cuda:{cuda}')
    
    # Load Datasets with Segformer feature extractor
    logger.info('loading datasets with feature extraction')
    
    feature_extractor = SegformerFeatureExtractor()
    
    train_dataset, valid_dataset, _ = split_dataset(
        cfg.data_paths.clean_data, 
        cfg.data_paths.test_set_filenames,
        transform_img=transformations_img,
        transform_both=transformations_both,
        feature_extractor=feature_extractor,
        data_real=cfg.data_augmentation.data_real,
        synthetic_data_ratio=cfg.data_augmentation.synthetic_data_ratio,
        train_valid_duplicate=cfg.data_augmentation.nb_train_valid_duplicate 
    )

    
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
    logger.info('load SegFormer pretrained model')
    id2label = get_mask_names()
    label2id = {v: k for k, v in id2label.items()}
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name,
        id2label=id2label,
        label2id=label2id
    )
    model = model.to(device)
    
    # Test the forward pass with dummy data
    logger.info('testing with dummy data')
    out = model(torch.randn(10, 3, 32, 32, device=device)).logits
    out = resize_logits(out, size=(32,32))
    assert out.size() == (10, cfg.segformer_parameters.nb_output_channels, 32, 32)
    
    # Set optimizer and schedulerd
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr = cfg.hyperparameters.learning_rate, 
        weight_decay = cfg.hyperparameters.weight_decay,
        eps=1e-6
    )
    
    #scheduler = CosineAnnealingLR(optimizer, T_max=cfg.hyperparameters.T_max)
    scheduler = ExponentialLR(optimizer, gamma=cfg.hyperparameters.gamma)
    
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
            
            output = model(rgb_img).logits
            output = resize_logits(output, size=(mask_img.size(-2),mask_img.size(-1)))
            
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
                        output = model(rgb_img).logits
                        output = resize_logits(output, size=(mask_img.size(-2),mask_img.size(-1)))
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
                print(f"             test DICE score: {valid_dice_score[-1]}")
                
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
            
        if (epoch % 20 == 0):
            # save intermediate models every 20 epochs
            logger.info(f'intermediate saving, epoch {epoch}')
            torch.save(model.state_dict(), cfg.model_paths.models+f'segformer_finetuned_{name}_epoch{epoch}.pt')     
        
        
    logger.info('FINISHED training')
    
    # Save model
    model.load_state_dict(best_model)
    torch.save(model.state_dict(), cfg.model_paths.models+f'segformer_finetuned_{name}.pt')     
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
