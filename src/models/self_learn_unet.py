# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from torchvision.transforms import ColorJitter

import hydra
from tqdm import tqdm
import numpy as np

from src.data.DeloitteDataset import split_dataset

from src.models.unet import UNet
from src.models.unet import OutConv

from src.models.performance_metrics import dice_score

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch import nn
from torchvision import transforms
import torch.optim as optim

import wandb

@click.command()
@hydra.main(version_base=None, config_path='conf', config_name="config_unet_self_learning")
def main(cfg):
    """ Self-learning for best fine-tuned Unet
    """
    logger = logging.getLogger(__name__)
    logger.info('finetune UNet pretrained model')
    
    cuda, name, log_wandb = cfg.cuda, cfg.name, cfg.log_wandb
    
    # Define image transformations
    transformations_img = transforms.Compose(
        [ColorJitter(brightness=(0.7,1.3), contrast=(0.7,1.3), saturation=(0.7,1.3), hue=(-0.5,0.5))]
    )
    
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
            project="unet-finetuning", 
            entity="kitkars", 
            name=name,
            config={
                "pt_name":f'unet_finetuned_{name}.pt',
                "learning_rate": cfg.hyperparameters.learning_rate,
                "epochs": cfg.hyperparameters.num_epochs,
                "batch_size": cfg.hyperparameters.batch_size,
                "weight_decay": cfg.hyperparameters.weight_decay,
                "finetuned_parameters": cfg.unet_parameters.to_finetune,
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
    
    # Load Datasets
    logger.info('loading datasets')
    # keep the same validation set as previous experiments
    _, valid_dataset, _ = split_dataset(
        cfg.data_paths.clean_data, 
        cfg.data_paths.test_set_filenames,
        transform_img=None,
        transform_both=None,
        data_real=cfg.data_augmentation.data_real,
        synthetic_data_ratio=cfg.data_augmentation.synthetic_data_ratio,
        train_valid_duplicate=cfg.data_augmentation.nb_train_valid_duplicate
    )
    
    # training set with all new pictures
    train_dataset, _, _ = split_dataset(
        cfg.data_paths.unlabeled_data, 
        cfg.data_paths.test_set_filenames,
        transform_img=transformations_img,
        transform_both=transformations_both,
        data_real=False,
        synthetic_data_ratio=0.0,
        train_valid_duplicate=1
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
    if cfg.reuse_finetune == 'None':
        logger.info('load U-net pretrained model')
        teacher_model = UNet(n_channels=3, n_classes=2)
        teacher_model.load_state_dict(torch.load(cfg.model_paths.unet_scale_05))
        # Replace final outc layer
        teacher_model.outc = OutConv(64, cfg.unet_parameters.nb_output_channels)
        teacher_model = teacher_model.to(device)
    else:
        logger.info(f'load U-net finetuned model: {cfg.reuse_finetune}')
        teacher_model = UNet(n_channels=3, n_classes=cfg.unet_parameters.nb_output_channels)
        teacher_model.load_state_dict(torch.load(cfg.model_paths.models+f'unet_finetuned_{cfg.reuse_finetune}.pt'))
        teacher_model = teacher_model.to(device)
    
    # Build student model
    student_model = UNet(n_channels=3, n_classes=cfg.unet_parameters.nb_output_channels)
    student_model.load_state_dict(teacher_model.state_dict())
    student_model = student_model.to(device)
    
    # Test the forward pass with dummy data
    logger.info('testing with dummy data')
    student_out = student_model(torch.randn(10, 3, 32, 32, device=device))
    teacher_out = teacher_model(torch.randn(10, 3, 32, 32, device=device))
    assert student_out.size() == (10, cfg.unet_parameters.nb_output_channels, 32, 32)
    assert teacher_out.size() == (10, cfg.unet_parameters.nb_output_channels, 32, 32)
    
    # Set optimizer and scheduler
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        student_model.parameters(), 
        lr = cfg.hyperparameters.learning_rate, 
        weight_decay = cfg.hyperparameters.weight_decay,
        eps=1e-6
    )
    
    num_epochs = cfg.hyperparameters.num_epochs
    
    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg.hyperparameters.T_max)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    #scheduler  = ExponentialLR(optimizer, gamma=cfg.hyperparameters.gamma)
    #intermSavingModelEpochNb = cfg.hyperparameters.T_max
    intermSavingModelEpochNb = 50
    
    # Freeze some parameters
    logger.info('freezing wanted parameters')
    for aName, param in student_model.named_parameters():
        name_keyword = aName.split('.')[0]
        if (cfg.unet_parameters.to_finetune == 'all') or (name_keyword in cfg.unet_parameters.to_finetune):
            print('To finetune:', aName)
            param.required_grad = True
        else:
            param.required_grad = False
    
    # Training loop
    validation_every_steps = cfg.hyperparameters.validation_every_steps
    
    # Parameter momentum to update teacher's weights with Exponential Moving Average (EMA)
    ema_momentum = cfg.self_learning.momentum

    step = 0
    cur_loss = 0

    train_dice_scores, valid_dice_score = [], []

    max_valid_dice_score, best_model = None, None
            
    for epoch in tqdm(range(num_epochs)):
        
        train_dice_scores_batches = []
        cur_loss = 0
        
        # Student model is training, teacher is not
        student_model.train()
        teacher_model.eval()
        
        for rgb_img, _ in train_loader:
            # rgb_img, mask_img = aDictionary['image'], aDictionary['mask']
            rgb_img = rgb_img.to(device)
            
            # Predict with teacher model to get the ground truth
            predicted_ground_truth = teacher_model(rgb_img)
            mask_img = torch.argmax(predicted_ground_truth,dim=1)
            mask_img = mask_img.to(device)
            
            # Forward pass, compute gradients, perform one training step.
            optimizer.zero_grad()
            
            # Train student model
            output = student_model(rgb_img)
            
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
                
                # Validation performed on TEACHER model
                
                # Append average training DICE score to list.
                train_dice_scores.append(np.mean(train_dice_scores_batches))
                
                train_dice_scores_batches = []
            
                # Compute DICE scores on validation set.
                valid_dice_scores_batches = []
                valid_loss = 0
                with torch.no_grad():
                    teacher_model.eval()
                    for rgb_img, mask_img in valid_loader:
                        rgb_img, mask_img = rgb_img.to(device), mask_img.to(device)
                        output = teacher_model(rgb_img)
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
                            best_model = teacher_model.state_dict()
                    
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
        
        if (epoch % intermSavingModelEpochNb == 0):
            # for self learning experiment, save intermediate models every 100 epochs
            logger.info(f'intermediate saving, epoch {epoch}')
            torch.save(best_model, cfg.model_paths.models+f'unet_finetuned_{name}_epoch{epoch}.pt')     
        
        # At the end of each epoch: update teacher's weight with Exponential Moving Average with student model
        student_params_dict = student_model.state_dict()
        with torch.no_grad():
            for aName, param in teacher_model.named_parameters():
                param.data = ema_momentum*param.data + (1-ema_momentum)*student_params_dict[aName]
        
    logger.info('FINISHED training')
    
    # Save model
    teacher_model.load_state_dict(best_model)
    torch.save(teacher_model.state_dict(), cfg.model_paths.models+f'unet_finetuned_{name}.pt')     
    wandb.watch(teacher_model) #optional

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
