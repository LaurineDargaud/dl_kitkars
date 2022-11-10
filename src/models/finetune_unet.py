# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import hydra
from tqdm import tqdm

from src.data.DeloitteDataset import split_dataset

from src.models.unet import UNet
from src.models.unet import OutConv

import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim

@click.command()
@hydra.main(version_base=None, config_path='conf', config_name="config_unet")
def main(cfg, cuda=0):
    """ Fine tuning our U-Net pretrained model - baseline
    """
    logger = logging.getLogger(__name__)
    logger.info('finetune UNet pretrained model')
    
    # Set torch device
    device = torch.device(f'cuda:{cuda}')
    
    # Load Datasets
    logger.info('loading datasets')
    train_dataset, valid_dataset, test_dataset = split_dataset(cfg.data_paths.clean_data, cfg.data_paths.test_set_filenames)
    
    # Get dataloaders
    logger.info('creating dataloaders')
    train_loader = DataLoader(train_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=False)
    
    # Load model
    logger.info('load U-net pretrained model')
    model = UNet(n_channels=3, n_classes=2)
    model.load_state_dict(torch.load(cfg.model_paths.unet_scale_05))
    
    # Replace final outc layer
    model.outc = OutConv(64, cfg.unet_parameters.nb_output_channels)
    model = model.to(device)
    
    # Test the forward pass with dummy data
    logger.info('testing with dummy data')
    out = model(torch.randn(10, 3, 32, 32, device=device))
    print("Output shape:", out.size())
    assert out.size() == (10, cfg.unet_parameters.nb_output_channels, 32, 32)
    print(f"Output logits:\n{out.detach().cpu().numpy()}")
    
    # Set optimizer and scheduler
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1E-3)
    
    # Freeze some parameters
    logger.info('freezing wanted parameters')
    for name, param in model.named_parameters():
        name_keyword = name.split('.')[0]
        if name_keyword in cfg.unet_parameters.to_finetune:
            print('To finetune:', name)
            param.required_grad = True
        else:
            param.required_grad = False
    
    # Training loop
    # TODO: weights & bias dashboard
    
    batch_size = cfg.hyperparameters.batch_size
    num_epochs = cfg.hyperparameters.num_epochs
    validation_every_steps = cfg.hyperparameters.validation_every_steps

    step = 0
    cur_loss = 0

    model.train()

    train_accuracies = []
    valid_accuracies = []

    loss_list = []

    max_valid_accuracy, best_model = None, None
            
    for epoch in tqdm(range(num_epochs)):
        
        train_accuracies_batches = []
        cur_loss = 0
        model.train()
        
        for aDictionary in train_loader:
            rgb_img, mask_img = aDictionary['image'], aDictionary['mask']
            rgb_img, mask_img = rgb_img.to(device), mask_img.to(device)
            
            # Forward pass, compute gradients, perform one training step.
            optimizer.zero_grad()
            
            output = model(inputs)
            
            batch_loss = loss_fn(output, mask_img.flatten(start_dim=1, end_dim=len(mask_img.size())))
            batch_loss.backward()
            optimizer.step()
            
            cur_loss += batch_loss  
        
            # Increment step counter
            step += 1
            
            # Compute accuracy.
            predictions = output.max(1)[1]
            train_accuracies_batches.append(accuracy(targets, predictions))
            
            if step % validation_every_steps == 0:
                
                # Append average training accuracy to list.
                train_accuracies.append(np.mean(train_accuracies_batches))
                
                train_accuracies_batches = []
            
                # Compute accuracies on validation set.
                valid_accuracies_batches = []
                with torch.no_grad():
                    model.eval()
                    for inputs, targets in test_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        output = model(inputs)
                        loss = loss_fn(output, targets)

                        predictions = output.max(1)[1]

                        # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
                        valid_accuracies_batches.append(accuracy(targets, predictions) * len(inputs))

                        # Keep the best model
                        if (max_valid_accuracy == None) or (valid_accuracies_batches[-1] > max_valid_accuracy):
                            max_valid_accuracy = valid_accuracies_batches[-1]
                            best_model = model.state_dict()

                    model.train()
                    
                # Append average validation accuracy to list.
                valid_accuracies.append(np.sum(valid_accuracies_batches) / len(test_set))
        
                print(f"Step {step:<5}   training accuracy: {train_accuracies[-1]}")
                print(f"             test accuracy: {valid_accuracies[-1]}")
                
        loss_list.append(cur_loss / batch_size)

    print("Finished training.")
    
    # Save model and performance results
    
    
    
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
