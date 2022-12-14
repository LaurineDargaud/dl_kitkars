# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import hydra
from tqdm import tqdm
import numpy as np

from src.visualization.visualization_fct import _MASK_NAMES_

from src.data.DeloitteDataset import split_dataset

from src.visualization.visualization_fct import get_mask_names

from transformers import SegformerFeatureExtractor
from transformers import SegformerForSemanticSegmentation

# from torchvision import transforms

from src.models.performance_metrics import dice_score, dice_score_class

from src.visualization.visualization_fct import mask_to_rgb

from src.models.finetune_segformer import resize_logits

import torch
from torch.utils.data import DataLoader
from torch import nn

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
    """ Predict test set with finetuned SegFormer model
    """
    cuda, name, log_wandb, dataset_to_predict = cfg.cuda, cfg.name_best, cfg.log_wandb, cfg.predict_dataset
    
    logger = logging.getLogger(__name__)
    logger.info(f'predict {dataset_to_predict} set with finetuned SegFormer model')
    
    # WANDB LOG
    if log_wandb:
        logger.info('setting wandb logging system')
        wandb.init(
            project="segformer-predictions", 
            entity="kitkars", 
            name=f'{name} ({dataset_to_predict})',
            config={
                "pt_name":f'segformer_finetuned_{name}.pt',
                "batch_size": cfg.hyperparameters.batch_size,
                "set_type": dataset_to_predict
            }
        )
        if cfg.predict_params.data_real:
            wandb.config.update({
                "data_real":cfg.predict_params.data_real,
                "synthetic_data_ratio":cfg.predict_params.synthetic_data_ratio,
                "train_valid_duplicate":cfg.predict_params.nb_train_valid_duplicate
            })
        else:
            wandb.config.update({
                "data_real":cfg.predict_params.data_real
            })
            
    else:
        logger.info('NO WANDB LOG')
    
    # Set torch device
    device = torch.device(f'cuda:{cuda}')
    
    # Define image transformations
    transformations_img = None

    # Define transformations to apply to both img and mask
    transformations_both = None
    
    # Load Datasets
    feature_extractor = SegformerFeatureExtractor()
    
    logger.info(f'loading {dataset_to_predict} set')
    train_dataset, valid_dataset, testing_dataset = split_dataset(
        cfg.data_paths.clean_data, 
        cfg.data_paths.test_set_filenames,
        transform_img=transformations_img,
        transform_both=transformations_both,
        feature_extractor=feature_extractor,
        train_ratio=1.0, valid_ratio=0.0,
        data_real=cfg.data_augmentation.data_real,
        synthetic_data_ratio=cfg.data_augmentation.synthetic_data_ratio,
        train_valid_duplicate=cfg.data_augmentation.nb_train_valid_duplicate
    )
    
    all_datasets = {
        'train':train_dataset, 'valid': valid_dataset, 'test': testing_dataset
    }
    
    test_dataset = all_datasets[dataset_to_predict]
    
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
    logger.info('load SegFormer finetuned model')
    pretrained_model_name = cfg.segformer_parameters.pretrained_name
    id2label = get_mask_names()
    label2id = {v: k for k, v in id2label.items()}
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name,
        id2label=id2label,
        label2id=label2id
    )
    model.load_state_dict(torch.load(cfg.model_paths.models+f'segformer_finetuned_{name}.pt'))
    model = model.to(device)
    
    # Set loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Make predictions    
    test_dice_scores_batches = []
    test_loss = 0
    all_predictions = []
    all_dice_scores = []
    
    dice_class = []
    all_dice_class = []
    
    logger.info(f'running {dataset_to_predict} predictions')
    
    with torch.no_grad():
        model.eval()
        for rgb_img, mask_img in tqdm(test_loader):
            rgb_img, mask_img = rgb_img.to(device), mask_img.to(device)
            output = model(rgb_img).logits
            output = resize_logits(output, size=(mask_img.size(-2),mask_img.size(-1)))
            
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
            
            # classes x batches
            dice_class.append(
                    dice_score_class(
                        predictions, 
                        mask_img.flatten(start_dim=1, end_dim=len(mask_img.size())-1)
                    )
                )
            
            for i in range(len(output)):
                all_predictions.append(output[i].cpu().detach().numpy())
                #DICE PER CLASS PER IMAGE INSIDE BATCH
                all_dice_scores.append(dice_score_class( # ADD FLOAT HERE IN CASE OF ANY ERROR
                            output[i].flatten(start_dim=1, end_dim=len(output.size())-2).unsqueeze(0),
                            mask_img[i].flatten(start_dim=0, end_dim=len(mask_img.size())-2).unsqueeze(0)
                            )
                        )

                all_dice_class.append(dice_score( # DICE PER IMAGE INSIDE BATCH
                        output[i].flatten(start_dim=1, end_dim=len(output.size())-2).unsqueeze(0),
                        mask_img[i].flatten(start_dim=0, end_dim=len(mask_img.size())-2).unsqueeze(0)
                    )
                )
    
    # Get performance metrics
    test_dice_score = np.sum(test_dice_scores_batches) / len(test_dataset)
    dice_class_average = np.array(dice_class).mean(0)
    print(f"Test DICE score: {test_dice_score}")
    print(f"Average DICE score per class: {dice_class_average}")

    
    # wandb log
    if log_wandb:
        
        overview = {
            "test_dice_score": test_dice_score,
            "test_loss": test_loss.cpu().detach().numpy() / len(test_dataset)
        }

        dice_class_average_list = []

        for i in range(len(dice_class_average)):
            s = dice_class_average[i]
            dice_class_average_list.append(s)
            overview[f'DICE_{i}_{_MASK_NAMES_[i]}'] = s

        wandb.log(overview)
        
        logger.info(f'creating wandb table for predictions visualization')
        
        # create a wandb.Table() with corresponding columns
        columns=["id", "filename", "RGB image", "features", "real mask", "prediction", "DICE score float", "DICE score"] + [f"DICE_{i}_{j}" for i,j in _MASK_NAMES_.items()]

        test_table = wandb.Table(columns=columns)
        
        for i in tqdm(range((len(test_dataset)))):            
            _, mask_img = test_dataset[i]
            rgb_image = test_dataset.get_rawImg(i)
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
            test_columns = [
                i, 
                filename, 
                wandb.Image(rgb_image), 
                wandb.Image(rgb_features), 
                wandb.Image(mask_img), 
                wandb.Image(predicted_mask_img),
                all_dice_scores[i],
                str(all_dice_class[i])] + [np.round(j*100, 3) for j in all_dice_scores[i]]

            test_table.add_data(*test_columns)
        
        wandb.log({"test_table": test_table})

        columns_dice_class = [str(f"{i}_{j}") for i,j in _MASK_NAMES_.items()]

        data = [[label, val] for (label, val) in zip(columns_dice_class, dice_class_average_list)]
        wandb.log({"Bar_chart": wandb.plot.bar(wandb.Table(data=data, columns = ["Classes", "Percentage"]) , "Classes", "Percentage", title= "Classes Bar Chart")})
    
    
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
