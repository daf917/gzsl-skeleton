# Training script for GZSL skeleton action recognition

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import GZSLModel
from data import create_dataset, GZSLSplit
from utils import GZSLEvaluator


def setup_logging(log_dir: str):
    """Setup logging"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(model, dataloader, optimizer, device, config, class_names, part_descriptions):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    loss_components = {
        'global_align': 0.0,
        'part_align': 0.0,
        'consistency': 0.0,
        'independence': 0.0
    }
    
    pbar = tqdm(dataloader, desc='Training')
    
    for batch_idx, (skeleton, labels) in enumerate(pbar):
        skeleton = skeleton.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model.forward(skeleton)
        
        # Compute loss
        loss, loss_dict = model.compute_loss(
            outputs,
            labels,
            class_names,
            part_descriptions
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        for k, v in loss_dict.items():
            if k != 'total':
                loss_components[k] += v
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    # Average losses
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}
    
    return avg_loss, avg_components


def validate(model, dataloader, evaluator, device, text_features_global):
    """Validate model"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for skeleton, labels in tqdm(dataloader, desc='Validation'):
            skeleton = skeleton.to(device)
            labels = labels.to(device)
            
            # Get predictions
            outputs = model.forward(skeleton)
            scores = outputs['global_composed'] @ text_features_global.T
            
            preds = scores.argmax(dim=-1)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_scores.append(scores.cpu())
    
    # Concatenate
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_scores = torch.cat(all_scores)
    
    # Compute metrics
    metrics = evaluator.evaluate_scores(all_labels, all_scores)
    
    return metrics


def main(args):
    # Load config
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config['logging']['log_dir'])
    logger.info(f"Starting training with config: {config}")
    
    # Set device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataset
    dataset_name = config['dataset']['name']
    split = GZSLSplit(dataset_name)
    
    seen_classes = split.get_seen_classes()
    unseen_classes = split.get_unseen_classes()
    
    logger.info(f"Seen classes: {len(seen_classes)}, Unseen classes: {len(unseen_classes)}")
    
    # Create data loaders
    train_dataset = create_dataset(
        dataset_name=dataset_name,
        data_dir=config['dataset']['data_dir'],
        split='train'
    )
    
    val_dataset = create_dataset(
        dataset_name=dataset_name,
        data_dir=config['dataset']['data_dir'],
        split='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    # Create model
    model = GZSLModel(config)
    model = model.to(device)
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs']
    )
    
    # Create evaluator
    evaluator = GZSLEvaluator(seen_classes, unseen_classes)
    
    # TODO: Load class names and part descriptions
    class_names = {}
    part_descriptions = {}
    text_features_global = None
    
    # Training loop
    best_hm = 0.0
    
    for epoch in range(config['training']['num_epochs']):
        logger.info(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss, train_components = train_epoch(
            model, train_loader, optimizer, device, config,
            class_names, part_descriptions
        )
        
        logger.info(f"Train Loss: {train_loss:.4f}")
        for k, v in train_components.items():
            logger.info(f"  {k}: {v:.4f}")
        
        # Validate
        if (epoch + 1) % config['evaluation']['test_interval'] == 0:
            val_metrics = validate(model, val_loader, evaluator, device, text_features_global)
            
            logger.info(f"Validation Metrics:")
            logger.info(f"  Acc_s: {val_metrics['Acc_s']:.2f}")
            logger.info(f"  Acc_u: {val_metrics['Acc_u']:.2f}")
            logger.info(f"  HM: {val_metrics['HM']:.2f}")
            
            # Save best model
            if val_metrics['HM'] > best_hm:
                best_hm = val_metrics['HM']
                
                checkpoint_path = os.path.join(
                    config['logging']['checkpoint_dir'],
                    'best_model.pth'
                )
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': val_metrics
                }, checkpoint_path)
                
                logger.info(f"Saved best model with HM: {best_hm:.2f}")
        
        # Step scheduler
        scheduler.step()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GZSL model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    main(args)
