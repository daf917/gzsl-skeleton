# Testing script for GZSL skeleton action recognition

import argparse
import os
import sys
import yaml
import torch
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
            logging.FileHandler(os.path.join(log_dir, 'test.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path: str, config: dict, device: torch.device) -> GZSLModel:
    """Load model from checkpoint"""
    model = GZSLModel(config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def test(model, dataloader, evaluator, device, text_features_global):
    """Test model"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for skeleton, labels in tqdm(dataloader, desc='Testing'):
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
    
    # Try different gamma values
    best_gamma, best_metrics = evaluator.find_optimal_gamma(all_labels, all_scores)
    
    print(f"\n=== Test Results ===")
    print(f"Without calibration:")
    print(f"  Acc_s: {metrics['Acc_s']:.2f}%")
    print(f"  Acc_u: {metrics['Acc_u']:.2f}%")
    print(f"  HM: {metrics['HM']:.2f}%")
    
    print(f"\nWith optimal gamma ({best_gamma}):")
    print(f"  Acc_s: {best_metrics['Acc_s']:.2f}%")
    print(f"  Acc_u: {best_metrics['Acc_u']:.2f}%")
    print(f"  HM: {best_metrics['HM']:.2f}%")
    
    return metrics, best_gamma, best_metrics


def main(args):
    # Load config
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config['logging']['log_dir'])
    logger.info(f"Starting testing with config: {config}")
    
    # Set device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, config, device)
    logger.info(f"Loaded model from {args.checkpoint}")
    
    # Create dataset
    dataset_name = config['dataset']['name']
    split = GZSLSplit(dataset_name)
    
    seen_classes = split.get_seen_classes()
    unseen_classes = split.get_unseen_classes()
    
    # Test dataset
    test_dataset = create_dataset(
        dataset_name=dataset_name,
        data_dir=config['dataset']['data_dir'],
        split='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    # Create evaluator
    evaluator = GZSLEvaluator(seen_classes, unseen_classes)
    
    # TODO: Load text features
    text_features_global = None
    
    # Test
    metrics, best_gamma, best_metrics = test(
        model, test_loader, evaluator, device, text_features_global
    )
    
    logger.info("Testing completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test GZSL model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    args = parser.parse_args()
    
    main(args)
