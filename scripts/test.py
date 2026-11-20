# Testing script for GZSL skeleton action recognition

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch
    from models import GZSLModel

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.metrics import aggregate_gzsl_metrics


def report_split_results(results_path: Path) -> None:
    """Report one method and protocol from three unrounded per-split results."""
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected an object with dataset, protocol, and splits fields.")

    dataset = payload["dataset"]
    protocol = payload["protocol"]
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("dataset must be a nonempty name.")
    if protocol not in ("random", "provided"):
        raise ValueError("protocol must be random or provided.")

    splits = payload["splits"]
    if not isinstance(splits, list) or any(not isinstance(split, dict) for split in splits):
        raise ValueError("splits must be an array of result objects.")
    indices = [split["split_index"] for split in splits]
    if any(type(index) is not int for index in indices) or sorted(indices) != [1, 2, 3]:
        raise ValueError("Expected one result for each split_index: 1, 2, 3.")

    if any(
        split.get("dataset", dataset) != dataset or split.get("protocol", protocol) != protocol
        for split in splits
    ):
        raise ValueError("All three results must use the same dataset and split protocol.")

    results = aggregate_gzsl_metrics(splits)
    print(f"{dataset}, 3-split {protocol}")
    for metric, value in results.items():
        print(f"  {metric}: {value:.1f}%")


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
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(
    checkpoint: dict,
    config: dict,
    device: torch.device,
    attribute_statistics: Optional[Path] = None,
) -> GZSLModel:
    """Restore the trained model and its training-set normalization statistics."""
    from models import GZSLModel

    mean = checkpoint.get("attribute_mean")
    std = checkpoint.get("attribute_std")
    if mean is None or std is None:
        if attribute_statistics is None:
            raise ValueError(
                "Provide the training-set motion statistics with --attribute-statistics "
                "when the checkpoint does not contain them."
            )

        import numpy as np

        with np.load(attribute_statistics, allow_pickle=False) as stats:
            mean = stats["attribute_mean"]
            std = stats["attribute_std"]

    model_config = dict(config)
    model_config["skeleton_encoder"] = {
        **config.get("skeleton_encoder", {}),
        "shift_gcn_pretrained": None,
    }
    model = GZSLModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.set_motion_attribute_statistics(mean, std)
    model.eval()
    return model


def test(model, dataloader, evaluator, device, text_features_global, *, diagnostic_gamma_sweep=False):
    """Test model"""
    import torch
    from tqdm import tqdm

    model.eval()
    
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for skeleton, labels in tqdm(dataloader, desc='Testing'):
            skeleton = skeleton.to(device)
            scores = model.predict(skeleton, text_features_global, None)
            all_labels.append(labels.cpu())
            all_scores.append(scores.cpu())
    
    # Concatenate
    all_labels = torch.cat(all_labels)
    all_scores = torch.cat(all_scores)
    
    # Compute metrics
    metrics = evaluator.evaluate_scores(all_labels, all_scores)
    
    print("\n=== Test Results ===")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.1f}%")

    if diagnostic_gamma_sweep:
        best_gamma, best_metrics = evaluator.find_optimal_gamma(all_labels, all_scores)
        print(f"\nDiagnostic gamma sweep on test labels ({best_gamma}):")
        for metric, value in best_metrics.items():
            print(f"  {metric}: {value:.1f}%")

    return metrics


def main(args):
    import torch
    from torch.utils.data import DataLoader
    from data import create_dataset, GZSLSplit
    from data.text_prompts import load_class_prompts
    from utils import GZSLEvaluator

    # Load config
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config['logging']['log_dir'])
    logger.info(f"Starting testing with config: {config}")
    
    # Set device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataset
    dataset_name = config['dataset']['name']
    split = GZSLSplit(
        dataset_name,
        split_type=args.split_protocol,
        split_index=args.split_index,
        split_dir=config["dataset"].get("split_dir"),
    )
    
    seen_classes = split.get_seen_classes()
    unseen_classes = split.get_unseen_classes()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    split_info = {
        "dataset": dataset_name,
        "protocol": args.split_protocol,
        "split_index": args.split_index,
        "seen_classes": seen_classes,
        "unseen_classes": unseen_classes,
    }
    saved_split = checkpoint.get("split")
    if saved_split is not None and any(
        saved_split.get(key) != value for key, value in split_info.items()
    ):
        raise ValueError("The checkpoint was trained for a different dataset or class split.")
    
    # Test dataset
    test_dataset = create_dataset(
        dataset_name=dataset_name,
        data_dir=config['dataset']['data_dir'],
        split='test',
        data_file=args.data_file or config["dataset"].get("data_file"),
    )
    
    test_labels = set(test_dataset.labels)
    split_labels = set(seen_classes) | set(unseen_classes)
    if not test_labels <= split_labels:
        raise ValueError("Test labels must match the zero-based labels in the selected class split.")
    missing_classes = split_labels - test_labels
    if missing_classes:
        raise ValueError(f"No held-out test samples for class IDs {sorted(missing_classes)}.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    # Create evaluator
    evaluator = GZSLEvaluator(seen_classes, unseen_classes)
    
    model_config = checkpoint.get("config", config)
    expected_joints = model_config["dataset"]["num_joints"]
    if test_dataset.samples.shape[-2] != expected_joints:
        raise ValueError(
            f"Checkpoint expects {expected_joints} joints, "
            f"but this test dataset has {test_dataset.samples.shape[-2]}."
        )
    model = load_model(checkpoint, model_config, device, args.attribute_statistics)
    logger.info(f"Loaded model from {args.checkpoint}")
    text_features_global = checkpoint.get("text_features_global")
    if args.text_features:
        text_features_global = torch.load(args.text_features, map_location=device, weights_only=True)
    if text_features_global is None:
        class_ids = sorted(split_labels)
        class_names, part_descriptions = load_class_prompts(
            dataset_name, class_ids, config.get("prompt", {}).get("descriptions_dir")
        )
        with torch.no_grad():
            text_features_global, _ = model.encode_text(class_ids, class_names, part_descriptions)

    expected_shape = (len(split_labels), model.feature_dim)
    if (
        not isinstance(text_features_global, torch.Tensor)
        or text_features_global.shape != expected_shape
    ):
        raise ValueError(
            f"Expected global text features of shape {expected_shape}, "
            "with one row per zero-based class ID."
        )
    if not text_features_global.is_floating_point() or not torch.isfinite(text_features_global).all():
        raise ValueError("Global text features must contain finite floating-point values.")
    text_features_global = torch.nn.functional.normalize(text_features_global.float(), dim=-1)

    metrics = test(
        model,
        test_loader,
        evaluator,
        device,
        text_features_global,
        diagnostic_gamma_sweep=args.diagnostic_gamma_sweep,
    )
    if args.metrics_output:
        result = {
            "dataset": dataset_name,
            "protocol": args.split_protocol,
            "split_index": args.split_index,
            **metrics,
        }
        args.metrics_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    
    logger.info("Testing completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test GZSL model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--checkpoint', help='Path to checkpoint for one split')
    mode.add_argument('--aggregate-split-results', type=Path,
                      help='Report results using the three splits of one protocol')
    parser.add_argument('--split-protocol', choices=('random', 'provided'), default='random',
                        help='Class split protocol for checkpoint evaluation')
    parser.add_argument('--split-index', type=int, choices=(1, 2, 3), default=1,
                        help='Class split number for checkpoint evaluation')
    parser.add_argument('--data-file', help='Preprocessed skeleton NPZ with held-out test samples')
    parser.add_argument('--text-features',
                        help='Matching .pt tensor of global text features, indexed by class ID')
    parser.add_argument('--attribute-statistics', type=Path,
                        help='Training-set motion statistics NPZ for legacy checkpoints')
    parser.add_argument('--metrics-output', type=Path,
                        help='Write unrounded checkpoint metrics as JSON for aggregation')
    parser.add_argument('--diagnostic-gamma-sweep', action='store_true',
                        help='Also print the legacy diagnostic sweep on test labels')
    args = parser.parse_args()

    try:
        if args.aggregate_split_results:
            report_split_results(args.aggregate_split_results)
        else:
            main(args)
    except (KeyError, OSError, ValueError) as exc:
        parser.error(str(exc))
