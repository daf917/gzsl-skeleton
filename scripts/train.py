"""Train the aligned recognition model for one GZSL class split."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import create_dataset, GZSLSplit
from data.text_prompts import load_class_prompts
from models import GZSLModel


def add_split_arguments(parser):
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--split-protocol', choices=('random', 'provided'), default='random')
    parser.add_argument('--split-index', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--data-file', help='Preprocessed skeleton NPZ for this split.')


def load_config(config_path):
    with open(config_path, encoding='utf-8') as file:
        return yaml.safe_load(file)


def load_seen_training_data(config, args):
    dataset_config = config['dataset']
    dataset_name = dataset_config['name']
    manifest = GZSLSplit(
        dataset_name, args.split_protocol, args.split_index, split_dir=dataset_config.get('split_dir'),
    )
    seen_classes = manifest.get_seen_classes()
    unseen_classes = manifest.get_unseen_classes()
    split = {
        'dataset': dataset_name,
        'protocol': args.split_protocol,
        'split_index': args.split_index,
        'seen_classes': list(seen_classes),
        'unseen_classes': list(unseen_classes),
    }
    dataset = create_dataset(
        dataset_name, dataset_config['data_dir'], split='train',
        data_file=args.data_file or dataset_config.get('data_file'),
    )
    labels = np.asarray(dataset.labels)
    indices = np.flatnonzero(np.isin(labels, seen_classes))
    missing = sorted(set(seen_classes) - set(labels[indices].tolist()))
    if missing:
        raise ValueError(f'No training samples for seen class IDs {missing}.')
    sample, _ = dataset[int(indices[0])]
    dataset_config['num_joints'] = sample.shape[-2]
    dataset_config['num_classes'] = len(seen_classes) + len(unseen_classes)
    return Subset(dataset, indices.tolist()), split


def split_output_dir(config, split):
    return (Path(config['logging']['checkpoint_dir']) / split['dataset']
            / f"{split['protocol']}_split_{split['split_index']}")


def save_checkpoint(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def fit_motion_statistics(model, dataloader, device):
    total = torch.zeros(8, dtype=torch.float64, device=device)
    squares = torch.zeros_like(total)
    count = 0
    for skeleton, _ in tqdm(dataloader, desc='Motion statistics'):
        attrs = model.extract_motion_attributes(skeleton.to(device), normalize=False)
        values = attrs.reshape(-1, 8).double()
        total += values.sum(0)
        squares += values.square().sum(0)
        count += values.shape[0]
    mean = total / count
    std = (squares / count - mean.square()).clamp_min(0).sqrt()
    model.set_motion_attribute_statistics(mean, std)


def train_epoch(model, dataloader, optimizer, device, class_names, part_descriptions, seen_classes):
    model.train()
    loss_total = 0.0
    for skeleton, labels in tqdm(dataloader, desc='Training'):
        skeleton = skeleton.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(skeleton)
        loss, _ = model.compute_loss(
            outputs, labels, class_names, part_descriptions, seen_classes=seen_classes,
        )
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
    return loss_total / len(dataloader)


def main(args):
    config = load_config(args.config)
    training = config['training']
    dataset, split = load_seen_training_data(config, args)
    output_dir = split_output_dir(config, split)
    encoder_config = config.setdefault('skeleton_encoder', {})
    pretrained_path = Path(
        args.shift_gcn_checkpoint or encoder_config.get('shift_gcn_pretrained')
        or output_dir / 'shift_gcn.pth',
    )
    if not pretrained_path.is_file():
        raise FileNotFoundError(
            f'{pretrained_path} is missing. Run scripts/pretrain_shift_gcn.py for the same split.',
        )
    pretrained = torch.load(pretrained_path, map_location='cpu', weights_only=True)
    if (pretrained.get('split') != split
            or pretrained.get('num_joints') != config['dataset']['num_joints']):
        raise ValueError('The Shift-GCN checkpoint does not match this dataset and class split.')
    encoder_config['shift_gcn_pretrained'] = None

    device = torch.device(training['device'] if torch.cuda.is_available() else 'cpu')
    model = GZSLModel(config).to(device)
    model.skeleton_encoder.shift_gcn.load_state_dict(pretrained['backbone_state_dict'])
    seen_classes = split['seen_classes']
    class_names, part_descriptions = load_class_prompts(
        split['dataset'], seen_classes, config.get('prompt', {}).get('descriptions_dir'),
    )
    loader_options = {
        'batch_size': training['batch_size'], 'num_workers': training['num_workers'],
    }
    fit_motion_statistics(model, DataLoader(dataset, **loader_options), device)
    dataloader = DataLoader(dataset, shuffle=True, **loader_options)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=training['learning_rate'], weight_decay=training['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, training['num_epochs'])
    for epoch in range(1, training['num_epochs'] + 1):
        average_loss = train_epoch(
            model, dataloader, optimizer, device, class_names, part_descriptions, seen_classes,
        )
        print(f'Epoch {epoch}/{training["num_epochs"]}: loss={average_loss:.4f}')
        scheduler.step()

    checkpoint_path = Path(args.output) if args.output else output_dir / 'model.pth'
    save_checkpoint({
        'epoch': training['num_epochs'],
        'config': config,
        'split': split,
        'attribute_mean': model.attr_mean.cpu(),
        'attribute_std': model.attr_std.cpu(),
        'model_state_dict': model.state_dict(),
    }, checkpoint_path)
    print(f'Saved {checkpoint_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_split_arguments(parser)
    parser.add_argument('--shift-gcn-checkpoint', help='Pretraining checkpoint for this split.')
    parser.add_argument('--output', help='Output checkpoint path.')
    main(parser.parse_args())
