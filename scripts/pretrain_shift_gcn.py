"""Pretrain the 2D Shift-GCN backbone on the labeled seen training samples of one split."""

import argparse
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.shift_gcn import ShiftGCN
from scripts.train import add_split_arguments, load_config, load_seen_training_data, save_checkpoint, split_output_dir


def main(args):
    config = load_config(args.config)
    training = config['training']
    dataset, split = load_seen_training_data(config, args)
    num_joints = config['dataset']['num_joints']
    seen_classes = split['seen_classes']
    device = torch.device(training['device'] if torch.cuda.is_available() else 'cpu')
    backbone = ShiftGCN(num_joints).to(device)
    classifier = nn.Linear(backbone.output_dim, len(seen_classes)).to(device)
    targets_by_label = torch.full((max(seen_classes) + 1,), -1, device=device, dtype=torch.long)
    targets_by_label[seen_classes] = torch.arange(len(seen_classes), device=device)
    optimizer = torch.optim.AdamW(
        [*backbone.parameters(), *classifier.parameters()],
        lr=training['learning_rate'], weight_decay=training['weight_decay'],
    )
    dataloader = DataLoader(
        dataset, batch_size=training['batch_size'], shuffle=True, num_workers=training['num_workers'],
    )
    epochs = args.epochs if args.epochs is not None else training['num_epochs']
    for epoch in range(1, epochs + 1):
        loss_total = 0.0
        for skeleton, labels in tqdm(dataloader, desc='Shift-GCN pretraining'):
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(backbone(skeleton.to(device)))
            loss = F.cross_entropy(logits, targets_by_label[labels.to(device)])
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
        print(f'Epoch {epoch}/{epochs}: loss={loss_total / len(dataloader):.4f}')

    checkpoint_path = Path(args.output) if args.output else split_output_dir(config, split) / 'shift_gcn.pth'
    save_checkpoint({
        'split': split,
        'num_joints': num_joints,
        'backbone_state_dict': backbone.state_dict(),
    }, checkpoint_path)
    print(f'Saved {checkpoint_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_split_arguments(parser)
    parser.add_argument('--epochs', type=int, help='Override the pretraining epoch count.')
    parser.add_argument('--output', help='Output checkpoint path.')
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error('--epochs must be positive')
    main(args)
