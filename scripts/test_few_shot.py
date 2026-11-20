"""Evaluate fixed K-shot prototypes on separate support and query partitions."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import FewShotModel, create_dataset
from scripts.test import load_model


@torch.no_grad()
def evaluate(model, support, query, shots, seed, device, batch_size):
    """Choose support samples without replacement and score the held-out queries."""
    if support.sample_ids is None or query.sample_ids is None:
        raise ValueError("The NPZ must contain distinct sample_ids for support and query items.")
    if np.intersect1d(support.sample_ids, query.sample_ids).size:
        raise ValueError("The support and query partitions contain overlapping sample IDs.")
    classes = np.unique(support.labels)
    if not np.array_equal(classes, np.unique(query.labels)):
        raise ValueError("Support and query partitions must cover the same class IDs.")

    rng = np.random.default_rng(seed)
    indexed_support = []
    for class_id in classes:
        class_indices = np.flatnonzero(support.labels == class_id)
        if len(class_indices) < shots:
            raise ValueError(f"Class {class_id} has fewer than {shots} distinct support samples.")
        indexed_support.append(rng.choice(class_indices, size=shots, replace=False))

    model.eval()
    few_shot = FewShotModel(model, num_way=len(classes))
    class_labels = torch.as_tensor(classes, device=device)
    prototypes = torch.cat([
        few_shot.compute_prototypes(
            torch.as_tensor(support.samples[indices], device=device),
            class_id.expand(shots),
        )
        for class_id, indices in zip(class_labels, indexed_support)
    ])

    correct = 0
    for skeletons, labels in DataLoader(query, batch_size=batch_size, shuffle=False):
        positions, _ = few_shot.few_shot_predict(skeletons.to(device), prototypes)
        correct += (class_labels[positions] == labels.to(device)).sum().item()
    return 100.0 * correct / len(query)


def main(args):
    # Both partitions come from the same prepared NPZ. The test labels do not
    # determine the candidate classes or select samples for the prototypes.
    support = create_dataset(args.dataset, args.data_file.parent, split="val", data_file=args.data_file)
    query = create_dataset(args.dataset, args.data_file.parent, split="test", data_file=args.data_file)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint.get("config")
    if config is None:
        raise ValueError("The checkpoint must include its original model configuration.")
    if config["dataset"]["num_joints"] != support.samples.shape[-2]:
        raise ValueError("The checkpoint and target dataset must use the same 2D joint layout.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(checkpoint, config, device)
    accuracy = evaluate(model, support, query, args.num_shots, args.seed, device, args.batch_size)
    print(f"{args.dataset}, {args.num_shots}-shot, seed {args.seed}: {accuracy:.1f}% top-1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("hmdb51", "ssv2"))
    parser.add_argument("--data-file", required=True, type=Path,
                        help="Prepared NPZ with support samples in split 1 and query samples in split 2")
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Trained model checkpoint with its configuration and motion statistics")
    parser.add_argument("--num-shots", type=int, choices=(2, 4, 8, 16), default=16)
    parser.add_argument("--seed", type=int, required=True, help="Seed for choosing support samples")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", help="Device for inference, for example cpu")
    args = parser.parse_args()

    try:
        main(args)
    except (KeyError, OSError, ValueError) as exc:
        parser.error(str(exc))
