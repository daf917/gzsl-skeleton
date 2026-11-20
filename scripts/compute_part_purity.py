"""Compute manuscript PartPurity from assigned per-part motion modes."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def compute_part_purity(labels, motion_modes, class_ids):
    """Return each part's mean class-wise dominant-mode fraction."""
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be a one-dimensional integer array.")
    if (
        motion_modes.ndim != 2
        or motion_modes.shape[0] != labels.size
        or motion_modes.shape[1] == 0
        or not np.issubdtype(motion_modes.dtype, np.integer)
        or (motion_modes < 0).any()
    ):
        raise ValueError("motion_modes must contain nonnegative integer mode IDs with shape (samples, parts).")
    if not class_ids:
        raise ValueError("At least one evaluated class is required.")
    missing = sorted(set(class_ids) - set(labels.tolist()))
    if missing:
        raise ValueError(f"No mode assignments for evaluated class IDs {missing}.")

    totals = np.zeros(motion_modes.shape[1])
    for class_id in class_ids:
        class_modes = motion_modes[labels == class_id]
        for part, modes in enumerate(class_modes.T):
            _, counts = np.unique(modes, return_counts=True)
            totals[part] += counts.max() / modes.size
    return totals / len(class_ids)


def read_assignments(path, split):
    with np.load(path, allow_pickle=False) as arrays:
        labels = arrays["labels"]
        modes = arrays["motion_modes"]
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be a one-dimensional integer array.")
    if not np.isin(labels, split.get_all_classes()).all():
        raise ValueError(f"{path} contains labels outside the selected class split.")
    return labels, modes


def main(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data import GZSLSplit

    split = GZSLSplit(args.dataset, args.split_protocol, args.split_index, split_dir=args.split_dir)
    labels, modes = read_assignments(args.assignments, split)
    seen = compute_part_purity(labels, modes, split.get_seen_classes())
    unseen = compute_part_purity(labels, modes, split.get_unseen_classes())
    scores = {"seen": seen, "unseen": unseen}

    if args.baseline is not None:
        baseline_labels, baseline_modes = read_assignments(args.baseline, split)
        baseline = compute_part_purity(baseline_labels, baseline_modes, split.get_unseen_classes())
        if baseline.shape != unseen.shape:
            raise ValueError("The full model and baseline must use the same number and order of parts.")
        scores["baseline_unseen"] = baseline

    print(json.dumps({
        "dataset": split.dataset_name,
        "protocol": split.split_type,
        "split_index": split.split_index,
        **{
            group: {"per_part": values.tolist(), "mean": float(values.mean())}
            for group, values in scores.items()
        },
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-protocol", choices=("random", "provided"), default="random")
    parser.add_argument("--split-index", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--split-dir", type=Path, help="Override the class manifest directory.")
    parser.add_argument("--assignments", type=Path, required=True,
                        help="NPZ with labels and motion_modes for all evaluated classes.")
    parser.add_argument("--baseline", type=Path,
                        help="Optional equivalent NPZ for the composition-removed baseline.")
    try:
        main(parser.parse_args())
    except (KeyError, OSError, ValueError) as exc:
        parser.error(str(exc))
