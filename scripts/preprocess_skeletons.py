"""
Preprocess skeleton data for reproducible GZSL experiments.

Supported inputs:
- NTU RGB+D .skeleton files.
- PKU-MMD .skeleton + .label files.
- Generic NPZ files containing skeletons and labels.
- COCO/OpenPose-style 2D pose JSON directories.

The output is a normalized NPZ with skeletons, labels, split masks, optional
motion attributes, and training-set normalization statistics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.motion_attribute import MotionAttributeExtractor, create_part_joint_mapping  # noqa: E402


def resample_sequence(sequence: np.ndarray, max_frames: int) -> np.ndarray:
    if len(sequence) == 0:
        raise ValueError("Cannot resample an empty skeleton sequence.")
    if len(sequence) == max_frames:
        return sequence
    indices = np.linspace(0, len(sequence) - 1, max_frames).round().astype(np.int64)
    return sequence[indices]


def select_main_actor(sequence: np.ndarray) -> np.ndarray:
    """Select the tracked body with the largest coordinate variance."""
    if sequence.ndim == 3:
        return sequence
    if sequence.ndim != 4:
        raise ValueError(f"Expected (T,J,C) or (T,M,J,C), got {sequence.shape}")
    motion_scores = np.nanvar(sequence[..., :3], axis=(0, 2, 3))
    return sequence[:, int(np.argmax(motion_scores))]


def center_and_scale(sequence: np.ndarray, center_joint: int = 0, eps: float = 1e-8) -> np.ndarray:
    sequence = sequence.astype(np.float32, copy=True)
    origin = sequence[0:1, center_joint:center_joint + 1, : sequence.shape[-1]]
    sequence -= origin
    scale = np.nanmax(np.linalg.norm(sequence[..., :2], axis=-1))
    if np.isfinite(scale) and scale > eps:
        sequence /= scale
    return np.nan_to_num(sequence)


def parse_ntu_skeleton(path: Path) -> np.ndarray:
    """Parse one NTU RGB+D .skeleton file into (T,M,25,3)."""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]
    cursor = 0
    num_frames = int(lines[cursor])
    cursor += 1
    frames: List[np.ndarray] = []
    max_bodies = 0

    for _ in range(num_frames):
        num_bodies = int(lines[cursor])
        cursor += 1
        bodies = []
        for _body_idx in range(num_bodies):
            cursor += 1  # body metadata
            num_joints = int(lines[cursor])
            cursor += 1
            joints = []
            for _joint_idx in range(num_joints):
                values = [float(v) for v in lines[cursor].split()]
                cursor += 1
                joints.append(values[:3])
            bodies.append(np.asarray(joints, dtype=np.float32))
        max_bodies = max(max_bodies, len(bodies))
        frames.append(np.stack(bodies) if bodies else np.zeros((0, 25, 3), dtype=np.float32))

    max_bodies = max(max_bodies, 1)
    output = np.zeros((num_frames, max_bodies, 25, 3), dtype=np.float32)
    for t, bodies in enumerate(frames):
        output[t, : bodies.shape[0], : bodies.shape[1], :] = bodies
    return output


def class_from_ntu_name(path: Path) -> int:
    name = path.stem
    marker = name.rfind("A")
    if marker < 0:
        raise ValueError(f"Cannot find NTU action marker A### in {path.name}")
    return int(name[marker + 1: marker + 4]) - 1


def parse_pku_skeleton(path: Path) -> np.ndarray:
    """Parse one PKU-MMD skeleton file into (T,2,25,3)."""
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            values = [float(v) for v in line.split()]
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float32)
            if arr.size < 150:
                padded = np.zeros(150, dtype=np.float32)
                padded[: arr.size] = arr
                arr = padded
            rows.append(arr[:150].reshape(2, 25, 3))
    if not rows:
        raise ValueError(f"No skeleton rows found in {path}")
    return np.stack(rows, axis=0)


def parse_pku_labels(path: Path) -> List[Tuple[int, int, int]]:
    """Return zero-based (label, start, end) tuples from PKU-MMD label files."""
    intervals = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            values = [int(float(v)) for v in line.split()]
            if len(values) >= 3:
                intervals.append((values[0] - 1, values[1], values[2]))
    return intervals


def load_coco_json_dir(path: Path, num_joints: int = 17) -> np.ndarray:
    frames = []
    for json_path in sorted(path.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        people = data.get("people", [])
        if not people:
            frames.append(np.zeros((num_joints, 3), dtype=np.float32))
            continue
        keypoints = people[0].get("pose_keypoints_2d", [])
        arr = np.asarray(keypoints, dtype=np.float32).reshape(-1, 3)[:num_joints]
        frames.append(arr)
    if not frames:
        raise ValueError(f"No JSON frames found in {path}")
    return np.stack(frames, axis=0)


def compute_attributes(skeletons: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    attrs = []
    extractor = MotionAttributeExtractor(num_parts=6)
    for sequence in skeletons:
        extractor.part_joints = create_part_joint_mapping(sequence.shape[1])
        tensor = torch.from_numpy(sequence).float()
        attr = extractor.compute_all_parts(tensor)
        attrs.append(attr.numpy())
    attrs_np = np.stack(attrs).astype(np.float32)
    flat = attrs_np.reshape(-1, attrs_np.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-8
    attrs_np = (attrs_np - mean.reshape(1, 1, 1, -1)) / std.reshape(1, 1, 1, -1)
    return attrs_np, mean.astype(np.float32), std.astype(np.float32)


def load_split_ids(split_file: Path | None) -> Dict[str, set[int]]:
    if split_file is None:
        return {"seen": set(), "unseen": set()}
    payload = json.loads(split_file.read_text(encoding="utf-8"))
    return {
        "seen": {int(item["class_id"]) for item in payload.get("seen_classes", [])},
        "unseen": {int(item["class_id"]) for item in payload.get("unseen_classes", [])},
    }


def split_mask(labels: np.ndarray, split_ids: Dict[str, set[int]]) -> np.ndarray:
    """Return 0 for seen/trainable classes and 2 for unseen classes."""
    mask = np.zeros(labels.shape[0], dtype=np.int64)
    unseen = split_ids.get("unseen", set())
    if unseen:
        mask[np.isin(labels, list(unseen))] = 2
    return mask


def build_from_ntu(raw_dir: Path, max_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    skeletons, labels = [], []
    for path in sorted(raw_dir.rglob("*.skeleton")):
        sequence = select_main_actor(parse_ntu_skeleton(path))
        skeletons.append(center_and_scale(resample_sequence(sequence, max_frames)))
        labels.append(class_from_ntu_name(path))
    return np.stack(skeletons).astype(np.float32), np.asarray(labels, dtype=np.int64)


def build_from_pku(raw_dir: Path, max_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    skeletons, labels = [], []
    for skel_path in sorted(raw_dir.rglob("*.skeleton")):
        label_path = skel_path.with_suffix(".label")
        if not label_path.exists():
            continue
        sequence = parse_pku_skeleton(skel_path)
        for label, start, end in parse_pku_labels(label_path):
            clip = sequence[max(0, start): max(start + 1, end + 1)]
            skeletons.append(center_and_scale(resample_sequence(select_main_actor(clip), max_frames)))
            labels.append(label)
    return np.stack(skeletons).astype(np.float32), np.asarray(labels, dtype=np.int64)


def build_from_npz(input_path: Path, max_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(input_path, allow_pickle=True)
    skeletons = [center_and_scale(resample_sequence(select_main_actor(x), max_frames)) for x in data["skeletons"]]
    return np.stack(skeletons).astype(np.float32), np.asarray(data["labels"], dtype=np.int64)


def build_from_coco(raw_dir: Path, max_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    skeletons, labels = [], []
    for class_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        try:
            label = int(class_dir.name.split("_")[0])
        except ValueError as exc:
            raise ValueError(f"COCO pose class folders must start with integer label: {class_dir}") from exc
        for sample_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            sequence = load_coco_json_dir(sample_dir)
            skeletons.append(center_and_scale(resample_sequence(sequence, max_frames)))
            labels.append(label)
    return np.stack(skeletons).astype(np.float32), np.asarray(labels, dtype=np.int64)


def save_output(output: Path, skeletons: np.ndarray, labels: np.ndarray, split_file: Path | None, with_attributes: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "skeletons": skeletons,
        "labels": labels,
        "split": split_mask(labels, load_split_ids(split_file)),
    }
    if with_attributes:
        attributes, attr_mean, attr_std = compute_attributes(skeletons)
        payload.update({"motion_attributes": attributes, "attribute_mean": attr_mean, "attribute_std": attr_std})
    np.savez_compressed(output, **payload)
    print(f"Saved {len(labels)} samples to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess skeleton datasets for GZSL.")
    parser.add_argument("--format", choices=["ntu", "pku_mmd", "npz", "coco_json"], required=True)
    parser.add_argument("--input", required=True, help="Raw directory or NPZ file.")
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument("--split-file", default=None, help="Optional split JSON from data/splits.")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--no-attributes", action="store_true", help="Skip motion-attribute extraction.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.format == "ntu":
        skeletons, labels = build_from_ntu(input_path, args.max_frames)
    elif args.format == "pku_mmd":
        skeletons, labels = build_from_pku(input_path, args.max_frames)
    elif args.format == "npz":
        skeletons, labels = build_from_npz(input_path, args.max_frames)
    else:
        skeletons, labels = build_from_coco(input_path, args.max_frames)

    save_output(
        Path(args.output),
        skeletons,
        labels,
        Path(args.split_file) if args.split_file else None,
        with_attributes=not args.no_attributes,
    )


if __name__ == "__main__":
    main()
