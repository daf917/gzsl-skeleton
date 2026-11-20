"""Load normalized 2D skeletons and explicit GZSL class partitions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SAMPLE_SPLITS = {"train": 0, "val": 1, "test": 2}
DATASET_SPECS = {
    "ntu60": (60, 25),
    "ntu120": (120, 25),
    "pku_mmd": (51, 25),
    "ucf101": (101, 17),
    "hmdb51": (51, 17),
    "ssv2": (174, 17),
}


def canonical_dataset_name(name):
    key = name.lower().replace("-", "_")
    aliases = {"hmdb_51": "hmdb51", "ss_v2": "ssv2"}
    return aliases.get(key, key)


class SkeletonDataset(Dataset):
    """Read a split from an NPZ created by scripts/preprocess_skeletons.py."""

    DATASET_NAME = None

    def __init__(self, data_dir, split="train", data_file=None, *, dataset_name=None):
        self.dataset_name = canonical_dataset_name(dataset_name or self.DATASET_NAME or "")
        if self.dataset_name not in DATASET_SPECS:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        num_classes, num_joints = DATASET_SPECS[self.dataset_name]
        self.split = split
        if split not in SAMPLE_SPLITS:
            raise ValueError(f"Unknown sample partition {split!r}. Use train, val, or test.")

        path = Path(data_file) if data_file is not None else Path(data_dir) / f"{self.dataset_name}_skeletons.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Preprocessed skeleton data not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            missing = {"skeletons", "labels", "split"} - set(data.files)
            if missing:
                raise ValueError(f"{path} is missing arrays: {', '.join(sorted(missing))}")
            skeletons, labels, split_ids = data["skeletons"], data["labels"], data["split"]
            if skeletons.ndim != 4 or skeletons.shape[1:] != (64, num_joints, 2):
                raise ValueError(f"Expected (N, 64, {num_joints}, 2) image-plane skeletons, got {skeletons.shape} in {path}")
            num_samples = len(skeletons)
            if not np.isfinite(skeletons).all():
                raise ValueError(f"Skeletons in {path} contain nonfinite coordinates.")
            if labels.shape != (num_samples,) or not np.issubdtype(labels.dtype, np.integer):
                raise ValueError(f"Expected one integer class label per skeleton in {path}.")
            if np.any((labels < 0) | (labels >= num_classes)):
                raise ValueError(f"Labels in {path} must be zero-based and below {num_classes}.")
            if (
                split_ids.shape != (num_samples,) or not np.issubdtype(split_ids.dtype, np.integer)
                or not np.isin(split_ids, tuple(SAMPLE_SPLITS.values())).all()
            ):
                raise ValueError(f"Expected one explicit sample partition (0 train, 1 val, 2 test) per skeleton in {path}.")
            mask = split_ids == SAMPLE_SPLITS[split]
            if not mask.any():
                raise ValueError(f"No {split} samples in {path}.")

            self.samples = skeletons[mask].astype(np.float32, copy=False)
            self.labels = labels[mask].astype(np.int64, copy=False)
            self.sample_ids = None
            if "sample_ids" in data:
                sample_ids = data["sample_ids"]
                if sample_ids.shape != (num_samples,) or np.unique(sample_ids).size != num_samples:
                    raise ValueError(f"Expected distinct sample_ids in {path}.")
                self.sample_ids = sample_ids[mask]
            self.attribute_mean = self.attribute_std = None
            if "attribute_mean" in data or "attribute_std" in data:
                if "attribute_mean" not in data or "attribute_std" not in data:
                    raise ValueError(f"Both attribute_mean and attribute_std must be present in {path}.")
                self.attribute_mean = data["attribute_mean"]
                self.attribute_std = data["attribute_std"]
                if (
                    self.attribute_mean.shape != (8,) or self.attribute_std.shape != (8,)
                    or not np.isfinite(self.attribute_mean).all()
                    or not np.isfinite(self.attribute_std).all()
                    or (self.attribute_std < 0).any()
                ):
                    raise ValueError(f"Expected eight finite attribute means and nonnegative standard deviations in {path}.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.from_numpy(self.samples[index]), int(self.labels[index])

    @classmethod
    def get_class_name(cls, label):
        prompts = Path(__file__).resolve().parent / "prompts" / f"{cls.DATASET_NAME}.json"
        if not prompts.is_file():
            raise FileNotFoundError(f"Class names are not available in {prompts}.")
        records = json.loads(prompts.read_text(encoding="utf-8"))
        return {entry["class_id"]: entry["action_class"] for entry in records}[label]


class NTU60Dataset(SkeletonDataset):
    DATASET_NAME = "ntu60"
    NUM_CLASSES, NUM_JOINTS = DATASET_SPECS[DATASET_NAME]


class NTU120Dataset(NTU60Dataset):
    DATASET_NAME = "ntu120"
    NUM_CLASSES, NUM_JOINTS = DATASET_SPECS[DATASET_NAME]


class PKUMMDDataset(SkeletonDataset):
    DATASET_NAME = "pku_mmd"
    NUM_CLASSES, NUM_JOINTS = DATASET_SPECS[DATASET_NAME]


class UCF101Dataset(SkeletonDataset):
    DATASET_NAME = "ucf101"
    NUM_CLASSES, NUM_JOINTS = DATASET_SPECS[DATASET_NAME]


class HMDB51Dataset(SkeletonDataset):
    DATASET_NAME = "hmdb51"
    NUM_CLASSES, NUM_JOINTS = DATASET_SPECS[DATASET_NAME]


class SSv2Dataset(SkeletonDataset):
    DATASET_NAME = "ssv2"
    NUM_CLASSES, NUM_JOINTS = DATASET_SPECS[DATASET_NAME]


class GZSLSplit:
    """Load a released class split independently of sample membership."""

    SPLITS = {
        "ntu60": {"seen": 55, "unseen": 5},
        "ntu120": {"seen": 110, "unseen": 10},
        "ucf101": {"seen": 80, "unseen": 21},
        "pku_mmd": {"seen": 46, "unseen": 5},
        "hmdb51": {"seen": 31, "unseen": 20},
    }

    def __init__(self, dataset_name, split_type="random", split_index=1, split_dir=None):
        self.dataset_name = canonical_dataset_name(dataset_name)
        self.split_type = split_type
        self.split_index = split_index
        self.split_info = self.SPLITS.get(self.dataset_name)
        if self.dataset_name not in DATASET_SPECS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        if split_type not in {"random", "provided"} or split_index not in {1, 2, 3}:
            raise ValueError("Expected random or provided class protocol with split index 1, 2, or 3.")
        split_dir = Path(split_dir) if split_dir is not None else Path(__file__).resolve().parent / "splits"
        self.path = split_dir / self.dataset_name / f"{split_type}_split_{split_index}.json"
        if not self.path.is_file():
            raise FileNotFoundError(f"Class split manifest not found: {self.path}")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("dataset", self.dataset_name) != self.dataset_name:
            raise ValueError(f"Dataset field does not match {self.dataset_name} in {self.path}.")
        if payload.get("split_index", split_index) != split_index:
            raise ValueError(f"Split index does not match {split_index} in {self.path}.")
        if payload.get("protocol", f"3-split {split_type}") not in {split_type, f"3-split {split_type}"}:
            raise ValueError(f"Protocol does not match {split_type} in {self.path}.")
        try:
            self._seen_classes = [entry["class_id"] for entry in payload["seen_classes"]]
            self._unseen_classes = [entry["class_id"] for entry in payload["unseen_classes"]]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Class split must supply seen_classes and unseen_classes with class_id values in {self.path}.") from exc
        labels = self.get_all_classes()
        num_classes, _ = DATASET_SPECS[self.dataset_name]
        if (
            not self._seen_classes or not self._unseen_classes or len(labels) != num_classes
            or any(type(label) is not int for label in labels) or set(labels) != set(range(num_classes))
        ):
            raise ValueError(f"Class split must partition all {num_classes} zero-based labels exactly once in {self.path}.")
        if self.split_info and (len(self._seen_classes), len(self._unseen_classes)) != (
            self.split_info["seen"], self.split_info["unseen"]
        ):
            raise ValueError(f"Class split sizes do not match the documented protocol in {self.path}.")

    def get_seen_classes(self):
        return list(self._seen_classes)

    def get_unseen_classes(self):
        return list(self._unseen_classes)

    def get_all_classes(self):
        return self._seen_classes + self._unseen_classes


class FewShotDataset(Dataset):
    """
    Few-shot dataset for few-shot learning experiments
    """

    def __init__(self,
                 base_dataset: Dataset,
                 num_shots: int = 16,
                 num_way: int = 5,
                 split: str = 'train'):
        """
        Args:
            base_dataset: Base dataset to sample from
            num_shots: Number of samples per class
            num_way: Number of classes per episode
            split: 'train', 'val', or 'test'
        """
        self.base_dataset = base_dataset
        self.num_shots = num_shots
        self.num_way = num_way
        self.split = split

        # Sample few-shot episodes
        self.episodes = self._create_episodes()

    def _create_episodes(self):
        """Create few-shot episodes"""
        episodes = []

        # Get class distribution
        unique_labels = np.unique(self.base_dataset.labels)

        # Create episodes
        for _ in range(1000):  # Number of episodes
            # Sample classes
            selected_classes = np.random.choice(unique_labels, self.num_way, replace=False)

            # Sample shots for each class
            episode = []
            for cls in selected_classes:
                # Get samples for this class
                cls_indices = np.where(self.base_dataset.labels == cls)[0]

                # Sample shots
                selected_indices = np.random.choice(
                    cls_indices,
                    min(self.num_shots, len(cls_indices)),
                    replace=False
                )

                for idx in selected_indices:
                    episode.append((idx, cls))

            episodes.append(episode)

        return episodes

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> tuple[list[torch.Tensor], list[int]]:
        """Get a few-shot episode"""
        episode = self.episodes[idx]

        skeletons = []
        labels = []

        for sample_idx, label in episode:
            skeleton, _ = self.base_dataset[sample_idx]
            skeletons.append(skeleton)
            labels.append(label)

        return skeletons, labels


_DATASET_CLASSES = {
    dataset.DATASET_NAME: dataset for dataset in
    (NTU60Dataset, NTU120Dataset, PKUMMDDataset, UCF101Dataset, HMDB51Dataset, SSv2Dataset)
}


def create_dataset(dataset_name, data_dir, split="train", **kwargs):
    name = canonical_dataset_name(dataset_name)
    if name not in _DATASET_CLASSES:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return _DATASET_CLASSES[name](data_dir, split, **kwargs)


def download_ntu60(data_dir):
    """Report the source of the NTU RGB+D 60 dataset."""
    print("Download NTU RGB+D 60 from https://github.com/shahroudy/NTURGB-D")
    print(f"Place the raw skeleton data in {data_dir}.")
