"""Create a split-aware NPZ of normalized image-plane skeleton sequences."""

from __future__ import annotations

import argparse
from itertools import chain
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import SAMPLE_SPLITS
from data.motion_attribute import MotionAttributeExtractor, create_part_joint_mapping


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _manifest(path):
    if path is None:
        return None
    mapping = _read_json(path)
    if not isinstance(mapping, dict) or not mapping or any(
        not isinstance(key, str) or not key or not isinstance(value, str) or value not in SAMPLE_SPLITS
        for key, value in mapping.items()
    ):
        raise ValueError(f"{path} must map sample IDs to train, val, or test.")
    return {key: SAMPLE_SPLITS[value] for key, value in mapping.items()}


def _relative_id(path, root):
    return path.relative_to(root).with_suffix("").as_posix()


def _selected(ids, sample_id):
    return ids is None or sample_id in ids


def _uniform_indices(length, count):
    if length == 0:
        raise ValueError("Cannot resample an empty skeleton sequence.")
    return np.linspace(0, length - 1, count).round().astype(np.int64)


def _fill_missing(sequence, sample_id):
    """Interpolate lost observations on the original timeline."""
    sequence = np.asarray(sequence, dtype=np.float32).copy()
    frames = np.arange(len(sequence))
    for joint in range(sequence.shape[1]):
        observed = np.isfinite(sequence[:, joint, :]).all(axis=-1)
        if not observed.any():
            raise ValueError(f"Joint {joint} was never tracked in sample {sample_id}.")
        for channel in range(2):
            sequence[:, joint, channel] = np.interp(
                frames, frames[observed], sequence[observed, joint, channel]
            )
    return sequence


def _main_actor(sequence):
    """Keep one tracked actor for the clip, preferring greater motion."""
    if sequence.ndim == 3:
        return sequence
    if sequence.ndim != 4 or sequence.shape[1] == 0:
        raise ValueError(f"Expected a skeleton clip shaped (T,J,C) or (T,M,J,C), got {sequence.shape}.")
    scores = []
    for track in sequence.transpose(1, 0, 2, 3):
        valid = np.isfinite(track).all(axis=(1, 2)) & np.any(track != 0, axis=(1, 2))
        if not valid.any():
            scores.append((-1, -1))
            continue
        # Compare temporal motion of the same joint, without treating untracked
        # frames or spatial differences between joints as motion.
        scores.append((float(np.var(track[valid], axis=0).mean()), int(valid.sum())))
    best = max(range(len(scores)), key=scores.__getitem__)
    if scores[best][0] < 0:
        raise ValueError("No tracked skeleton in the clip.")
    track = sequence[:, best].copy()
    track[~np.any(track != 0, axis=(1, 2))] = np.nan
    return track


def _normalize(sequence, sample_id, max_frames):
    """Fill missing frames, center at the initial body center, and scale."""
    if sequence.ndim != 3 or sequence.shape[-1] != 2 or sequence.shape[1] not in {17, 25}:
        raise ValueError(f"Expected joints in image-plane XY for {sample_id}, got {sequence.shape}.")
    sequence = _fill_missing(sequence, sample_id)
    sequence = sequence[_uniform_indices(len(sequence), max_frames)]
    center_joints = [0] if sequence.shape[1] == 25 else [11, 12]
    sequence -= sequence[0, center_joints].mean(axis=0)
    scale = np.linalg.norm(sequence, axis=-1).max()
    if scale > 0:
        sequence /= scale
    return sequence


def _ntu_sequence(path):
    """Read RGB-projected colorX/colorY and keep Kinect track IDs."""
    tracks = {}
    with path.open(encoding="utf-8") as source:
        frame_count = int(next(source))
        for frame in range(frame_count):
            for _ in range(int(next(source))):
                body_id = next(source).split()[0]
                joints = int(next(source))
                if joints != 25:
                    raise ValueError(f"Expected 25 joints in {path}, got {joints}.")
                if body_id not in tracks:
                    tracks[body_id] = np.full((frame_count, joints, 2), np.nan, dtype=np.float32)
                track = tracks[body_id]
                for joint in range(joints):
                    fields = next(source).split()
                    if len(fields) < 7:
                        raise ValueError(f"NTU joint has no RGB image-plane coordinates in {path}.")
                    track[frame, joint] = [float(fields[5]), float(fields[6])]
    if not tracks:
        raise ValueError(f"No tracked bodies in {path}.")
    return _main_actor(np.stack(list(tracks.values()), axis=1))


def _ntu_records(root, selected):
    for path in sorted(root.rglob("*.skeleton")):
        sample_id = _relative_id(path, root)
        if _selected(selected, sample_id):
            match = re.search(r"A(\d{3})$", path.stem)
            if not match:
                raise ValueError(f"Expected an NTU filename ending with A###: {path.name}")
            yield sample_id, int(match.group(1)) - 1, _ntu_sequence(path), None


def _load_camera_matrices(path):
    if path.suffix.lower() == ".npy":
        return {"default": np.load(path, allow_pickle=False)}
    if path.suffix.lower() == ".json":
        value = _read_json(path)
        return value if isinstance(value, dict) else {"default": value}
    return {"default": np.loadtxt(path)}


def _camera_for(sample_id, matrices):
    stem = sample_id.rsplit("/", 1)[-1]
    view = stem.rsplit("-", 1)[-1]
    for key in dict.fromkeys((sample_id, stem, view, "default")):
        if key in matrices:
            matrix = np.asarray(matrices[key], dtype=np.float64)
            if matrix.shape not in {(3, 3), (3, 4)} or not np.isfinite(matrix).all():
                raise ValueError(f"Camera calibration {key!r} must be a finite 3x3 or 3x4 matrix.")
            return matrix
    raise ValueError(f"No RGB camera calibration supplied for PKU recording {sample_id}.")


def _project_to_rgb(sequence, matrix):
    valid = np.isfinite(sequence).all(axis=-1) & np.any(sequence != 0, axis=-1)
    points = sequence
    if matrix.shape == (3, 4):
        points = np.concatenate((sequence, np.ones_like(sequence[..., :1])), axis=-1)
    homogeneous = points @ matrix.T
    depth = homogeneous[..., 2]
    valid &= np.isfinite(homogeneous).all(axis=-1) & (depth > 0)
    projected = np.full((*sequence.shape[:-1], 2), np.nan, dtype=np.float32)
    projected[valid] = homogeneous[valid, :2] / depth[valid, None]
    return projected


def _pku_labels(path):
    with path.open(encoding="utf-8-sig") as source:
        for line in source:
            if not line.strip():
                continue
            fields = re.split(r"[\s,]+", line.strip())
            if len(fields) not in {3, 4}:
                raise ValueError(f"Expected label,start,end[,confidence] in {path}: {line.strip()!r}")
            label, start, end = map(int, fields[:3])
            yield label - 1, start, end


def _pku_records(root, selected, annotations, matrices, frame_base):
    captures = sorted(chain(root.rglob("*.skeleton"), root.rglob("*.txt")))
    for path in captures:
        capture_id = _relative_id(path, root)
        if capture_id not in selected:
            continue
        relative = Path(capture_id)
        candidates = [
            annotations / relative.with_suffix(".txt"),
            annotations / relative.with_suffix(".label"),
            annotations / f"{path.stem}.txt",
            annotations / f"{path.stem}.label",
        ]
        annotation = next(
            (candidate for candidate in candidates if candidate.is_file() and candidate.resolve() != path.resolve()), None
        )
        if annotation is None:
            raise FileNotFoundError(f"No official PKU temporal annotation file for {capture_id}.")
        sequence = np.loadtxt(path, dtype=np.float32, ndmin=2)
        if sequence.shape[1] not in {75, 150}:
            raise ValueError(f"Expected 25 XYZ joints per actor in {path}, got {sequence.shape[1]} values per frame.")
        sequence = sequence.reshape(len(sequence), -1, 25, 3)
        matrix = _camera_for(capture_id, matrices)
        for index, (label, start, end) in enumerate(_pku_labels(annotation), 1):
            sample_id = f"{capture_id}:{index}"
            if sample_id not in selected[capture_id]:
                continue
            first, last = start - frame_base, end - frame_base
            if not 0 <= first <= last < len(sequence):
                raise ValueError(f"Interval [{start}, {end}] in {annotation} is outside the recording.")
            track = _main_actor(sequence[first:last + 1])
            yield sample_id, label, _project_to_rgb(track, matrix), None


def _pose_person(person):
    coordinates = person.get("pose_keypoints_2d", person.get("keypoints"))
    if coordinates is None:
        return None
    points = np.asarray(coordinates, dtype=np.float32)
    if points.ndim == 1 and points.size in {34, 51}:
        points = points.reshape(17, -1)
    if points.shape not in {(17, 2), (17, 3)}:
        raise ValueError(f"Expected 17 COCO-order HRNet keypoints per person, got {points.shape}.")
    score = person.get("keypoint_scores", points[:, 2] if points.shape[1] == 3 else np.ones(17))
    score = np.asarray(score, dtype=np.float32)
    if score.shape != (17,):
        raise ValueError("Expected one HRNet confidence score per keypoint.")
    xy = points[:, :2].copy()
    xy[~np.isfinite(score) | (score <= 0)] = np.nan
    return xy, float(np.nan_to_num(score).mean())


def _pose_frame(path):
    raw = _read_json(path)
    if isinstance(raw, dict):
        people = raw.get("people", raw.get("instances", raw.get("predictions", [raw])))
    else:
        people = raw
    if isinstance(people, list) and len(people) == 1 and isinstance(people[0], list):
        people = people[0]
    if not isinstance(people, list) or any(not isinstance(person, dict) for person in people):
        raise ValueError(f"Expected a list of 2D pose detections in {path}.")
    candidates = [entry for person in people if (entry := _pose_person(person)) is not None]
    if not candidates:
        return np.full((17, 2), np.nan, dtype=np.float32)
    return max(candidates, key=lambda entry: entry[1])[0]


def _pose_records(root, selected):
    def frame_order(path):
        return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.stem)]

    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for sample_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            sample_id = sample_dir.relative_to(root).as_posix()
            if not _selected(selected, sample_id):
                continue
            try:
                label = int(class_dir.name.split("_", 1)[0])
            except ValueError as exc:
                raise ValueError(f"Pose class directory must start with a zero-based label: {class_dir}") from exc
            frames = sorted(sample_dir.glob("*.json"), key=frame_order)
            if not frames:
                raise ValueError(f"No pose JSON files in {sample_dir}.")
            yield sample_id, label, np.stack([_pose_frame(path) for path in frames]), None


def _npz_records(path, selected):
    with np.load(path, allow_pickle=False) as data:
        sequences, labels = data["skeletons"], data["labels"]
        sample_ids = data.get("sample_ids", np.arange(len(labels)).astype(str))
        partitions = data.get("split")
        if sequences.ndim not in {4, 5} or sequences.shape[-1] != 2:
            raise ValueError("NPZ input must contain XY image-plane skeletons shaped (N,T,J,2) or (N,T,M,J,2).")
        if sample_ids.shape != (len(sequences),) or np.unique(sample_ids).size != len(sequences):
            raise ValueError("NPZ sample_ids must identify each skeleton exactly once.")
        if labels.shape != (len(sequences),) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("NPZ input must supply one integer label per skeleton.")
        if partitions is not None and (
            partitions.shape != (len(sequences),) or not np.isin(partitions, tuple(SAMPLE_SPLITS.values())).all()
        ):
            raise ValueError("NPZ split IDs must be 0 (train), 1 (val) or 2 (test).")
        for index, sample_id in enumerate(sample_ids.astype(str)):
            if _selected(selected, sample_id):
                partition = None if partitions is None else int(partitions[index])
                yield sample_id, int(labels[index]), _main_actor(sequences[index]), partition


def _collect(records, membership, max_frames):
    ids, labels, samples, splits = [], [], [], []
    for sample_id, label, sequence, partition in records:
        if membership is not None:
            partition = membership[sample_id]
        if partition is None:
            raise ValueError("Supply --sample-splits or a source NPZ with an explicit split array.")
        if label < 0:
            raise ValueError(f"Sample {sample_id} has a negative class label.")
        ids.append(sample_id)
        labels.append(label)
        samples.append(_normalize(sequence, sample_id, max_frames))
        splits.append(partition)
    if not samples:
        raise ValueError("No skeleton samples matched the supplied input.")
    if len(set(ids)) != len(ids):
        raise ValueError("Input sources contain duplicate sample IDs.")
    if membership is not None and (missing := set(membership) - set(ids)):
        raise ValueError(f"No input found for {len(missing)} manifest entries, including {min(missing)!r}.")
    return {
        "sample_ids": np.asarray(ids),
        "labels": np.asarray(labels, dtype=np.int64),
        "skeletons": np.stack(samples),
        "split": np.asarray(splits, dtype=np.int64),
    }


def _motion_attributes(payload, class_split, window):
    skeletons = payload["skeletons"]
    use_for_statistics = payload["split"] == SAMPLE_SPLITS["train"]
    if class_split:
        split = _read_json(class_split)
        seen_classes = [entry["class_id"] for entry in split["seen_classes"]]
        use_for_statistics &= np.isin(payload["labels"], seen_classes)
    if not use_for_statistics.any():
        raise ValueError("Motion statistics require samples in the selected seen-class training partition.")
    extractor = MotionAttributeExtractor(num_parts=6, temporal_window=window)
    extractor.part_joints = create_part_joint_mapping(skeletons.shape[2])
    with torch.inference_mode():
        attributes = np.stack([
            extractor.compute_all_parts(torch.from_numpy(sample)).numpy()
            for sample in skeletons
        ])
    train_attributes = attributes[use_for_statistics].reshape(-1, 8)
    mean = train_attributes.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_attributes.std(axis=0, dtype=np.float64).astype(np.float32)
    payload.update(
        motion_attributes=(attributes - mean) / (std + 1e-8),
        attribute_mean=mean,
        attribute_std=std,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=["ntu", "pku_mmd", "npz", "hrnet_json", "coco_json"], required=True)
    parser.add_argument("--input", type=Path, required=True, help="Raw directory or image-plane XY NPZ.")
    parser.add_argument("--output", type=Path, required=True, help="Destination .npz file.")
    parser.add_argument("--sample-splits", type=Path, help="JSON mapping sample IDs to train, val, or test.")
    parser.add_argument("--split-file", type=Path, help="Optional class manifest. Only seen training samples fit attribute statistics.")
    parser.add_argument("--annotations", type=Path, help="Directory of official PKU-MMD temporal annotations.")
    parser.add_argument("--camera-matrix", type=Path, help="PKU RGB projection matrix file or a JSON map of matrices.")
    parser.add_argument("--frame-index-base", "--pku-frame-base", type=int, choices=(0, 1), help="First frame index used by PKU-MMD annotations.")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--temporal-window", type=int, default=10)
    parser.add_argument("--no-attributes", action="store_true", help="Export skeletons without computing motion attributes.")
    args = parser.parse_args()

    if args.max_frames < 1 or args.temporal_window < 1:
        parser.error("Frame count and temporal window must be positive.")
    if args.format != "npz" and args.sample_splits is None:
        parser.error("--sample-splits is required for raw inputs.")
    membership = _manifest(args.sample_splits)
    if args.format == "ntu":
        records = _ntu_records(args.input, membership)
    elif args.format == "pku_mmd":
        if args.annotations is None or args.camera_matrix is None or args.frame_index_base is None:
            parser.error("PKU-MMD needs --annotations, --camera-matrix, and --frame-index-base.")
        requested = {}
        for sample_id in membership:
            recording, separator, index = sample_id.rpartition(":")
            if not recording or not separator or not index.isdigit() or int(index) < 1:
                parser.error(f"Expected PKU-MMD sample ID recording:interval_number, got {sample_id!r}.")
            requested.setdefault(recording, set()).add(sample_id)
        matrices = _load_camera_matrices(args.camera_matrix)
        records = _pku_records(args.input, requested, args.annotations, matrices, args.frame_index_base)
    elif args.format == "npz":
        records = _npz_records(args.input, membership)
    else:
        records = _pose_records(args.input, membership)
    payload = _collect(records, membership, args.max_frames)
    if not args.no_attributes:
        _motion_attributes(payload, args.split_file, args.temporal_window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"Saved {len(payload['labels'])} samples to {args.output}.")


if __name__ == "__main__":
    main()
