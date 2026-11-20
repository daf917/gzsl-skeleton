"""Extract COCO-order 2D keypoints from RGB clip frames using MMPose HRNet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
COCO_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)


def _frame_order(path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.stem)]


def _clips(root):
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for clip_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            frames = sorted(
                (path for path in clip_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
                key=_frame_order,
            )
            if not frames:
                continue
            if not class_dir.name.split("_", 1)[0].isdigit():
                raise ValueError(f"Class directory must start with a zero-based label: {class_dir}")
            if len({path.stem.lower() for path in frames}) != len(frames):
                raise ValueError(f"Each frame must have a distinct filename stem: {clip_dir}")
            yield frames


def _box_manifest(path):
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Boxes must be a JSON object mapping relative frame paths to XYXY boxes.")
    result = {}
    for frame_id, values in payload.items():
        boxes = np.asarray(values, dtype=np.float32)
        if isinstance(values, list) and not values:
            boxes = boxes.reshape(0, 4)
        if (
            boxes.ndim != 2 or boxes.shape[1] != 4
            or not np.isfinite(boxes).all()
            or (boxes[:, 2:] <= boxes[:, :2]).any()
        ):
            raise ValueError(f"Expected a list of positive-area XYXY boxes for {frame_id}.")
        result[frame_id] = boxes
    return result


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _result(sample):
    pose = sample.pred_instances
    coords = _array(pose.keypoints)
    scores = _array(pose.keypoint_scores)
    if coords.shape != (1, 17, 2) or scores.shape != (1, 17):
        raise ValueError("The HRNet config must return one pose with 17 COCO-order XY keypoints per crop.")
    coords, scores = coords[0], scores[0]
    valid = np.isfinite(coords).all(axis=1) & np.isfinite(scores)
    coords = np.where(valid[:, None], coords, 0)
    scores = np.where(valid, scores, 0)
    return {"keypoints": coords.tolist(), "keypoint_scores": scores.tolist()}, float(scores.mean())


def _overlap(first, second):
    sizes = np.maximum(0, np.minimum(first[2:], second[2:]) - np.maximum(first[:2], second[:2]))
    intersection = sizes.prod()
    union = (first[2:] - first[:2]).prod() + (second[2:] - second[:2]).prod() - intersection
    return float(intersection / union)


def extract(input_dir, output_dir, config, weights, boxes=None, device="cuda:0"):
    clips = list(_clips(input_dir))
    if not clips:
        raise ValueError(f"No frames found under {input_dir}/<class>/<clip>/.")
    expected = {frame.relative_to(input_dir).as_posix() for frames in clips for frame in frames}
    if boxes is not None and (missing := expected - set(boxes)):
        raise ValueError(f"Boxes JSON has no entry for frame {min(missing)}.")
    expected_json = {
        frame.relative_to(input_dir).with_suffix(".json").as_posix()
        for frames in clips for frame in frames
    }
    existing = {path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*.json")}
    if extras := existing - expected_json:
        raise ValueError(f"Choose a different output directory. It contains unrelated pose file {min(extras)}.")

    try:
        from mmpose.apis import inference_topdown, init_model
    except ImportError as exc:
        raise RuntimeError("Install MMPose 1.x and its compatible MMCV/MMEngine dependencies for HRNet inference.") from exc

    model = init_model(str(config), str(weights), device=device)
    names = model.dataset_meta.get("keypoint_id2name")
    if names and tuple(names.get(index) for index in range(17)) != COCO_KEYPOINTS:
        raise ValueError("Select an HRNet configuration and weights for COCO-order keypoints.")

    for frames in clips:
        previous_box = None
        for frame in frames:
            relative = frame.relative_to(input_dir)
            frame_boxes = boxes[relative.as_posix()] if boxes is not None else None
            instances = []
            # An explicitly empty detection list is a missing observation.
            if frame_boxes is None or len(frame_boxes):
                samples = inference_topdown(model, str(frame), bboxes=frame_boxes, bbox_format="xyxy")
                expected_poses = len(frame_boxes) if frame_boxes is not None else 1
                if len(samples) != expected_poses:
                    raise ValueError(f"HRNet returned {len(samples)} poses for {expected_poses} crops in {frame}.")
                poses, confidence = zip(*(_result(sample) for sample in samples))
                ranking = confidence
                if frame_boxes is not None and previous_box is not None:
                    ranking = tuple(
                        (_overlap(previous_box, box), score)
                        for box, score in zip(frame_boxes, confidence)
                    )
                best = max(range(len(poses)), key=ranking.__getitem__)
                instances = [poses[best]]
                if frame_boxes is not None:
                    previous_box = frame_boxes[best]
            destination = output_dir / relative.with_suffix(".json")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".json.tmp")
            temporary.write_text(json.dumps({"instances": instances}, allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(destination)
    return len(clips), len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="RGB frames in <class>/<clip>/<frame> directories.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for per-frame pose JSON files.")
    parser.add_argument("--config", type=Path, required=True, help="MMPose top-down HRNet COCO model configuration.")
    parser.add_argument("--weights", type=Path, required=True, help="Local HRNet checkpoint for that configuration.")
    parser.add_argument("--boxes", type=Path, help="Optional JSON of per-frame XYXY person detections.")
    parser.add_argument("--device", default="cuda:0", help="MMPose device, for example cuda:0 or cpu.")
    args = parser.parse_args()

    if not args.input.is_dir() or not args.config.is_file() or not args.weights.is_file():
        parser.error("--input must be a directory, and --config and --weights must be existing files.")
    if args.input.resolve() == args.output.resolve():
        parser.error("Use a separate output directory for pose JSON.")
    boxes = _box_manifest(args.boxes) if args.boxes is not None else None
    clips, frames = extract(args.input, args.output, args.config, args.weights, boxes, args.device)
    print(f"Saved HRNet pose records for {frames} frames from {clips} clips to {args.output}.")


if __name__ == "__main__":
    main()
