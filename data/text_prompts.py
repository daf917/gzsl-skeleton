"""Load the released class and body-part descriptions by dataset label."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


PART_KEYS = ("head", "torso", "left_arm", "right_arm", "left_leg", "right_leg")


def load_class_prompts(
    dataset_name: str,
    class_ids: Sequence[int],
    descriptions_dir: Optional[str] = None,
) -> Tuple[Dict[int, str], Dict[int, List[str]]]:
    """Return class names and body-part descriptions in the model's part order."""
    root = Path(descriptions_dir) if descriptions_dir else Path(__file__).resolve().parent / "prompts"
    path = root / f"{dataset_name}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a list of prompt records.")

    requested_ids = set(class_ids)
    names = {}
    descriptions = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{path} contains an invalid prompt record.")
        class_id = row["class_id"]
        if type(class_id) is not int:
            raise ValueError(f"{path} contains a non-integer class_id.")
        if class_id not in requested_ids:
            continue
        if class_id in names:
            raise ValueError(f"Duplicate prompt record for class {class_id} in {path}.")
        name = row["action_class"]
        parts = [row[key] for key in PART_KEYS]
        if any(not isinstance(value, str) or not value.strip() for value in [name, *parts]):
            raise ValueError(f"Incomplete prompt record for class {class_id} in {path}.")
        names[class_id] = name
        descriptions[class_id] = parts

    missing = requested_ids.difference(names)
    if missing:
        raise ValueError(f"{path} lacks prompts for class IDs {sorted(missing)}.")
    return names, descriptions
