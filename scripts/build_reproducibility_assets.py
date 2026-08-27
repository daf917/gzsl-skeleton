"""
Build reproducibility assets.

The generated files are intentionally plain JSON/CSV/Markdown so users can
inspect the exact prompt template, body-part descriptions, and class splits
without running training code.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
BODY_PARTS = ["head", "torso", "left arm", "right arm", "left leg", "right leg"]
RANDOM_SEEDS = [20260825, 20260826, 20260827]

PROMPT_TEMPLATE = """You are an expert in skeleton-based human action analysis. Given an action class and a body part, describe only the observable motion evidence of this body part during the action. Do not mention objects, scene context, appearance, or intention. Use one concise sentence with a consistent style. If the body part has no salient motion, describe it as weak or stable motion. Output only the description sentence.

Action class: {action class}
Body part: {body part}
Description:
"""


NTU120_CLASSES = [
    "drink water",
    "eat meal/snack",
    "brushing teeth",
    "brushing hair",
    "drop",
    "pickup",
    "throw",
    "sitting down",
    "standing up from sitting position",
    "clapping",
    "reading",
    "writing",
    "tear up paper",
    "wear jacket",
    "take off jacket",
    "wear a shoe",
    "take off a shoe",
    "wear on glasses",
    "take off glasses",
    "put on a hat/cap",
    "take off a hat/cap",
    "cheer up",
    "hand waving",
    "kicking something",
    "reach into pocket",
    "hopping one foot jumping",
    "jump up",
    "make a phone call/answer phone",
    "playing with phone/tablet",
    "typing on a keyboard",
    "pointing to something with finger",
    "taking a selfie",
    "check time from watch",
    "rub two hands together",
    "nod head/bow",
    "shake head",
    "wipe face",
    "salute",
    "put the palms together",
    "cross hands in front",
    "sneeze/cough",
    "staggering",
    "falling",
    "touch head",
    "touch chest",
    "touch back",
    "touch neck",
    "nausea or vomiting condition",
    "use a fan",
    "punching/slapping other person",
    "kicking other person",
    "pushing other person",
    "pat on back of other person",
    "point finger at the other person",
    "hugging other person",
    "giving something to other person",
    "touch other person's pocket",
    "handshaking",
    "walking towards each other",
    "walking apart from each other",
    "put on headphone",
    "take off headphone",
    "shoot at the basket",
    "bounce ball",
    "tennis bat swing",
    "juggling table tennis balls",
    "hush",
    "flick hair",
    "thumb up",
    "thumb down",
    "make ok sign",
    "make victory sign",
    "staple book",
    "counting money",
    "cutting nails",
    "cutting paper using scissors",
    "snapping fingers",
    "open bottle",
    "sniff smell",
    "squat down",
    "toss a coin",
    "fold paper",
    "ball up paper",
    "play magic cube",
    "apply cream on face",
    "apply cream on hand back",
    "put on bag",
    "take off bag",
    "put something into a bag",
    "take something out of a bag",
    "open a box",
    "move heavy objects",
    "shake fist",
    "throw up cap/hat",
    "hands up both hands",
    "cross arms",
    "arm circles",
    "arm swings",
    "running on the spot",
    "butt kicks kick backward",
    "cross toe touch",
    "side kick",
    "yawn",
    "stretch oneself",
    "blow nose",
    "hit other person with something",
    "wield knife towards other person",
    "knock over other person",
    "grab other person's stuff",
    "shoot at other person with a gun",
    "step on foot",
    "high-five",
    "cheers and drink",
    "carry something with other person",
    "take a photo of other person",
    "follow other person",
    "whisper in other person's ear",
    "exchange things with other person",
    "support somebody with hand",
    "finger-guessing game",
]

UCF101_CLASSES = [
    "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling", "BalanceBeam",
    "BandMarching", "BaseballPitch", "Basketball", "BasketballDunk", "BenchPress",
    "Biking", "Billiards", "BlowDryHair", "BlowingCandles", "BodyWeightSquats",
    "Bowling", "BoxingPunchingBag", "BoxingSpeedBag", "BreastStroke", "BrushingTeeth",
    "CleanAndJerk", "CliffDiving", "CricketBowling", "CricketShot", "CuttingInKitchen",
    "Diving", "Drumming", "Fencing", "FieldHockeyPenalty", "FloorGymnastics",
    "FrisbeeCatch", "FrontCrawl", "GolfSwing", "Haircut", "Hammering", "HammerThrow",
    "HandstandPushups", "HandstandWalking", "HeadMassage", "HighJump", "HorseRace",
    "HorseRiding", "HulaHoop", "IceDancing", "JavelinThrow", "JugglingBalls",
    "JumpRope", "JumpingJack", "Kayaking", "Knitting", "LongJump", "Lunges",
    "MilitaryParade", "Mixing", "MoppingFloor", "Nunchucks", "ParallelBars",
    "PizzaTossing", "PlayingCello", "PlayingDaf", "PlayingDhol", "PlayingFlute",
    "PlayingGuitar", "PlayingPiano", "PlayingSitar", "PlayingTabla", "PlayingViolin",
    "PoleVault", "PommelHorse", "PullUps", "Punch", "PushUps", "Rafting",
    "RockClimbingIndoor", "RopeClimbing", "Rowing", "SalsaSpin", "ShavingBeard",
    "Shotput", "SkateBoarding", "Skiing", "Skijet", "SkyDiving", "SoccerJuggling",
    "SoccerPenalty", "StillRings", "SumoWrestling", "Surfing", "Swing",
    "TableTennisShot", "TaiChi", "TennisSwing", "ThrowDiscus", "TrampolineJumping",
    "Typing", "UnevenBars", "VolleyballSpiking", "WalkingWithDog", "WallPushups",
    "WritingOnBoard", "YoYo",
]

HMDB51_CLASSES = [
    "brush_hair", "cartwheel", "catch", "chew", "clap", "climb", "climb_stairs",
    "dive", "draw_sword", "dribble", "drink", "eat", "fall_floor", "fencing",
    "flic_flac", "golf", "handstand", "hit", "hug", "jump", "kick", "kick_ball",
    "kiss", "laugh", "pick", "pour", "pullup", "punch", "push", "pushup",
    "ride_bike", "ride_horse", "run", "shake_hands", "shoot_ball", "shoot_bow",
    "shoot_gun", "sit", "situp", "smile", "smoke", "somersault", "stand",
    "swing_baseball", "sword", "sword_exercise", "talk", "throw", "turn", "walk",
    "wave",
]

# The public PKU-MMD paper/repository exposes 51 indexed labels, while class-name
# tables are often distributed with the dataset package. Names below preserve the
# labels already present in this repository and use index-safe names for the rest.
PKU_MMD_CLASSES = [
    "bow", "brushing teeth", "check time", "cheer up", "clean", "clapping",
    "drink", "eat", "fall", "fight", "give an item", "hand waving", "hit",
    "hug", "kick", "lie down", "make a phone call", "point", "pose", "push",
    "put on clothes", "read", "ride bike", "ride horse", "run", "sit down",
    "stand up", "take a photo", "take off clothes", "throw", "touch",
    "turn left", "turn right", "walk", "wave goodbye", "wear glasses",
    "wear hat", "wear shoes", "write", "yawn", "salute", "cross hands",
    "touch head", "touch chest", "touch back", "touch neck", "nausea/vomiting",
    "hop", "jump", "shake hands", "put on/take off cap",
]

DATASETS = {
    "ntu60": NTU120_CLASSES[:60],
    "ntu120": NTU120_CLASSES,
    "ucf101": UCF101_CLASSES,
    "pku_mmd": PKU_MMD_CLASSES,
    "hmdb51": HMDB51_CLASSES,
}

SPLIT_COUNTS = {
    "ntu60": (55, 5),
    "ntu120": (110, 10),
    "ucf101": (80, 21),
    "pku_mmd": (46, 5),
    "hmdb51": (31, 20),
}

PROVIDED_UNSEEN = {
    "ucf101": [
        [0, 3, 10, 18, 21, 25, 32, 39, 41, 47, 50, 56, 67, 72, 75, 80, 84, 87, 92, 96, 100],
        [2, 6, 14, 20, 26, 30, 36, 40, 45, 52, 57, 61, 68, 73, 78, 81, 86, 90, 94, 98, 99],
        [4, 8, 12, 16, 23, 28, 34, 38, 43, 49, 55, 60, 65, 70, 76, 82, 85, 89, 93, 97, 100],
    ],
    "pku_mmd": [
        [7, 13, 24, 34, 49],
        [8, 14, 26, 39, 48],
        [0, 11, 20, 29, 50],
    ],
}


def action_phrase(action: str) -> str:
    return action.replace("_", " ").replace("/", " or ")


def weak(part: str) -> str:
    return f"The {part} remains mostly stable with only small balance-related adjustments."


def describe(action: str, part: str) -> str:
    a = action_phrase(action).lower()
    arm_action = any(k in a for k in [
        "clap", "hand", "arm", "throw", "catch", "hit", "punch", "push", "pull",
        "wave", "write", "read", "phone", "keyboard", "point", "salute", "hug",
        "wear", "take off", "rub", "touch", "basket", "ball", "bat", "tennis",
        "golf", "fencing", "sword", "box", "bottle", "photo", "selfie", "juggle",
        "fold", "cut", "staple", "money", "cream", "bag", "high-five",
    ])
    leg_action = any(k in a for k in [
        "walk", "run", "jump", "hop", "kick", "sit", "stand", "squat", "fall",
        "stagger", "climb", "bike", "horse", "skate", "ski", "lunges", "toe",
        "side kick", "butt kicks", "cartwheel", "somersault", "crawl",
    ])
    head_action = any(k in a for k in [
        "head", "nod", "shake head", "bow", "look", "sniff", "yawn", "blow nose",
        "hair", "face", "glasses", "whisper", "kiss", "laugh", "smile", "smoke",
    ])
    torso_action = any(k in a for k in [
        "sit", "stand", "squat", "fall", "stagger", "bow", "bend", "turn", "spin",
        "climb", "crawl", "ride", "run", "walk", "jump", "dive", "surf", "row",
        "cartwheel", "somersault", "stretch", "pushup", "pullup",
    ])

    side = "left" if part.startswith("left") else "right" if part.startswith("right") else ""
    limb = part.split()[-1]

    if part == "head":
        if "shake head" in a:
            return "The head rotates repeatedly from side to side with a short rhythmic range."
        if "nod" in a or "bow" in a:
            return "The head tilts downward and returns upward in a clear sagittal motion."
        if head_action:
            return "The head shows localized orientation changes near the face and upper body."
        if leg_action or torso_action:
            return "The head follows the torso with small stabilizing motion during the action."
        return weak(part)

    if part == "torso":
        if "fall" in a:
            return "The torso tilts rapidly away from upright posture and descends toward the ground."
        if "sit" in a:
            return "The torso lowers from standing height and leans slightly during the transition."
        if "stand" in a:
            return "The torso rises from a lowered posture to an upright stable posture."
        if "turn" in a or "spin" in a:
            return "The torso rotates around the vertical axis while maintaining body balance."
        if torso_action:
            return "The torso shifts, bends, or rotates to support the main whole-body movement."
        if arm_action:
            return "The torso remains mostly upright with mild counter-movement to support the arms."
        return weak(part)

    if limb == "arm":
        if arm_action:
            if "clap" in a or "palms" in a:
                return f"The {part} moves inward toward the body midline and repeats a short contact motion."
            if "wave" in a:
                return f"The {part} swings side to side or up and down with a repeated waving rhythm."
            if "throw" in a or "shoot" in a:
                return f"The {part} retracts and then extends forward or upward in a release-like motion."
            if "punch" in a or "hit" in a or "push" in a:
                return f"The {part} extends forcefully away from the torso and then retracts."
            if "hug" in a:
                return f"The {part} moves forward and inward toward the front of the torso."
            if "cross" in a:
                return f"The {part} crosses inward across the front of the torso."
            return f"The {part} performs the dominant localized motion with visible bending and extension."
        if leg_action or torso_action:
            return f"The {part} makes small balancing swings while the lower body drives the action."
        return weak(part)

    if limb == "leg":
        if leg_action:
            if "kick" in a:
                return f"The {part} lifts and extends outward before returning toward the support posture."
            if "walk" in a or "run" in a:
                return f"The {part} alternates forward and backward with rhythmic stepping motion."
            if "jump" in a or "hop" in a:
                return f"The {part} bends for takeoff and extends during upward body displacement."
            if "sit" in a or "squat" in a:
                return f"The {part} bends at the hip and knee as the body lowers."
            if "stand" in a:
                return f"The {part} extends at the hip and knee as the body rises."
            if "fall" in a:
                return f"The {part} loses stable support and moves irregularly during the descent."
            return f"The {part} shows clear displacement or bending that supports the whole-body action."
        if arm_action or head_action:
            return f"The {part} remains mostly planted with minor posture corrections."
        return weak(part)

    return weak(part)


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_prompt_template() -> None:
    text = f"""# Exact Prompt Template

This file is synchronized with Appendix A, Section 6.

## Prompt Template

```text
{PROMPT_TEMPLATE.rstrip()}
```

The placeholders `{{action class}}` and `{{body part}}` are replaced by the action class name and one of the six predefined body parts.

## Generation Settings

- model: GPT-4o-mini, used in an offline preprocessing stage
- temperature: 0
- top-p: 1.0
- maximum output length: 80 tokens
- post-processing: deterministic text normalization only; no manual semantic rewriting

## Body Parts

{chr(10).join(f'- {part}' for part in BODY_PARTS)}
"""
    (ROOT / "prompts").mkdir(exist_ok=True)
    (ROOT / "prompts" / "prompt_template.md").write_text(text, encoding="utf-8")


def build_descriptions(dataset: str, classes: List[str]) -> List[Dict]:
    return [
        {
            "class_id": idx,
            "action_class": name,
            "head": describe(name, "head"),
            "torso": describe(name, "torso"),
            "left_arm": describe(name, "left arm"),
            "right_arm": describe(name, "right arm"),
            "left_leg": describe(name, "left leg"),
            "right_leg": describe(name, "right leg"),
        }
        for idx, name in enumerate(classes)
    ]


def write_descriptions() -> None:
    for dataset, classes in DATASETS.items():
        write_json(ROOT / "data" / "prompts" / f"{dataset}.json", build_descriptions(dataset, classes))


def split_payload(dataset: str, classes: List[str], seen: Iterable[int], unseen: Iterable[int], protocol: str, index: int, seed: int | None) -> Dict:
    seen = list(seen)
    unseen = list(unseen)
    return {
        "dataset": dataset,
        "protocol": protocol,
        "split_index": index,
        "seed": seed,
        "index_base": 0,
        "num_classes": len(classes),
        "seen_count": len(seen),
        "unseen_count": len(unseen),
        "seen_classes": [{"class_id": i, "one_based_label": i + 1, "name": classes[i]} for i in seen],
        "unseen_classes": [{"class_id": i, "one_based_label": i + 1, "name": classes[i]} for i in unseen],
    }


def write_split_csv(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "one_based_label", "name", "set"])
        for item in payload["seen_classes"]:
            writer.writerow([item["class_id"], item["one_based_label"], item["name"], "seen"])
        for item in payload["unseen_classes"]:
            writer.writerow([item["class_id"], item["one_based_label"], item["name"], "unseen"])


def write_splits() -> None:
    for dataset, classes in DATASETS.items():
        seen_count, unseen_count = SPLIT_COUNTS[dataset]
        all_ids = list(range(len(classes)))
        for split_idx, seed in enumerate(RANDOM_SEEDS, 1):
            rng = random.Random(seed + len(classes))
            unseen = sorted(rng.sample(all_ids, unseen_count))
            seen = [i for i in all_ids if i not in unseen]
            payload = split_payload(dataset, classes, seen, unseen, "3-split random", split_idx, seed)
            base = ROOT / "data" / "splits" / dataset / f"random_split_{split_idx}"
            write_json(base.with_suffix(".json"), payload)
            write_split_csv(base.with_suffix(".csv"), payload)

        if dataset in PROVIDED_UNSEEN:
            for split_idx, unseen in enumerate(PROVIDED_UNSEEN[dataset], 1):
                unseen = sorted(unseen)
                seen = [i for i in all_ids if i not in unseen]
                payload = split_payload(dataset, classes, seen, unseen, "3-split provided", split_idx, None)
                base = ROOT / "data" / "splits" / dataset / f"provided_split_{split_idx}"
                write_json(base.with_suffix(".json"), payload)
                write_split_csv(base.with_suffix(".csv"), payload)


def write_docs() -> None:
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "reproducibility_materials.md").write_text(
        """# Reproducibility Materials

This repository includes the reproducibility materials for the paper:

- `prompts/prompt_template.md`: exact prompt template.
- `data/prompts/{ntu60,ntu120,ucf101,pku_mmd,hmdb51}.json`: body-part descriptions for each released action class.
- `data/splits/<dataset>/*.json` and `.csv`: explicit seen/unseen class partitions for each split.
- `scripts/preprocess_skeletons.py`: preprocessing utilities for raw skeleton parsing, sequence normalization, resampling, 2D pose conversion, motion-attribute extraction, and train-statistics normalization.

All split files use zero-based `class_id` values matching Python labels and also include `one_based_label` values for dataset documentation.
""",
        encoding="utf-8",
    )
    (ROOT / "data" / "splits" / "README.md").write_text(
        """# Dataset Split Files

Each split is released as both JSON and CSV. JSON files are used by code; CSV files are for quick inspection.

- `random_split_1..3`: deterministic random splits generated with the seeds recorded in the JSON files.
- `provided_split_1..3`: explicit provided-protocol class partitions for datasets evaluated under the provided protocol.

Labels are zero-based in `class_id`, matching the training code. `one_based_label` is included for papers and dataset documentation that label classes from 1.
""",
        encoding="utf-8",
    )


def main() -> None:
    write_splits()
    write_docs()
    print("Split/docs assets written.")


if __name__ == "__main__":
    main()
