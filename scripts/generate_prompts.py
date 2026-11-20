"""
Text prompt utility script.
Builds template prompts and CLIP text features from released descriptions.

This script implements the prompt construction described in Section 3.2 of the paper:
"Generalized Zero-Shot Skeleton Action Recognition with Compositional Motion-Attribute Primitives"
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

# Body part definitions following human topology priors
BODY_PARTS = [
    "head",
    "torso",
    "left arm",
    "right arm",
    "left leg",
    "right leg"
]

# Prompt template released with the reproducibility materials.
SYSTEM_PROMPT = (
    "You are an expert in skeleton-based human action analysis."
)

PROMPT_TEMPLATE = """You are an expert in skeleton-based human action analysis. Given an action class and a body part, describe only the observable motion evidence of this body part during the action. Do not mention objects, scene context, appearance, or intention. Use one concise sentence with a consistent style. If the body part has no salient motion, describe it as weak or stable motion. Output only the description sentence.

Action class: {action}
Body part: {body_part}
Description:
"""

BODY_PART_KEYS = {
    "head": "head",
    "torso": "torso",
    "left arm": "left_arm",
    "right arm": "right_arm",
    "left leg": "left_leg",
    "right leg": "right_leg",
}


def load_released_descriptions(dataset: str) -> Dict[str, Dict[str, str]]:
    """Load body-part descriptions from the released JSON file."""
    prompt_path = Path(__file__).resolve().parents[1] / "data" / "prompts" / f"{dataset}.json"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Released description file not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return {
        row["action_class"]: {part: row[key] for part, key in BODY_PART_KEYS.items()}
        for row in rows
    }


def load_released_action_classes(dataset: str) -> List[str]:
    """Load action classes from the released description JSON file."""
    return list(load_released_descriptions(dataset).keys())


class PromptGenerator:
    """
    Builds prompt strings for action classes.
    """

    def __init__(self,
                 template: str = PROMPT_TEMPLATE):
        """
        Args:
            template: Prompt template
        """
        self.template = template

    def generate_single_prompt(self, action: str, body_part: str) -> str:
        """Generate a single prompt for action + body part"""
        return self.template.format(action=action, body_part=body_part)

    def generate_part_prompts(self, action: str) -> List[str]:
        """Generate prompts for all body parts for a given action"""
        prompts = []
        for body_part in BODY_PARTS:
            prompt = self.generate_single_prompt(action, body_part)
            prompts.append(prompt)
        return prompts

    def generate_for_dataset(self,
                           dataset: str,
                           output_path: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Generate prompt strings for all classes in a dataset.

        Args:
            dataset: Dataset name
            output_path: Path to save results

        Returns:
            Dictionary mapping action to prompt strings
        """
        classes = load_released_action_classes(dataset)

        print(f"Building prompt strings for {len(classes)} classes in {dataset}...")

        results = {
            action: self.generate_part_prompts(action)
            for action in tqdm(classes, desc=f"Building {dataset} prompts")
        }

        # Save results
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Saved prompts to {output_path}")

        return results

    def generate_global_description(self, action: str) -> str:
        """
        Generate a global description for an action (for all parts combined)

        Args:
            action: Action name

        Returns:
            Global description
        """
        return f"A person is {action}."


def export_released_prompt_inputs(dataset: str, output_path: str):
    """
    Export released descriptions with matching prompt inputs.

    Args:
        dataset: Dataset name
        output_path: Output file path
    """
    generator = PromptGenerator()
    descriptions = load_released_descriptions(dataset)
    prompt_inputs = generator.generate_for_dataset(dataset)

    global_descriptions = {}
    for action in descriptions:
        global_descriptions[action] = generator.generate_global_description(action)

    final_results = {
        'prompt_inputs': prompt_inputs,
        'part_descriptions': descriptions,
        'global_descriptions': global_descriptions
    }

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"Saved prompts to {output_path}")
    return final_results


def load_prompts(prompt_path: str) -> Dict:
    """
    Load pre-generated prompts

    Args:
        prompt_path: Path to prompts file

    Returns:
        Dictionary with prompts
    """
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    return prompts


def create_text_features_from_prompts(prompts: Dict, text_encoder, device: str = 'cuda'):
    """
    Create text features from prompts using CLIP

    Args:
        prompts: Dictionary with prompts
        text_encoder: TextEncoder instance

    Returns:
        Dictionary with text features
    """
    import torch

    part_descriptions = prompts.get('part_descriptions', prompts)
    global_descriptions = prompts.get('global_descriptions', {})

    text_features_global = []
    text_features_part = []
    class_names = []

    for action, part_desc in tqdm(part_descriptions.items(), desc="Encoding prompts"):
        class_names.append(action)

        # Global text
        global_text = global_descriptions.get(action, f"{action} action")

        # Part texts
        part_texts = [part_desc.get(bp, f"{action} motion") for bp in BODY_PARTS]

        # Encode (assuming single class at a time for simplicity)
        with torch.no_grad():
            global_feat, part_feat = text_encoder([part_texts], global_text)

        text_features_global.append(global_feat.cpu())
        text_features_part.append(part_feat.cpu())

    # Concatenate
    text_features_global = torch.cat(text_features_global, dim=0)
    text_features_part = torch.cat(text_features_part, dim=0)

    return {
        'global': text_features_global,
        'part': text_features_part,
        'class_names': class_names
    }


def generate_descriptions(dataset, output_path, classes_path=None, *, client=None):
    """Generate class descriptions offline with the manuscript's decoding settings."""
    if classes_path is None:
        source = Path(__file__).resolve().parents[1] / 'data' / 'prompts' / f'{dataset}.json'
    else:
        source = Path(classes_path)
    rows = json.loads(source.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('The class file must contain a nonempty JSON array.')
    classes = []
    for index, row in enumerate(rows):
        if isinstance(row, str):
            class_id, action = index, row
        else:
            class_id, action = row['class_id'], row['action_class']
        if type(class_id) is not int or class_id < 0 or not isinstance(action, str) or not action.strip():
            raise ValueError('Each class needs a nonnegative integer ID and a nonempty action name.')
        classes.append((class_id, action))
    if len({class_id for class_id, _ in classes}) != len(classes):
        raise ValueError('Class IDs must be unique.')
    if client is None:
        from openai import OpenAI
        client = OpenAI()
    generator = PromptGenerator()
    descriptions = []
    for class_id, action in tqdm(classes, desc=f'Generating {dataset} descriptions'):
        entry = {'class_id': class_id, 'action_class': action}
        for part, key in BODY_PART_KEYS.items():
            result = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{'role': 'user', 'content': generator.generate_single_prompt(action, part)}],
                temperature=0,
                top_p=1.0,
                max_tokens=80,
            )
            content = result.choices[0].message.content or ''
            normalized = ' '.join(content.split())
            if not normalized:
                raise ValueError(f'Empty description for class {class_id}, {part}.')
            entry[key] = normalized
        descriptions.append(entry)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(descriptions, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export or generate class body-part descriptions')
    parser.add_argument('--dataset', default='ntu60')
    parser.add_argument('--output', default='data/prompts/', help='Output directory for prompt-input export')
    parser.add_argument('--generate', type=Path, metavar='DESCRIPTIONS_JSON',
                        help='Generate fresh descriptions to a specified JSON file using the OpenAI API')
    parser.add_argument('--classes', type=Path, help='Optional JSON array of class names or labelled records')
    args = parser.parse_args()
    if args.generate is not None:
        if args.generate.exists():
            parser.error('--generate must name a new file')
        generate_descriptions(args.dataset, args.generate, args.classes)
    elif args.classes is not None:
        parser.error('--classes requires --generate')
    else:
        output_path = os.path.join(args.output, f'{args.dataset}_prompt_inputs.json')
        export_released_prompt_inputs(args.dataset, output_path)
