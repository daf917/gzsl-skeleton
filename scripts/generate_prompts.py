"""
LLM Prompt Generation Script
Generates part-level textual descriptions for action classes using LLM

This script implements the prompt construction described in Section 3.2 of the paper:
"Generalized Zero-Shot Skeleton Action Recognition with Compositional Motion-Attribute Primitives"
"""

import argparse
import json
import os
import pickle
from typing import Dict, List, Optional
from tqdm import tqdm

# Try to import openai, fall back to offline mode
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# Body part definitions following human topology priors
BODY_PARTS = [
    "head",
    "torso", 
    "left arm", 
    "right arm", 
    "left leg", 
    "right leg"
]

# Prompt template as described in the paper
PROMPT_TEMPLATE = """Describe the observable motion evidence of the {body_part} when performing the action '{action}'. 

Requirements:
1. Focus ONLY on the motion characteristics of the specified body part
2. Use concise, standardized language
3. Describe observable evidence (position, direction, movement pattern)
4. If the body part has no salient motion, use a weak-motion description
5. Do not include information about other body parts

Example format: "The [body_part] [motion_description]"
"""

# Action class names for different datasets
DATASET_CLASSES = {
    'ntu60': [
        "drink water", "eat meal", "brushing teeth", "brushing hair", "drop",
        "pickup", "throw", "sitting down", "standing up", "clapping",
        "reading", "writing", "tear up paper", "wear jacket", "take off jacket",
        "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses",
        "put on bag", "take off bag", "put on / take off hat", "cheer up",
        "hand waving", "kicking something", "put something somewhere", "grab someone's part",
        "stick out hand", "touch back", "squat", "jump up", "make arm straight",
        "cross arms in front", "put arms behind", "put arms behind head",
        "go sideways", "turn around", "look at person", "shake fist", "hit someone",
        "kick ball", "swing baseball bat", "play with phone", "play with tablet",
        "point at person", "point at something", "checking watch", "rub hands",
        "nod head", "shake head", "touch head", "touch face", "wipe face",
        "salute", "shake hands", "hug", "touch someone's shoulder"
    ],
    'ntu120': [
        # 110 seen classes + 10 unseen classes
        # This is a subset, actual implementation needs full class list
        "drink water", "eat meal", "brushing teeth", "brushing hair", "drop",
        "pickup", "throw", "sitting down", "standing up", "clapping",
        "reading", "writing", "tear up paper", "wear jacket", "take off jacket",
        "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses",
        "put on bag", "take off bag", "put on / take off hat", "cheer up",
        "hand waving", "kicking something", "put something somewhere", "grab someone's part",
        "stick out hand", "touch back", "squat", "jump up", "make arm straight",
        "cross arms in front", "put arms behind", "put arms behind head",
        "go sideways", "turn around", "look at person", "shake fist", "hit someone",
        "kick ball", "swing baseball bat", "play with phone", "play with tablet",
        "point at person", "point at something", "checking watch", "rub hands",
        "nod head", "shake head", "touch head", "touch face", "wipe face",
        "salute", "shake hands", "hug", "touch someone's shoulder"
    ],
    'pku_mmd': [
        "bow", "brushing teeth", "check time", "cheer up", "clean",
        "clapping", "drink", "eat", "fall", "fight",
        "give an item", "hand waving", "hit", "hug", "kick",
        "lie down", "make a phone call", "point", "pose", "push",
        "put on clothes", "read", "ride bike", "ride horse", "run",
        "sit down", "stand up", "take a photo", "take off clothes", "throw",
        "touch", "turn left", "turn right", "walk", "wave goodbye",
        "wear glasses", "wear hat", "wear shoes", "write", "yawn"
    ],
    'ucf101': [
        "Basketball", "BasketballDunk", "Biking", "CliffDiving", "CricketBowling",
        "CricketShot", "Diving", "Fencing", "FloorGymnastics", "GolfSwing",
        "HorseRiding", "IceDancing", "JumpingJack", "Lunges", "MilitaryParade",
        "PullUps", "Punch", "PushUps", "RockClimbingIndoor", "RopeClimbing",
        "Rowing", "SalsaSpin", "SkateBoarding", "Skiing", "Skijet",
        "SoccerJuggling", "Swing", "TaiChi", "TennisSwing", "ThrowDisc",
        "VolleyballSpiking", "Walking", "YoYo"
    ],
    'hmdb51': [
        "brush_hair", "catch", "clap", "climb", "climb_stairs",
        "dance", "drink", "drive", "eat", "fall_flat",
        "fling", "golf", "hit", "hug", "jump",
        "kick", "kiss", "laugh", "pour", "pullup",
        "punch", "push", "ride_bike", "ride_horse", "run",
        "shoot_ball", "shoot_bow", "shoot_gun", "sit", "situp",
        "smile", "smoke", "somersault", "stand", "straw_hat",
        "swing", "talk", "throw", "turn", "walk",
        "wave"
    ]
}


class PromptGenerator:
    """
    Generates part-level textual descriptions for action classes using LLM
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-3.5-turbo",
                 template: str = PROMPT_TEMPLATE):
        """
        Args:
            api_key: OpenAI API key
            model: LLM model to use
            template: Prompt template
        """
        self.template = template
        self.model = model
        
        if OPENAI_AVAILABLE and api_key:
            openai.api_key = api_key
            self.client = openai.OpenAI(api_key=api_key)
        else:
            self.client = None
            print("Warning: OpenAI not available. Using offline mode.")
    
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
    
    def call_llm(self, prompt: str) -> str:
        """
        Call LLM to generate description
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated description
        """
        if self.client is None:
            # Offline mode: return placeholder
            return f"The {BODY_PARTS[0]} performs a motion."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in describing human motion for action recognition."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Deterministic output
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return f"The {BODY_PARTS[0]} performs a motion."
    
    def generate_descriptions(self, 
                            action: str, 
                            use_llm: bool = True) -> Dict[str, str]:
        """
        Generate part-level descriptions for an action
        
        Args:
            action: Action class name
            use_llm: Whether to use LLM or return templates
            
        Returns:
            Dictionary mapping body part to description
        """
        descriptions = {}
        
        for body_part in BODY_PARTS:
            if use_llm and self.client:
                prompt = self.generate_single_prompt(action, body_part)
                description = self.call_llm(prompt)
            else:
                # Fallback: template-based description
                description = self._template_description(action, body_part)
            
            descriptions[body_part] = description
        
        return descriptions
    
    def _template_description(self, action: str, body_part: str) -> str:
        """
        Generate template-based description (fallback)
        
        Args:
            action: Action name
            body_part: Body part name
            
        Returns:
            Description string
        """
        # Simple template-based descriptions
        templates = {
            "head": f"The head moves during {action}",
            "torso": f"The torso is involved in {action}",
            "left arm": f"The left arm performs motions for {action}",
            "right arm": f"The right arm performs motions for {action}",
            "left leg": f"The left leg is used in {action}",
            "right leg": f"The right leg is used in {action}"
        }
        return templates.get(body_part, f"The {body_part} moves during {action}")
    
    def generate_for_dataset(self, 
                           dataset: str,
                           use_llm: bool = True,
                           output_path: Optional[str] = None) -> Dict[str, Dict[str, str]]:
        """
        Generate descriptions for all classes in a dataset
        
        Args:
            dataset: Dataset name
            use_llm: Whether to use LLM
            output_path: Path to save results
            
        Returns:
            Dictionary mapping action to body part descriptions
        """
        classes = DATASET_CLASSES.get(dataset, [])
        
        print(f"Generating prompts for {len(classes)} classes in {dataset}...")
        
        results = {}
        for action in tqdm(classes, desc=f"Generating {dataset} prompts"):
            results[action] = self.generate_descriptions(action, use_llm=use_llm)
        
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
        if self.client and OPENAI_AVAILABLE:
            prompt = f"Provide a concise description of the action '{action}' focusing on overall body movement."
            return self.call_llm(prompt)
        else:
            return f"A person is {action}."


def generate_prompts_offline(dataset: str, output_path: str):
    """
    Generate prompts without LLM (using templates)
    
    Args:
        dataset: Dataset name
        output_path: Output file path
    """
    generator = PromptGenerator()
    results = generator.generate_for_dataset(dataset, use_llm=False)
    
    # Also add global descriptions
    global_descriptions = {}
    for action in DATASET_CLASSES.get(dataset, []):
        global_descriptions[action] = generator.generate_global_description(action)
    
    # Combine
    final_results = {
        'part_descriptions': results,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate text prompts for GZSL')
    parser.add_argument('--dataset', type=str, default='ntu60',
                       choices=['ntu60', 'ntu120', 'pku_mmd', 'ucf101', 'hmdb51'],
                       help='Dataset name')
    parser.add_argument('--output', type=str, default='data/prompts/',
                       help='Output directory')
    parser.add_argument('--api-key', type=str, default=None,
                       help='OpenAI API key')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo',
                       help='LLM model name')
    parser.add_argument('--offline', action='store_true',
                       help='Use offline mode (no LLM)')
    
    args = parser.parse_args()
    
    output_path = os.path.join(args.output, f"{args.dataset}_prompts.json")
    
    if args.offline or not OPENAI_AVAILABLE:
        print("Using offline mode...")
        generate_prompts_offline(args.dataset, output_path)
    else:
        print(f"Using LLM: {args.model}")
        generator = PromptGenerator(api_key=args.api_key, model=args.model)
        results = generator.generate_for_dataset(args.dataset, use_llm=True, output_path=output_path)
        print(f"Generated {len(results)} prompts for {args.dataset}")
