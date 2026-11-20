"""
Text Encoder Module
Implements the text branch with CLIP encoder and Local Text Encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import clip


class LocalTextEncoder(nn.Module):
    """
    Local Text Encoder for part-level text features
    Refines initial part-level text features under the semantic guidance of global text
    """
    
    def __init__(self, clip_dim: int = 512, hidden_dim: int = 256, num_parts: int = 6):
        """
        Args:
            clip_dim: Dimension of CLIP text embeddings
            hidden_dim: Hidden dimension for the encoder
            num_parts: Number of body parts
        """
        super().__init__()
        
        self.num_parts = num_parts
        self.clip_dim = clip_dim
        
        # Part-level query attention
        self.part_query = nn.Parameter(torch.randn(1, num_parts, hidden_dim))
        
        # Cross-attention for refining part-level features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Output projection to match feature dimension
        self.output_proj = nn.Linear(clip_dim, hidden_dim) if clip_dim != hidden_dim else nn.Identity()
        
    def forward(self, 
                part_features: torch.Tensor, 
                global_feature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            part_features: Initial part-level text features (B, P, D)
            global_feature: Global text feature (B, D)
            
        Returns:
            Refined part-level text features (B, P, D)
        """
        part_tokens = self.output_proj(part_features)
        context = self.norm1(torch.cat((global_feature.unsqueeze(1), part_tokens), dim=1))
        queries = self.part_query + part_tokens
        attended, _ = self.cross_attention(queries, context, context, need_weights=False)
        refined = part_tokens + attended
        return refined + self.ffn(self.norm2(refined))


class TextEncoder(nn.Module):
    """
    Complete Text Encoder combining CLIP and Local Text Encoder
    """
    
    def __init__(self, 
                 clip_model_name: str = "ViT-B/32",
                 freeze_clip: bool = True,
                 feature_dim: int = 512,
                 num_parts: int = 6):
        """
        Args:
            clip_model_name: Name of CLIP model
            freeze_clip: Whether to freeze CLIP parameters
            feature_dim: Output feature dimension
            num_parts: Number of body parts
        """
        super().__init__()
        
        self.num_parts = num_parts
        self.feature_dim = feature_dim
        self.freeze_clip = freeze_clip
        self._text_cache = {}
        self.register_load_state_dict_post_hook(self._reset_text_cache)
        
        # Load CLIP model
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device='cpu')
        self.clip_dim = self.clip_model.text_projection.shape[-1]
        
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.clip_model.eval()
        
        # Local text encoder
        self.local_encoder = LocalTextEncoder(
            clip_dim=self.clip_dim,
            hidden_dim=feature_dim,
            num_parts=num_parts
        )
        
        # Global feature projection
        if self.clip_dim != feature_dim:
            self.global_proj = nn.Linear(self.clip_dim, feature_dim)
        else:
            self.global_proj = nn.Identity()
    
    def _reset_text_cache(self, module, incompatible_keys):
        self._text_cache.clear()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_clip:
            self.clip_model.eval()
        return self

    def _encode_clip_batch(self, prompts: list) -> torch.Tensor:
        context_length = self.clip_model.context_length
        text_rows = [(i, text) for i, text in enumerate(prompts) if isinstance(text, str)]
        part_rows = [(i, parts) for i, parts in enumerate(prompts) if isinstance(parts, tuple)]
        tokens = torch.empty((len(prompts), context_length), dtype=torch.long)
        if text_rows:
            indices, texts = zip(*text_rows)
            tokens[list(indices)] = clip.tokenize(list(texts), context_length=context_length, truncate=True).long()
        if part_rows:
            indices, parts = zip(*part_rows)
            tokens[list(indices)] = clip.tokenize_part_groups(parts, context_length)
        tokens = tokens.to(self.clip_model.text_projection.device)
        features = self.clip_model.encode_text(tokens).to(dtype=self.local_encoder.part_query.dtype)
        return F.normalize(features, dim=-1)

    def _encode_prompts(self, prompts: list) -> torch.Tensor:
        if not self.freeze_clip:
            return self._encode_clip_batch(prompts)

        # Cache fixed CLIP outputs. The local text encoder is trained each step.
        missing = list(dict.fromkeys(prompt for prompt in prompts if prompt not in self._text_cache))
        with torch.no_grad():
            for start in range(0, len(missing), 64):
                batch = missing[start:start + 64]
                self._text_cache.update(zip(batch, self._encode_clip_batch(batch).cpu()))
        return torch.stack([self._text_cache[prompt] for prompt in prompts]).to(self.local_encoder.part_query)

    def encode_text(self, texts: list) -> torch.Tensor:
        """Encode standalone prompts under CLIP's fixed context length."""
        return self._encode_prompts(texts)

    def forward(
        self, part_texts: list, global_text: str = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a global class anchor and separate anchors for each body part."""
        if not part_texts:
            raise ValueError("At least one class with body-part descriptions is required.")
        parts_by_class = [part_texts] if isinstance(part_texts[0], str) else part_texts
        if any(len(parts) != self.num_parts for parts in parts_by_class):
            raise ValueError(f"Each class must supply {self.num_parts} body-part descriptions.")

        if global_text is None:
            global_prompts = [tuple(parts) for parts in parts_by_class]
        else:
            global_prompts = [global_text] * len(parts_by_class)
        local_prompts = [text for parts in parts_by_class for text in parts]
        features = self._encode_prompts(global_prompts + local_prompts)

        num_classes = len(parts_by_class)
        global_features = F.normalize(self.global_proj(features[:num_classes]), dim=-1)
        part_features = features[num_classes:].reshape(num_classes, self.num_parts, self.clip_dim)
        part_features = F.normalize(self.local_encoder(part_features, global_features), dim=-1)
        return global_features, part_features


class TextPromptGenerator:
    """
    Generates template-based text prompts for action classes.
    """
    
    BODY_PARTS = ["head", "torso", "left arm", "right arm", "left leg", "right leg"]
    
    def __init__(self, template: str = None):
        """
        Args:
            template: Prompt template with {body_part} and {action} placeholders
        """
        self.template = template or "Describe the motion of {body_part} when performing the action '{action}'. Focus only on observable motion evidence."
    
    def generate_prompt(self, action: str, body_part: str) -> str:
        """Generate a single prompt"""
        return self.template.format(action=action, body_part=body_part)
    
    def generate_part_prompts(self, action: str) -> list:
        """Generate prompts for all body parts for a given action"""
        return [self.generate_prompt(action, part) for part in self.BODY_PARTS]
    
    def generate_class_prompts(self, actions: list) -> dict:
        """
        Generate prompts for all action classes
        
        Args:
            actions: List of action class names
            
        Returns:
            Dictionary mapping action name to list of part prompts
        """
        return {action: self.generate_part_prompts(action) for action in actions}


# Example usage
if __name__ == "__main__":
    # Test text encoder
    encoder = TextEncoder(clip_model_name="ViT-B/32", feature_dim=512, num_parts=6)
    
    # Test prompt generator
    generator = TextPromptGenerator()
    
    # Example action classes
    actions = ["running", "jumping", "walking"]
    
    # Generate prompts
    prompts = generator.generate_class_prompts(actions)
    
    print("Generated prompts for action 'running':")
    for i, prompt in enumerate(prompts["running"]):
        print(f"  {generator.BODY_PARTS[i]}: {prompt}")
    
    # Test encoding (requires GPU)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # encoder = encoder.to(device)
    # global_feat, part_feat = encoder(prompts["running"])
    # print(f"\nGlobal feature shape: {global_feat.shape}")
    # print(f"Part features shape: {part_feat.shape}")
