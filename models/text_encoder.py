"""
Text Encoder Module
Implements the text branch with CLIP encoder and Local Text Encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
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
        B, P, D = part_features.shape
        
        # Project to hidden dimension if needed
        if isinstance(self.output_proj, nn.Identity):
            hidden_features = part_features
        else:
            hidden_features = self.output_proj(part_features)
        
        # Create queries from learnable part queries
        queries = self.part_query.expand(B, -1, -1)  # (B, P, hidden_dim)
        
        # Cross-attention with part features as keys/values
        refined, _ = self.cross_attention(
            query=queries,
            key=hidden_features,
            value=hidden_features
        )
        
        # Residual connection and FFN
        hidden_features = self.norm1(hidden_features + refined)
        hidden_features = hidden_features + self.ffn(hidden_features)
        hidden_features = self.norm2(hidden_features)
        
        return hidden_features


class TextEncoder(nn.Module):
    """
    Complete Text Encoder combining CLIP and Local Text Encoder
    """
    
    def __init__(self, 
                 clip_model_name: str = "ViT-B/32",
                 freeze_clip: bool = True,
                 feature_dim: int = 256,
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
        
        # Load CLIP model
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name)
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
    
    @torch.no_grad()
    def encode_text(self, texts: list) -> torch.Tensor:
        """
        Encode text prompts using CLIP
        
        Args:
            texts: List of text strings
            
        Returns:
            CLIP text embeddings (N, D)
        """
        # Tokenize
        text_tokens = clip.tokenize(texts).to(next(self.parameters()).device)
        
        # Encode
        text_features = self.clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def forward(self, 
                part_texts: list,
                global_text: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            part_texts: List of part-level text descriptions (P texts per class)
            global_text: Optional global text description
            
        Returns:
            Tuple of (global_text_features, part_text_features)
                global_text_features: (B, D)
                part_text_features: (B, P, D)
        """
        # Encode part-level texts
        part_features_list = []
        
        # Batch encode part texts for efficiency
        # Assuming part_texts is a list of lists: [class1_parts, class2_parts, ...]
        if isinstance(part_texts[0], list):
            # Multiple classes
            for class_parts in part_texts:
                with torch.no_grad():
                    class_features = self.encode_text(class_parts)
                part_features_list.append(class_features.unsqueeze(0))
            part_features = torch.cat(part_features_list, dim=0)  # (B, P, D)
        else:
            # Single class
            with torch.no_grad():
                part_features = self.encode_text(part_texts).unsqueeze(0)  # (1, P, D)
        
        # Encode global text if provided
        if global_text is not None:
            with torch.no_grad():
                global_features = self.encode_text([global_text])  # (1, D)
        else:
            # Use concatenated part features as global
            global_features = part_features.mean(dim=1)  # (B, D)
        
        # Project global features
        global_features = self.global_proj(global_features)
        
        # Refine part-level features using local encoder
        part_features = self.local_encoder(part_features, global_features)
        
        # L2 normalize
        global_features = F.normalize(global_features, p=2, dim=-1)
        part_features = F.normalize(part_features, p=2, dim=-1)
        
        return global_features, part_features


class TextPromptGenerator:
    """
    Generates text prompts for action classes using LLM
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
    encoder = TextEncoder(clip_model_name="ViT-B/32", feature_dim=256, num_parts=6)
    
    # Test prompt generator
    generator = TextPromptGenerator()
    
    # Example action classes
    actions = ["running", "jumping", "walking"]
    
    # Generate prompts
    prompts = generator.generate_class_prompts(actions)
    
    print("Generated prompts for action 'running':")
    for i, prompt in enumerate(prompts["running"]):
        print(f"  {encoder.BODY_PARTS[i]}: {prompt}")
    
    # Test encoding (requires GPU)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # encoder = encoder.to(device)
    # global_feat, part_feat = encoder(prompts["running"])
    # print(f"\nGlobal feature shape: {global_feat.shape}")
    # print(f"Part features shape: {part_feat.shape}")
