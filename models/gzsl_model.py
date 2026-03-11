"""
Main GZSL Model
Combines text encoder, skeleton encoder, and aggregation module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional

from .text_encoder import TextEncoder
from .skeleton_encoder import SkeletonEncoder
from .aggregation import PrimitiveAggregation, PrimitiveIndependenceRegularizer, GlobalConsistencyLoss, AlignmentLoss
from data.motion_attribute import MotionAttributeExtractor


class GZSLModel(nn.Module):
    """
    Generalized Zero-Shot Skeleton Action Recognition Model
    
    Combines:
    - Text Encoder: CLIP + Local Text Encoder
    - Skeleton Encoder: Shift-GCN + Local Skeleton Encoder
    - Aggregation Module: Compositional global representation
    """
    
    def __init__(self,
                 config: dict):
        """
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        # Extract config
        self.feature_dim = config.get('feature_dim', 256)
        self.num_parts = config.get('num_parts', 6)
        self.temperature = config.get('temperature', 0.07)
        
        # Loss weights
        self.lambda_p = config.get('lambda_p', 1.0)  # Primitive alignment
        self.lambda_g = config.get('lambda_g', 1.0)  # Global alignment
        self.lambda_c = config.get('lambda_c', 0.5)   # Consistency
        self.lambda_i = config.get('lambda_i', 0.3)   # Independence
        
        # Build components
        # Text encoder
        text_config = config.get('text_encoder', {})
        self.text_encoder = TextEncoder(
            clip_model_name=text_config.get('clip_model', 'ViT-B/32'),
            freeze_clip=text_config.get('freeze_clip', True),
            feature_dim=self.feature_dim,
            num_parts=self.num_parts
        )
        
        # Skeleton encoder
        skeleton_config = config.get('skeleton_encoder', {})
        self.skeleton_encoder = SkeletonEncoder(
            num_joints=config.get('num_joints', 25),
            num_classes=config.get('num_classes', 60),
            feature_dim=self.feature_dim,
            num_parts=self.num_parts,
            dropout=config.get('dropout', 0.5),
            pretrained_path=skeleton_config.get('shift_gcn_pretrained', None)
        )
        
        # Motion attribute extractor
        self.motion_extractor = MotionAttributeExtractor(
            num_parts=self.num_parts,
            temporal_window=config.get('temporal_window', 10)
        )
        
        # Aggregation module
        self.aggregation = PrimitiveAggregation(
            feature_dim=self.feature_dim,
            num_parts=self.num_parts
        )
        
        # Loss functions
        self.alignment_loss = AlignmentLoss(temperature=self.temperature)
        self.independence_loss = PrimitiveIndependenceRegularizer()
        self.consistency_loss = GlobalConsistencyLoss()
        
        # Store normalization statistics
        self.attr_mean = None
        self.attr_std = None
        
        # Text features cache for seen classes
        self.text_features_cache = {}
        
    def extract_motion_attributes(self, skeleton: torch.Tensor, 
                                  normalize: bool = True) -> torch.Tensor:
        """
        Extract motion attributes from skeleton sequence
        
        Args:
            skeleton: Skeleton tensor (B, T, J, 3)
            normalize: Whether to normalize attributes
            
        Returns:
            Motion attributes (B, T, P, 8)
        """
        B = skeleton.shape[0]
        
        # Compute motion attributes for each sample
        attributes_list = []
        
        for i in range(B):
            attrs = self.motion_extractor.compute_all_parts(skeleton[i])
            attributes_list.append(attrs)
        
        attributes = torch.stack(attributes_list, dim=0)  # (B, T, P, 8)
        
        # Normalize
        if normalize:
            if self.attr_mean is None:
                # Compute statistics from first batch
                attributes, self.attr_mean, self.attr_std = self.motion_extractor.normalize_attributes(attributes)
            else:
                attributes = (attributes - self.attr_mean) / (self.attr_std + 1e-8)
        
        return attributes
    
    def encode_text(self, class_labels: List[int], 
                    class_names: Dict[int, str],
                    part_descriptions: Dict[int, List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text for given classes
        
        Args:
            class_labels: List of class indices
            class_names: Dictionary mapping class index to class name
            part_descriptions: Dictionary mapping class index to list of part descriptions
            
        Returns:
            Tuple of (global_text_features, part_text_features)
        """
        global_features = []
        part_features_list = []
        
        for label in class_labels:
            # Get part descriptions
            parts = part_descriptions.get(label, [f"{class_names[label]} motion"] * self.num_parts)
            
            # Encode using text encoder (need to implement batch encoding)
            # For now, assume single class encoding
            global_feat, part_feat = self.text_encoder(parts)
            
            global_features.append(global_feat)
            part_features_list.append(part_feat)
        
        global_features = torch.cat(global_features, dim=0)
        part_features_list = torch.cat(part_features_list, dim=0)
        
        return global_features, part_features_list
    
    def forward(self, 
                skeleton: torch.Tensor,
                class_labels: Optional[torch.Tensor] = None,
                class_names: Optional[Dict[int, str]] = None,
                part_descriptions: Optional[Dict[int, List[str]]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            skeleton: Skeleton sequence (B, T, J, 3)
            class_labels: Class labels (B,) - for training
            class_names: Dictionary of class names
            part_descriptions: Dictionary of part descriptions per class
            
        Returns:
            Dictionary containing:
                - global_skel: Global skeleton features
                - part_skel: Part-level skeleton features
                - global_composed: Compositional global features
                - attention_weights: Attention weights for aggregation
                - global_text: Global text features
                - part_text: Part-level text features
        """
        # Extract motion attributes
        motion_attrs = self.extract_motion_attributes(skeleton)
        
        # Get skeleton features
        global_skel, part_skel = self.skeleton_encoder(skeleton, motion_attrs)
        
        # Aggregate part features into compositional global
        global_composed, attention_weights = self.aggregation(part_skel)
        
        # Encode text features (if labels provided)
        global_text = None
        part_text = None
        
        if class_labels is not None and class_names is not None and part_descriptions is not None:
            # Get unique labels
            unique_labels = class_labels.unique().tolist()
            
            # Encode text for these classes
            global_text, part_text = self.encode_text(
                unique_labels,
                class_names,
                part_descriptions
            )
        
        return {
            'global_skel': global_skel,
            'part_skel': part_skel,
            'global_composed': global_composed,
            'attention_weights': attention_weights,
            'global_text': global_text,
            'part_text': part_text
        }
    
    def compute_loss(self, 
                    outputs: Dict[str, torch.Tensor],
                    labels: torch.Tensor,
                    class_names: Dict[int, str],
                    part_descriptions: Dict[int, List[str]]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss
        
        Args:
            outputs: Model outputs
            labels: Ground truth labels (B,)
            class_names: Class names dictionary
            part_descriptions: Part descriptions dictionary
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Get features
        global_skel = outputs['global_skel']
        part_skel = outputs['part_skel']
        global_composed = outputs['global_composed']
        
        # Get text features for the labels
        global_text, part_text = self.encode_text(
            labels.unique().tolist(),
            class_names,
            part_descriptions
        )
        
        # Expand text features to match batch
        label_to_idx = {l: i for i, l in enumerate(labels.unique().tolist())}
        global_text_batch = global_text[[label_to_idx[l.item()] for l in labels]]
        part_text_batch = part_text[[label_to_idx[l.item()] for l in labels]]
        
        # Compute alignment losses
        global_align_loss, part_align_loss = self.alignment_loss(
            global_skel, part_skel,
            global_text_batch, part_text_batch,
            labels
        )
        
        # Compute consistency loss
        consis_loss = self.consistency_loss(global_composed, global_skel)
        
        # Compute independence loss
        ind_loss = self.independence_loss(part_skel)
        
        # Total loss
        total_loss = (
            self.lambda_g * global_align_loss +
            self.lambda_p * part_align_loss +
            self.lambda_c * consis_loss +
            self.lambda_i * ind_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'global_align': global_align_loss.item(),
            'part_align': part_align_loss.item(),
            'consistency': consis_loss.item(),
            'independence': ind_loss.item()
        }
        
        return total_loss, loss_dict
    
    def predict(self, 
               skeleton: torch.Tensor,
               text_features_global: torch.Tensor,
               text_features_part: torch.Tensor) -> torch.Tensor:
        """
        Predict class scores for given skeleton
        
        Args:
            skeleton: Skeleton sequence (B, T, J, 3)
            text_features_global: Global text features for all classes (C, D)
            text_features_part: Part-level text features for all classes (C, P, D)
            
        Returns:
            Similarity scores (B, C)
        """
        # Extract features
        outputs = self.forward(skeleton)
        
        global_composed = outputs['global_composed']  # (B, D)
        
        # Compute similarity with global text features
        scores = torch.matmul(global_composed, text_features_global.T)  # (B, C)
        
        return scores


class GZSLClassifier:
    """
    GZSL Classifier that handles inference
    """
    
    def __init__(self, model: GZSLModel, seen_classes: List[int], unseen_classes: List[int]):
        """
        Args:
            model: GZSL model
            seen_classes: List of seen class indices
            unseen_classes: List of unseen class indices
        """
        self.model = model
        self.seen_classes = seen_classes
        self.unseen_classes = unseen_classes
        self.all_classes = seen_classes + unseen_classes
        
        self.text_features_global = None
        self.text_features_part = None
        
    def set_text_features(self, 
                         global_features: torch.Tensor, 
                         part_features: torch.Tensor):
        """
        Set text features for all classes
        
        Args:
            global_features: Global text features (C, D)
            part_features: Part-level text features (C, P, D)
        """
        self.text_features_global = global_features
        self.text_features_part = part_features
    
    def predict(self, 
               skeleton: torch.Tensor,
               return_seen_unseen: bool = True) -> Dict[str, torch.Tensor]:
        """
        Predict class
        
        Args:
            skeleton: Skeleton sequence (B, T, J, 3)
            return_seen_unseen: Whether to return separate seen/unseen predictions
            
        Returns:
            Dictionary with predictions
        """
        # Extract model features
        outputs = self.model.forward(skeleton)
        global_composed = outputs['global_composed']
        
        # Compute similarities with all class text features
        all_scores = torch.matmul(global_composed, self.text_features_global.T)  # (B, C)
        
        # Get predictions
        pred_classes = torch.argmax(all_scores, dim=-1)  # (B,)
        
        result = {
            'pred': pred_classes,
            'scores': all_scores
        }
        
        if return_seen_unseen:
            # Compute separate accuracies for seen and unseen
            seen_scores = all_scores[:, self.seen_classes]
            unseen_scores = all_scores[:, self.unseen_classes]
            
            result['seen_scores'] = seen_scores
            result['unseen_scores'] = unseen_scores
        
        return result


if __name__ == "__main__":
    # Test GZSL model
    config = {
        'feature_dim': 256,
        'num_parts': 6,
        'temperature': 0.07,
        'lambda_p': 1.0,
        'lambda_g': 1.0,
        'lambda_c': 0.5,
        'lambda_i': 0.3,
        'num_joints': 25,
        'num_classes': 60,
        'dropout': 0.5,
        'temporal_window': 10,
        'text_encoder': {
            'clip_model': 'ViT-B/32',
            'freeze_clip': True
        }
    }
    
    model = GZSLModel(config)
    
    # Dummy input
    B, T, J = 4, 64, 25
    skeleton = torch.randn(B, T, J, 3)
    labels = torch.tensor([0, 1, 2, 3])
    
    # Forward pass
    outputs = model.forward(skeleton)
    
    print("Model outputs:")
    for k, v in outputs.items():
        if v is not None:
            print(f"  {k}: {v.shape}")
