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
from .aggregation import (
    AlignmentLoss,
    GlobalConsistencyLoss,
    PrimitiveAggregation,
    PrimitiveDecorrelationRegularizer,
)
from data.motion_attribute import MotionAttributeExtractor, create_part_joint_mapping


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
        
        # Extract nested YAML config while preserving support for flat dicts.
        model_config = config.get('model', config)
        dataset_config = config.get('dataset', config)
        training_config = config.get('training', config)

        self.feature_dim = model_config.get('feature_dim', config.get('feature_dim', 512))
        self.num_parts = model_config.get('num_parts', config.get('num_parts', 6))
        self.temperature = training_config.get('temperature', config.get('temperature', 0.07))
        
        # Loss weights
        self.lambda_p = training_config.get('lambda_p', config.get('lambda_p', 1.0))  # Primitive alignment
        self.lambda_g = training_config.get('lambda_g', config.get('lambda_g', 1.0))  # Global alignment
        self.lambda_c = training_config.get('lambda_c', config.get('lambda_c', 0.5))   # Consistency
        legacy_weight = training_config.get('lambda_i', config.get('lambda_i'))
        # Old configurations weighted the average instead of the sum of correlations.
        self.num_part_pairs = max(self.num_parts * (self.num_parts - 1), 1)
        default_decorr_weight = 0.1 if legacy_weight is None else legacy_weight / self.num_part_pairs
        self.lambda_d = training_config.get('lambda_d', config.get('lambda_d', default_decorr_weight))
        
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
        num_joints = dataset_config.get('num_joints', config.get('num_joints', 25))
        self.skeleton_encoder = SkeletonEncoder(
            num_joints=num_joints,
            num_classes=dataset_config.get('num_classes', config.get('num_classes', 60)),
            feature_dim=self.feature_dim,
            num_parts=self.num_parts,
            dropout=model_config.get('dropout', config.get('dropout', 0.5)),
            pretrained_path=skeleton_config.get('shift_gcn_pretrained', None)
        )
        
        # Motion attribute extractor
        self.motion_extractor = MotionAttributeExtractor(
            num_parts=self.num_parts,
            temporal_window=model_config.get('temporal_window', config.get('temporal_window', 10))
        )
        
        self.motion_extractor.part_joints = create_part_joint_mapping(num_joints)

        # Aggregation module
        self.aggregation = PrimitiveAggregation(
            feature_dim=self.feature_dim,
            num_parts=self.num_parts
        )
        
        # Loss functions
        self.alignment_loss = AlignmentLoss(temperature=self.temperature)
        self.decorrelation_loss = PrimitiveDecorrelationRegularizer()
        self.consistency_loss = GlobalConsistencyLoss()
        
        self.register_buffer("attr_mean", None, persistent=False)
        self.register_buffer("attr_std", None, persistent=False)

    def set_motion_attribute_statistics(self, mean, std) -> None:
        """Set the training split's fitted statistics for preprocessing at all stages."""
        parameter = next(self.parameters())
        mean = torch.as_tensor(mean, device=parameter.device, dtype=parameter.dtype)
        std = torch.as_tensor(std, device=parameter.device, dtype=parameter.dtype)
        if mean.shape != (8,) or std.shape != (8,):
            raise ValueError("Motion-attribute mean and standard deviation must each have shape (8,).")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 0).any():
            raise ValueError("Motion-attribute statistics must be finite with nonnegative standard deviations.")
        self.attr_mean = mean.detach().clone()
        self.attr_std = std.detach().clone()

    def extract_motion_attributes(self, skeleton: torch.Tensor, 
                                  normalize: bool = True) -> torch.Tensor:
        """
        Extract motion attributes from skeleton sequence
        
        Args:
            skeleton: Skeleton tensor (B, T, J, 2)
            normalize: Whether to normalize attributes
            
        Returns:
            Motion attributes (B, T, P, 8)
        """
        attributes = self.motion_extractor.compute_all_parts(skeleton)

        if normalize:
            if self.attr_mean is None or self.attr_std is None:
                raise RuntimeError("Set motion-attribute statistics from the training split before normalization.")
            attributes, _, _ = self.motion_extractor.normalize_attributes(attributes, self.attr_mean, self.attr_std)

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
        descriptions = [
            part_descriptions.get(label) or [f"{class_names[label]} motion"] * self.num_parts
            for label in class_labels
        ]
        return self.text_encoder(descriptions)

    def forward(self, 
                skeleton: torch.Tensor,
                class_labels: Optional[torch.Tensor] = None,
                class_names: Optional[Dict[int, str]] = None,
                part_descriptions: Optional[Dict[int, List[str]]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            skeleton: Skeleton sequence (B, T, J, 2)
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
                    part_descriptions: Dict[int, List[str]],
                    seen_classes: Optional[List[int]] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss
        
        Args:
            outputs: Model outputs
            labels: Ground truth labels (B,)
            class_names: Class names dictionary
            part_descriptions: Part descriptions dictionary
            seen_classes: Ordered IDs for the full set of seen-class anchors.
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Get features
        global_skel = outputs['global_skel']
        part_skel = outputs['part_skel']
        global_composed = outputs['global_composed']
        
        anchor_classes = list(seen_classes) if seen_classes is not None else labels.unique().tolist()
        if not anchor_classes or len(set(anchor_classes)) != len(anchor_classes):
            raise ValueError("Seen-class IDs must be nonempty and unique.")
        anchor_ids = torch.tensor(anchor_classes, device=labels.device, dtype=labels.dtype)
        matches = labels[:, None] == anchor_ids
        if not matches.any(dim=1).all():
            raise ValueError("Training samples must belong to the selected seen classes.")
        targets = matches.long().argmax(dim=1)

        global_text, part_text = self.encode_text(anchor_classes, class_names, part_descriptions)
        global_align_loss, part_align_loss = self.alignment_loss(
            global_skel, part_skel, global_text, part_text, targets
        )

        # Compute consistency loss
        consis_loss = self.consistency_loss(global_composed, global_skel)
        
        decorr_loss = self.decorrelation_loss(part_skel)
        
        # Total loss
        total_loss = (
            self.lambda_g * global_align_loss +
            self.lambda_p * part_align_loss +
            self.lambda_c * consis_loss +
            self.lambda_d * decorr_loss
        )
        
        decorr_value = decorr_loss.item()
        loss_dict = {
            'total': total_loss.item(),
            'global_align': global_align_loss.item(),
            'part_align': part_align_loss.item(),
            'consistency': consis_loss.item(),
            'decorrelation': decorr_value,
            'independence': decorr_value / self.num_part_pairs,  # Legacy metric key
        }
        
        return total_loss, loss_dict
    
    def predict(self, 
               skeleton: torch.Tensor,
               text_features_global: torch.Tensor,
               text_features_part: torch.Tensor) -> torch.Tensor:
        """
        Predict class scores for given skeleton
        
        Args:
            skeleton: Skeleton sequence (B, T, J, 2)
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
            skeleton: Skeleton sequence (B, T, J, 2)
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
        'feature_dim': 512,
        'num_parts': 6,
        'temperature': 0.07,
        'lambda_p': 1.0,
        'lambda_g': 1.0,
        'lambda_c': 0.5,
        'lambda_d': 0.1,
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
    model.set_motion_attribute_statistics(torch.zeros(8), torch.ones(8))
    
    # Example input
    B, T, J = 4, 64, 25
    skeleton = torch.randn(B, T, J, 2)
    labels = torch.tensor([0, 1, 2, 3])
    
    # Forward pass
    outputs = model.forward(skeleton)
    
    print("Model outputs:")
    for k, v in outputs.items():
        if v is not None:
            print(f"  {k}: {v.shape}")
