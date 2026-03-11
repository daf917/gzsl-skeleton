"""
GZSL: Generalized Zero-Shot Skeleton Action Recognition

This project implements the paper:
"Generalized Zero-Shot Skeleton Action Recognition with Compositional Motion-Attribute Primitives"
"""

__version__ = "1.0.0"

from .models import GZSLModel, GZSLClassifier
from .data import create_dataset, GZSLSplit, MotionAttributeExtractor
from .utils.metrics import compute_gzsl_metrics

__all__ = [
    'GZSLModel',
    'GZSLClassifier', 
    'create_dataset',
    'GZSLSplit',
    'MotionAttributeExtractor',
    'compute_gzsl_metrics'
]
