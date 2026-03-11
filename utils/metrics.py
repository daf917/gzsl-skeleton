# Evaluation metrics for GZSL

import torch
import numpy as np
from typing import Dict, List, Tuple


def compute_acc(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Compute accuracy
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy percentage
    """
    correct = (y_true == y_pred).sum().item()
    total = y_true.shape[0]
    return 100.0 * correct / total if total > 0 else 0.0


def compute_gzsl_metrics(y_true: torch.Tensor, 
                        y_pred: torch.Tensor,
                        seen_classes: List[int],
                        unseen_classes: List[int]) -> Dict[str, float]:
    """
    Compute GZSL metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        seen_classes: List of seen class indices
        unseen_classes: List of unseen class indices
        
    Returns:
        Dictionary with Acc_s, Acc_u, and HM
    """
    # Convert to numpy for easier processing
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    # Compute seen accuracy
    seen_mask = np.isin(y_true, seen_classes)
    if seen_mask.sum() > 0:
        acc_s = compute_acc(
            torch.from_numpy(y_true[seen_mask]),
            torch.from_numpy(y_pred[seen_mask])
        )
    else:
        acc_s = 0.0
    
    # Compute unseen accuracy
    unseen_mask = np.isin(y_true, unseen_classes)
    if unseen_mask.sum() > 0:
        acc_u = compute_acc(
            torch.from_numpy(y_true[unseen_mask]),
            torch.from_numpy(y_pred[unseen_mask])
        )
    else:
        acc_u = 0.0
    
    # Compute harmonic mean
    if acc_s + acc_u > 0:
        hm = 2 * acc_s * acc_u / (acc_s + acc_u)
    else:
        hm = 0.0
    
    return {
        'Acc_s': acc_s,
        'Acc_u': acc_u,
        'HM': hm
    }


def compute_per_class_accuracy(y_true: torch.Tensor,
                               y_pred: torch.Tensor,
                               num_classes: int) -> np.ndarray:
    """
    Compute per-class accuracy
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Total number of classes
        
    Returns:
        Array of per-class accuracies
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    per_class_acc = np.zeros(num_classes)
    
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() > 0:
            per_class_acc[c] = (y_pred[mask] == c).sum() / mask.sum()
    
    return per_class_acc * 100.0


def compute_confusion_matrix(y_true: torch.Tensor,
                            y_pred: torch.Tensor,
                            num_classes: int) -> np.ndarray:
    """
    Compute confusion matrix
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes
        
    Returns:
        Confusion matrix (num_classes, num_classes)
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    conf_matrix = np.zeros((num_classes, num_classes))
    
    for true, pred in zip(y_true, y_pred):
        conf_matrix[true, pred] += 1
    
    return conf_matrix


def compute_top_k_accuracy(y_true: torch.Tensor,
                          y_pred_scores: torch.Tensor,
                          k: int = 5) -> float:
    """
    Compute top-k accuracy
    
    Args:
        y_true: Ground truth labels (B,)
        y_pred_scores: Prediction scores (B, C)
        k: Top k to consider
        
    Returns:
        Top-k accuracy percentage
    """
    B = y_true.shape[0]
    
    # Get top-k predictions
    _, top_k_pred = y_pred_scores.topk(k, dim=-1)
    
    # Check if true label is in top-k
    correct = (top_k_pred == y_true.unsqueeze(-1)).any(dim=-1).sum().item()
    
    return 100.0 * correct / B if B > 0 else 0.0


class GZSLEvaluator:
    """
    Evaluator for GZSL tasks
    """
    
    def __init__(self, seen_classes: List[int], unseen_classes: List[int]):
        """
        Args:
            seen_classes: List of seen class indices
            unseen_classes: List of unseen class indices
        """
        self.seen_classes = seen_classes
        self.unseen_classes = unseen_classes
        self.all_classes = seen_classes + unseen_classes
    
    def evaluate(self,
                y_true: torch.Tensor,
                y_pred: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate predictions
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with metrics
        """
        return compute_gzsl_metrics(y_true, y_pred, self.seen_classes, self.unseen_classes)
    
    def evaluate_scores(self,
                       y_true: torch.Tensor,
                       y_scores: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate using prediction scores
        
        Args:
            y_true: Ground truth labels
            y_scores: Prediction scores (B, C)
            
        Returns:
            Dictionary with metrics
        """
        # Get predictions
        y_pred = y_scores.argmax(dim=-1)
        
        return self.evaluate(y_true, y_pred)
    
    def compute_calibrated_scores(self,
                                 y_scores: torch.Tensor,
                                 gamma: float = 0.0) -> torch.Tensor:
        """
        Compute calibrated scores using gamma factor
        
        Args:
            y_scores: Prediction scores (B, C)
            gamma: Calibration factor (positive for unseen, negative for seen)
            
        Returns:
            Calibrated scores
        """
        calibrated = y_scores.clone()
        
        # Apply gamma to seen classes
        seen_indices = torch.tensor(self.seen_classes, device=y_scores.device)
        calibrated[:, seen_indices] -= gamma
        
        return calibrated
    
    def find_optimal_gamma(self,
                          y_true: torch.Tensor,
                          y_scores: torch.Tensor,
                          gamma_range: List[float] = None) -> Tuple[float, Dict[str, float]]:
        """
        Find optimal gamma for calibration
        
        Args:
            y_true: Ground truth labels
            y_scores: Prediction scores (B, C)
            gamma_range: Range of gamma values to try
            
        Returns:
            Tuple of (best_gamma, best_metrics)
        """
        if gamma_range is None:
            gamma_range = [i * 0.1 for i in range(-20, 21)]
        
        best_gamma = 0.0
        best_hm = 0.0
        best_metrics = {}
        
        for gamma in gamma_range:
            calibrated = self.compute_calibrated_scores(y_scores, gamma)
            metrics = self.evaluate_scores(y_true, calibrated)
            
            if metrics['HM'] > best_hm:
                best_hm = metrics['HM']
                best_gamma = gamma
                best_metrics = metrics
        
        return best_gamma, best_metrics


if __name__ == "__main__":
    # Test metrics
    y_true = torch.tensor([0, 0, 1, 1, 2, 2])
    y_pred = torch.tensor([0, 0, 1, 0, 2, 2])
    
    seen_classes = [0, 1]
    unseen_classes = [2]
    
    evaluator = GZSLEvaluator(seen_classes, unseen_classes)
    metrics = evaluator.evaluate(y_true, y_pred)
    
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}")
