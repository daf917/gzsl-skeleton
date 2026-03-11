"""
Few-Shot Learning Module

This module implements few-shot learning support as described in the paper:
- K-shot adaptation on HMDB-51 and SSv2 (K in {2, 4, 8, 16})
- Episodic training for better generalization
- Test-time adaptation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm


class FewShotSampler(Sampler):
    """
    Sampler for few-shot learning episodes
    """
    
    def __init__(self, 
                 labels: List[int],
                 num_way: int = 5,
                 num_shots: int = 16,
                 num_episodes: int = 1000,
                 shuffle: bool = True):
        """
        Args:
            labels: List of all sample labels
            num_way: Number of classes per episode
            num_shots: Number of samples per class
            num_episodes: Number of episodes to sample
            shuffle: Whether to shuffle classes
        """
        self.labels = np.array(labels)
        self.num_way = num_way
        self.num_shots = num_shots
        self.num_episodes = num_episodes
        self.shuffle = shuffle
        
        # Get unique classes
        self.unique_classes = np.unique(self.labels)
    
    def __iter__(self):
        """Generate episodes"""
        for _ in range(self.num_episodes):
            # Sample classes
            if self.shuffle:
                episode_classes = np.random.choice(
                    self.unique_classes, 
                    self.num_way, 
                    replace=False
                )
            else:
                episode_classes = self.unique_classes[:self.num_way]
            
            # Sample samples for each class
            episode_indices = []
            
            for cls in episode_classes:
                cls_indices = np.where(self.labels == cls)[0]
                
                # Sample shots
                if len(cls_indices) >= self.num_shots:
                    selected = np.random.choice(cls_indices, self.num_shots, replace=False)
                else:
                    # If not enough samples, repeat
                    selected = np.random.choice(cls_indices, self.num_shots, replace=True)
                
                episode_indices.extend(selected)
            
            # Shuffle within episode
            if self.shuffle:
                np.random.shuffle(episode_indices)
            
            yield from episode_indices
    
    def __len__(self):
        return self.num_episodes * self.num_way * self.num_shots


class EpisodicDataLoader:
    """
    DataLoader for episodic few-shot learning
    """
    
    def __init__(self,
                 dataset,
                 num_way: int = 5,
                 num_shots: int = 16,
                 num_episodes: int = 1000,
                 batch_size: int = 1,
                 shuffle: bool = True):
        """
        Args:
            dataset: Dataset instance
            num_way: Number of classes per episode
            num_shots: Number of samples per class
            num_episodes: Number of episodes per epoch
            batch_size: Batch size (typically 1 for episodic)
            shuffle: Whether to shuffle
        """
        self.dataset = dataset
        self.num_way = num_way
        self.num_shots = num_shots
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        
        # Create sampler
        labels = [dataset[i][1] for i in range(len(dataset))]
        
        self.sampler = FewShotSampler(
            labels=labels,
            num_way=num_way,
            num_shots=num_shots,
            num_episodes=num_episodes,
            shuffle=shuffle
        )
        
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size * num_way * num_shots,
            sampler=self.sampler,
            num_workers=0,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Custom collate function for episodes"""
        skeletons, labels = zip(*batch)
        
        # Reshape to episodes
        episode_size = self.num_way * self.num_shots
        
        skeletons = torch.stack(skeletons)  # (N, T, J, 3)
        labels = torch.tensor(labels)  # (N,)
        
        return skeletons, labels
    
    def __iter__(self):
        return iter(self.loader)
    
    def __len__(self):
        return self.num_episodes


class FewShotModel(nn.Module):
    """
    Few-shot learning wrapper for GZSL model
    """
    
    def __init__(self, base_model: nn.Module, num_way: int = 5):
        """
        Args:
            base_model: Base GZSL model
            num_way: Number of ways for few-shot
        """
        super().__init__()
        
        self.base_model = base_model
        self.num_way = num_way
        
        # Feature embedding dimension
        self.feature_dim = base_model.feature_dim
    
    def forward(self, 
                skeleton: torch.Tensor,
                return_features: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            skeleton: Skeleton tensor (B, T, J, 3)
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary with predictions and features
        """
        return self.base_model.forward(skeleton, return_features=return_features)
    
    def compute_prototypes(self, 
                         support_skeletons: torch.Tensor,
                         support_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute class prototypes from support set
        
        Args:
            support_skeletons: Support set skeletons (N, T, J, 3)
            support_labels: Support set labels (N,)
            
        Returns:
            Class prototypes (num_way, D)
        """
        # Extract features
        with torch.no_grad():
            outputs = self.base_model.forward(support_skeletons)
        
        features = outputs['global_composed']  # (N, D)
        
        # Compute prototypes (mean of each class)
        prototypes = []
        
        unique_labels = support_labels.unique()
        
        for label in unique_labels:
            mask = support_labels == label
            class_features = features[mask]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)  # (num_way, D)
        
        return prototypes
    
    def few_shot_predict(self,
                        query_skeletons: torch.Tensor,
                        prototypes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Few-shot prediction using prototypes
        
        Args:
            query_skeletons: Query skeletons (M, T, J, 3)
            prototypes: Class prototypes (num_way, D)
            
        Returns:
            Tuple of (predictions, similarities)
        """
        # Extract query features
        with torch.no_grad():
            outputs = self.base_model.forward(query_skeletons)
        
        query_features = outputs['global_composed']  # (M, D)
        
        # Compute similarities with prototypes
        similarities = F.cosine_similarity(
            query_features.unsqueeze(1),
            prototypes.unsqueeze(0),
            dim=-1
        )  # (M, num_way)
        
        # Get predictions
        predictions = similarities.argmax(dim=-1)  # (M,)
        
        return predictions, similarities


class TestTimeAdaptation:
    """
    Test-time adaptation for few-shot setting
    """
    
    def __init__(self, 
                 model: FewShotModel,
                 lr: float = 0.01,
                 num_steps: int = 10):
        """
        Args:
            model: Few-shot model
            lr: Learning rate for adaptation
            num_steps: Number of adaptation steps
        """
        self.model = model
        self.lr = lr
        self.num_steps = num_steps
    
    def adapt(self,
             support_skeletons: torch.Tensor,
             support_labels: torch.Tensor,
             query_skeletons: torch.Tensor,
             text_features: torch.Tensor) -> torch.Tensor:
        """
        Adapt model using support set and make predictions on query set
        
        Args:
            support_skeletons: Support set (N, T, J, 3)
            support_labels: Support labels (N,)
            query_skeletons: Query set (M, T, J, 3)
            text_features: Text features for all classes (C, D)
            
        Returns:
            Predictions for query set (M,)
        """
        # Clone model for adaptation
        adapted_model = FewShotModel(
            self.model.base_model,
            num_way=self.model.num_way
        )
        adapted_model.load_state_dict(self.model.state_dict())
        
        # Set to train mode for adaptation
        adapted_model.train()
        
        # Optimizer for adaptation
        optimizer = torch.optim.Adam(
            adapted_model.parameters(),
            lr=self.lr
        )
        
        # Adaptation steps
        for step in range(self.num_steps):
            optimizer.zero_grad()
            
            # Forward pass on support set
            outputs = adapted_model(support_skeletons)
            support_features = outputs['global_composed']
            
            # Compute prototypes
            prototypes = []
            unique_labels = support_labels.unique()
            
            for label in unique_labels:
                mask = support_labels == label
                class_features = support_features[mask]
                prototype = class_features.mean(dim=0)
                prototypes.append(prototype)
            
            if len(prototypes) > 0:
                prototypes = torch.stack(prototypes)
                
                # Compute loss (prototype-based classification)
                support_sim = F.cosine_similarity(
                    support_features.unsqueeze(1),
                    prototypes.unsqueeze(0),
                    dim=-1
                )
                
                # Get target labels (convert to 0-based indices)
                target_idx = torch.zeros_like(support_labels)
                for i, label in enumerate(unique_labels):
                    target_idx[support_labels == label] = i
                
                loss = F.cross_entropy(support_sim, target_idx)
                
                loss.backward()
                optimizer.step()
        
        # Set to eval mode for prediction
        adapted_model.eval()
        
        # Make predictions
        with torch.no_grad():
            # Compute prototypes from adapted model
            outputs = adapted_model(support_skeletons)
            support_features = outputs['global_composed']
            
            prototypes = []
            unique_labels = support_labels.unique()
            
            for label in unique_labels:
                mask = support_labels == label
                class_features = support_features[mask]
                prototype = class_features.mean(dim=0)
                prototypes.append(prototype)
            
            if len(prototypes) > 0:
                prototypes = torch.stack(prototypes)
                
                # Query predictions
                outputs = adapted_model(query_skeletons)
                query_features = outputs['global_composed']
                
                similarities = F.cosine_similarity(
                    query_features.unsqueeze(1),
                    prototypes.unsqueeze(0),
                    dim=-1
                )
                
                predictions = similarities.argmax(dim=-1)
            else:
                # Fallback: use text features
                outputs = adapted_model(query_skeletons)
                query_features = outputs['global_composed']
                similarities = torch.matmul(query_features, text_features.T)
                predictions = similarities.argmax(dim=-1)
        
        return predictions


def train_few_shot(model: nn.Module,
                  train_loader: EpisodicDataLoader,
                  val_loader: EpisodicDataLoader,
                  optimizer: torch.optim.Optimizer,
                  device: torch.device,
                  num_epochs: int = 100,
                  num_way: int = 5,
                  save_path: Optional[str] = None) -> Dict[str, List[float]]:
    """
    Train model with few-shot episodic training
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        device: Device to use
        num_epochs: Number of epochs
        num_way: Number of ways
        save_path: Path to save best model
        
    Returns:
        Training history
    """
    history = {
        'train_loss': [],
        'val_acc': []
    }
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for skeletons, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            skeletons = skeletons.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(skeletons)
            features = outputs['global_composed']
            
            # Compute prototype-based loss
            unique_labels = labels.unique()
            
            # Compute prototypes
            prototypes = []
            for label in unique_labels:
                mask = labels == label
                class_features = features[mask]
                prototype = class_features.mean(dim=0)
                prototypes.append(prototype)
            
            prototypes = torch.stack(prototypes)
            
            # Compute similarities
            similarities = F.cosine_similarity(
                features.unsqueeze(1),
                prototypes.unsqueeze(0),
                dim=-1
            )
            
            # Create target indices
            target = torch.zeros_like(labels)
            for i, label in enumerate(unique_labels):
                target[labels == label] = i
            
            loss = F.cross_entropy(similarities, target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation
        val_acc = evaluate_few_shot(model, val_loader, device, num_way)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Val Acc={val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc and save_path:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with acc={best_acc:.2f}%")
    
    return history


def evaluate_few_shot(model: nn.Module,
                     data_loader: DataLoader,
                     device: torch.device,
                     num_way: int = 5) -> float:
    """
    Evaluate model in few-shot setting
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device
        num_way: Number of ways
        
    Returns:
        Accuracy
    """
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for skeletons, labels in data_loader:
            skeletons = skeletons.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(skeletons)
            features = outputs['global_composed']
            
            # Compute prototypes
            unique_labels = labels.unique()
            
            prototypes = []
            for label in unique_labels:
                mask = labels == label
                class_features = features[mask]
                prototype = class_features.mean(dim=0)
                prototypes.append(prototype)
            
            if len(prototypes) > 0:
                prototypes = torch.stack(prototypes)
                
                # Predictions
                similarities = F.cosine_similarity(
                    features.unsqueeze(1),
                    prototypes.unsqueeze(0),
                    dim=-1
                )
                
                preds = similarities.argmax(dim=-1)
                
                # Convert to original labels
                for i, label in enumerate(unique_labels):
                    preds[preds == i] = label
                
                correct += (preds == labels).sum().item()
                total += labels.shape[0]
    
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    return accuracy


def run_few_shot_experiment(model: nn.Module,
                           dataset,
                           num_shots: int = 16,
                           num_way: int = 5,
                           num_episodes: int = 1000,
                           device: str = 'cuda') -> Dict[str, float]:
    """
    Run few-shot experiment as described in paper
    
    Args:
        model: Model to test
        dataset: Dataset
        num_shots: Number of shots (K in {2, 4, 8, 16})
        num_way: Number of ways
        num_episodes: Number of test episodes
        device: Device
        
    Returns:
        Results dictionary
    """
    # Create few-shot data loader
    few_shot_loader = EpisodicDataLoader(
        dataset=dataset,
        num_way=num_way,
        num_shots=num_shots,
        num_episodes=num_episodes,
        batch_size=1,
        shuffle=False  # Fixed episodes for evaluation
    )
    
    # Evaluate
    accuracy = evaluate_few_shot(model, few_shot_loader, device, num_way)
    
    return {
        'accuracy': accuracy,
        'num_shots': num_shots,
        'num_way': num_way,
        'num_episodes': num_episodes
    }


if __name__ == "__main__":
    # Test few-shot module
    print("Testing few-shot module...")
    
    # Create dummy dataset
    class DummyDataset:
        def __init__(self, num_samples=100, num_classes=10):
            self.num_samples = num_samples
            self.num_classes = num_classes
            
            # Generate random skeletons
            self.skeletons = torch.randn(num_samples, 64, 25, 3)
            self.labels = torch.randint(0, num_classes, (num_samples,))
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            return self.skeletons[idx], self.labels[idx]
    
    dataset = DummyDataset(num_samples=200, num_classes=20)
    
    # Test sampler
    sampler = FewShotSampler(
        labels=[dataset[i][1] for i in range(len(dataset))],
        num_way=5,
        num_shots=4,
        num_episodes=10
    )
    
    print(f"Created {len(sampler)} episodes")
    
    # Test episodic loader
    loader = EpisodicDataLoader(
        dataset=dataset,
        num_way=5,
        num_shots=4,
        num_episodes=10
    )
    
    print(f"Created loader with {len(loader)} batches")
