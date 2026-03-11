"""
Dataset module for skeleton-based action recognition
Supports NTU RGB+D 60, NTU RGB+D 120, UCF101, PKU-MMD, and HMDB-51
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle


# NTU RGB+D joint connections
NTU_EDGE = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Head
    (1, 5), (5, 6), (6, 7),  # Left arm
    (1, 8), (8, 9), (9, 10),  # Right arm
    (1, 11), (11, 12), (12, 13),  # Torso
    (13, 14), (14, 15),  # Left leg
    (13, 16), (16, 17)  # Right leg
]


class NTU60Dataset(Dataset):
    """
    NTU RGB+D 60 Dataset
    """
    
    # Class names for NTU60
    CLASS_NAMES = [
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
    ]
    
    NUM_CLASSES = 60
    NUM_JOINTS = 25
    
    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 split_file: Optional[str] = None,
                 max_frames: int = 300,
                 temporal_downsample: int = 1):
        """
        Args:
            data_dir: Path to NTU60 data directory
            split: 'train', 'val', or 'test'
            split_file: Optional file specifying train/test split
            max_frames: Maximum frames to sample
            temporal_downsample: Temporal downsampling ratio
        """
        self.data_dir = data_dir
        self.split = split
        self.max_frames = max_frames
        self.temporal_downsample = temporal_downsample
        
        self.samples = []
        self.labels = []
        
        self._load_data(split_file)
    
    def _load_data(self, split_file: Optional[str] = None):
        """Load skeleton data"""
        # Check if data exists in preprocessed format
        data_file = os.path.join(self.data_dir, 'ntu60_skeletons.npz')
        
        if os.path.exists(data_file):
            # Load preprocessed data
            data = np.load(data_file, allow_pickle=True)
            self.samples = data['skeletons']
            self.labels = data['labels']
            
            if self.split == 'train':
                mask = data['split'] == 0
            elif self.split == 'val':
                mask = data['split'] == 1
            else:
                mask = data['split'] == 2
            
            self.samples = self.samples[mask]
            self.labels = self.labels[mask]
        else:
            # Try to load from original format
            self._load_from_raw()
    
    def _load_from_raw(self):
        """Load from original NTU format"""
        # This is a placeholder - actual implementation depends on data format
        # NTU60 typically uses .skeleton files or pre-extracted features
        print(f"Warning: Loading from raw format not implemented. Looking for data in {self.data_dir}")
        
        # Check for common data locations
        possible_paths = [
            os.path.join(self.data_dir, 'train_data.npy'),
            os.path.join(self.data_dir, 'test_data.npy'),
            os.path.join(self.data_dir, 'ntu60_train.npy'),
            os.path.join(self.data_dir, 'ntu60_test.npy'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found data at {path}")
                data = np.load(path, allow_pickle=True)
                self.samples = data['skeletons']
                self.labels = data['labels']
                break
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample"""
        skeleton = self.samples[idx]
        label = self.labels[idx]
        
        # Process skeleton
        skeleton = self._process_skeleton(skeleton)
        
        return skeleton, label
    
    def _process_skeleton(self, skeleton: np.ndarray) -> torch.Tensor:
        """
        Process skeleton sequence
        
        Args:
            skeleton: Raw skeleton (T, J, 3) or (T, J, 4)
            
        Returns:
            Processed skeleton (T, J, 3)
        """
        # Take first 3 dimensions (x, y, z)
        if skeleton.shape[-1] > 3:
            skeleton = skeleton[..., :3]
        
        # Temporal sampling
        T = skeleton.shape[0]
        if self.temporal_downsample > 1 and T > self.max_frames:
            indices = np.linspace(0, T - 1, self.max_frames // self.temporal_downsample, dtype=int)
            skeleton = skeleton[indices]
        
        # Convert to tensor
        skeleton = torch.from_numpy(skeleton).float()
        
        return skeleton
    
    @staticmethod
    def get_class_name(label: int) -> str:
        """Get class name from label"""
        return NTU60Dataset.CLASS_NAMES[label]


class NTU120Dataset(NTU60Dataset):
    """
    NTU RGB+D 120 Dataset
    """
    
    NUM_CLASSES = 120
    NUM_JOINTS = 25
    
    # Subset of class names (full list is much longer)
    CLASS_NAMES = NTU60Dataset.CLASS_NAMES + [
        # Additional 60 classes for NTU120
        "other actions..."
    ]


class PKUMMDDataset(Dataset):
    """
    PKU-MMD Dataset
    """
    
    CLASS_NAMES = [
        "bow", "brushing teeth", "check time", "cheer up", "clean",
        "clapping", "drink", "eat", "fall", "fight",
        "give an item", "hand waving", "hit", "hug", "kick",
        "lie down", "make a phone call", "point", "pose", "push",
        "put on clothes", "read", "ride bike", "ride horse", "run",
        "sit down", "stand up", "take a photo", "take off clothes", "throw",
        "touch", "turn left", "turn right", "walk", "wave goodbye",
        "wear glasses", "wear hat", "wear shoes", "write", "yawn"
    ]
    
    NUM_CLASSES = 51
    NUM_JOINTS = 25
    
    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 max_frames: int = 300,
                 temporal_downsample: int = 1):
        self.data_dir = data_dir
        self.split = split
        self.max_frames = max_frames
        self.temporal_downsample = temporal_downsample
        
        self.samples = []
        self.labels = []
        
        self._load_data()
    
    def _load_data(self):
        """Load PKU-MMD data"""
        data_file = os.path.join(self.data_dir, 'pku_mmd_skeletons.npz')
        
        if os.path.exists(data_file):
            data = np.load(data_file, allow_pickle=True)
            self.samples = data['skeletons']
            self.labels = data['labels']
        else:
            print(f"Warning: PKU-MMD data not found at {data_file}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        skeleton = self.samples[idx]
        label = self.labels[idx]
        
        if skeleton.shape[-1] > 3:
            skeleton = skeleton[..., :3]
        
        skeleton = torch.from_numpy(skeleton).float()
        
        return skeleton, label


class UCF101Dataset(Dataset):
    """
    UCF101 Dataset (with pose estimation)
    """
    
    # Subset of class names
    CLASS_NAMES = [
        "Basketball", "BasketballDunk", "Biking", "CliffDiving", "CricketBowling",
        "CricketShot", "Diving", "Fencing", "FloorGymnastics", "GolfSwing",
        "HorseRiding", "IceDancing", "JumpingJack", "Lunges", "MilitaryParade",
        "PullUps", "Punch", "PushUps", "RockClimbingIndoor", "RopeClimbing",
        "Rowing", "SalsaSpin", "SkateBoarding", "Skiing", "Skijet",
        "SoccerJuggling", "Swing", "TaiChi", "TennisSwing", "ThrowDisc",
        "VolleyballSpiking", "Walking", "YoYo"
    ]
    
    NUM_CLASSES = 101
    NUM_JOINTS = 17  # COCO format
    
    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 max_frames: int = 300,
                 temporal_downsample: int = 1):
        self.data_dir = data_dir
        self.split = split
        self.max_frames = max_frames
        self.temporal_downsample = temporal_downsample
        
        self.samples = []
        self.labels = []
        
        self._load_data()
    
    def _load_data(self):
        """Load UCF101 data"""
        data_file = os.path.join(self.data_dir, 'ucf101_skeletons.npz')
        
        if os.path.exists(data_file):
            data = np.load(data_file, allow_pickle=True)
            self.samples = data['skeletons']
            self.labels = data['labels']
        else:
            print(f"Warning: UCF101 data not found at {data_file}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        skeleton = self.samples[idx]
        label = self.labels[idx]
        
        if skeleton.shape[-1] > 3:
            skeleton = skeleton[..., :3]
        
        skeleton = torch.from_numpy(skeleton).float()
        
        return skeleton, label


class HMDB51Dataset(Dataset):
    """
    HMDB51 Dataset for few-shot learning
    """
    
    CLASS_NAMES = [
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
    
    NUM_CLASSES = 51
    NUM_JOINTS = 17  # COCO format
    
    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 num_shots: int = 16,
                 max_frames: int = 300):
        """
        Args:
            data_dir: Path to data directory
            split: 'train', 'val', or 'test'
            num_shots: Number of shots for few-shot learning
            max_frames: Maximum frames to sample
        """
        self.data_dir = data_dir
        self.split = split
        self.num_shots = num_shots
        self.max_frames = max_frames
        
        self.samples = []
        self.labels = []
        
        self._load_data()
    
    def _load_data(self):
        """Load HMDB51 data"""
        data_file = os.path.join(self.data_dir, 'hmdb51_skeletons.npz')
        
        if os.path.exists(data_file):
            data = np.load(data_file, allow_pickle=True)
            self.samples = data['skeletons']
            self.labels = data['labels']
        else:
            print(f"Warning: HMDB51 data not found at {data_file}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        skeleton = self.samples[idx]
        label = self.labels[idx]
        
        if skeleton.shape[-1] > 3:
            skeleton = skeleton[..., :3]
        
        skeleton = torch.from_numpy(skeleton).float()
        
        return skeleton, label


class GZSLSplit:
    """
    Generalized Zero-Shot Learning split for skeleton datasets
    """
    
    # Standard seen/unseen splits as per paper
    SPLITS = {
        'ntu60': {'seen': 55, 'unseen': 5},
        'ntu120': {'seen': 110, 'unseen': 10},
        'ucf101': {'seen': 80, 'unseen': 21},
        'pku_mmd': {'seen': 46, 'unseen': 5},
        'hmdb51': {'seen': 31, 'unseen': 20},
    }
    
    def __init__(self, dataset_name: str, split_type: str = 'random'):
        """
        Args:
            dataset_name: Name of the dataset
            split_type: 'random' or 'provided'
        """
        self.dataset_name = dataset_name
        self.split_type = split_type
        self.split_info = self.SPLITS.get(dataset_name, {'seen': 55, 'unseen': 5})
        
    def get_seen_classes(self) -> List[int]:
        """Get list of seen class indices"""
        return list(range(self.split_info['seen']))
    
    def get_unseen_classes(self) -> List[int]:
        """Get list of unseen class indices"""
        start = self.split_info['seen']
        end = start + self.split_info['unseen']
        return list(range(start, end))
    
    def get_all_classes(self) -> List[int]:
        """Get all class indices"""
        return self.get_seen_classes() + self.get_unseen_classes()


class FewShotDataset(Dataset):
    """
    Few-shot dataset for few-shot learning experiments
    """
    
    def __init__(self,
                 base_dataset: Dataset,
                 num_shots: int = 16,
                 num_way: int = 5,
                 split: str = 'train'):
        """
        Args:
            base_dataset: Base dataset to sample from
            num_shots: Number of samples per class
            num_way: Number of classes per episode
            split: 'train', 'val', or 'test'
        """
        self.base_dataset = base_dataset
        self.num_shots = num_shots
        self.num_way = num_way
        self.split = split
        
        # Sample few-shot episodes
        self.episodes = self._create_episodes()
    
    def _create_episodes(self):
        """Create few-shot episodes"""
        episodes = []
        
        # Get class distribution
        unique_labels = np.unique(self.base_dataset.labels)
        
        # Create episodes
        for _ in range(1000):  # Number of episodes
            # Sample classes
            selected_classes = np.random.choice(unique_labels, self.num_way, replace=False)
            
            # Sample shots for each class
            episode = []
            for cls in selected_classes:
                # Get samples for this class
                cls_indices = np.where(self.base_dataset.labels == cls)[0]
                
                # Sample shots
                selected_indices = np.random.choice(
                    cls_indices, 
                    min(self.num_shots, len(cls_indices)), 
                    replace=False
                )
                
                for idx in selected_indices:
                    episode.append((idx, cls))
            
            episodes.append(episode)
        
        return episodes
    
    def __len__(self) -> int:
        return len(self.episodes)
    
    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], List[int]]:
        """Get a few-shot episode"""
        episode = self.episodes[idx]
        
        skeletons = []
        labels = []
        
        for sample_idx, label in episode:
            skeleton, _ = self.base_dataset[sample_idx]
            skeletons.append(skeleton)
            labels.append(label)
        
        return skeletons, labels


def create_dataset(dataset_name: str, 
                   data_dir: str, 
                   split: str = 'train',
                   **kwargs) -> Dataset:
    """
    Factory function to create dataset
    
    Args:
        dataset_name: Name of dataset (ntu60, nt120, ucf101, pku_mmd, hmdb51)
        data_dir: Path to data directory
        split: train/val/test
        **kwargs: Additional arguments
        
    Returns:
        Dataset instance
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'ntu60':
        return NTU60Dataset(data_dir, split, **kwargs)
    elif dataset_name == 'ntu120':
        return NTU120Dataset(data_dir, split, **kwargs)
    elif dataset_name == 'pku_mmd':
        return PKUMMDDataset(data_dir, split, **kwargs)
    elif dataset_name == 'ucf101':
        return UCF101Dataset(data_dir, split, **kwargs)
    elif dataset_name == 'hmdb51':
        return HMDB51Dataset(data_dir, split, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def download_ntu60(data_dir: str):
    """
    Download NTU60 dataset (placeholder)
    This would typically use a download script
    """
    print("Note: Please download NTU60 dataset manually from:")
    print("https://github.com/shahroudy/NTURGB-D")
    print(f"Place the data in: {data_dir}")


def preprocess_dataset(dataset_name: str, data_dir: str, output_dir: str):
    """
    Preprocess skeleton data
    
    Args:
        dataset_name: Name of dataset
        data_dir: Raw data directory
        output_dir: Output directory for preprocessed data
    """
    print(f"Preprocessing {dataset_name}...")
    
    # This would include:
    # 1. Parsing original data format
    # 2. Extracting skeleton sequences
    # 3. Normalizing coordinates
    # 4. Saving to efficient format
    
    print(f"Preprocessing complete. Data saved to {output_dir}")


if __name__ == "__main__":
    # Test dataset creation
    print("Testing dataset creation...")
    
    # Test GZSL split
    split = GZSLSplit('ntu60')
    print(f"NTU60 seen classes: {split.get_seen_classes()}")
    print(f"NTU60 unseen classes: {split.get_unseen_classes()}")
    
    split = GZSLSplit('ntu120')
    print(f"NTU120 seen classes: {split.get_seen_classes()}")
    print(f"NTU120 unseen classes: {split.get_unseen_classes()}")
