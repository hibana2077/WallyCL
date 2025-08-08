import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from torch.utils.data import DataLoader, Sampler
import random


class GroupSampler(Sampler):
    """Sampler for creating groups with m positives + 1 odd for WallyCL training"""
    
    def __init__(self, 
                 dataset,
                 group_size: int = 5,  # m + 1
                 groups_per_batch: int = 4,
                 shuffle: bool = True):
        """
        Args:
            dataset: Dataset with .data attribute containing labels
            group_size: K = m + 1 (m positives + 1 odd)
            groups_per_batch: Number of groups per batch
            shuffle: Whether to shuffle the data
        """
        self.dataset = dataset
        self.group_size = group_size
        self.m = group_size - 1  # Number of positives
        self.groups_per_batch = groups_per_batch
        self.shuffle = shuffle
        
        # Group samples by class
        self.class_to_indices = {}
        for idx, row in dataset.data.iterrows():
            class_name = row['class_name']
            if class_name not in self.class_to_indices:
                self.class_to_indices[class_name] = []
            self.class_to_indices[class_name].append(idx)
        
        self.classes = list(self.class_to_indices.keys())
        
        # Filter classes with enough samples for positive groups
        self.valid_classes = [cls for cls in self.classes 
                            if len(self.class_to_indices[cls]) >= self.m]
        
        if len(self.valid_classes) < 2:
            raise ValueError(f"Need at least 2 classes with {self.m}+ samples each")
        
        # Estimate number of possible groups
        self.groups_per_epoch = min(
            len(self.valid_classes) * 10,  # Conservative estimate
            sum(len(indices) // self.m for indices in self.class_to_indices.values()) * 2
        )
    
    def _create_group(self) -> List[int]:
        """Create a single group: m positives + 1 odd"""
        # Select positive class
        pos_class = random.choice(self.valid_classes)
        pos_indices = self.class_to_indices[pos_class].copy()
        
        if len(pos_indices) < self.m:
            # Fallback: sample with replacement if needed
            pos_samples = random.choices(pos_indices, k=self.m)
        else:
            pos_samples = random.sample(pos_indices, k=self.m)
        
        # Select odd class (different from positive class)
        odd_classes = [cls for cls in self.valid_classes if cls != pos_class]
        if not odd_classes:
            # Fallback: use different samples from same class if only one class
            odd_class = pos_class
            available_odd = [idx for idx in pos_indices if idx not in pos_samples]
            if available_odd:
                odd_sample = random.choice(available_odd)
            else:
                odd_sample = random.choice(pos_indices)  # Last resort
        else:
            odd_class = random.choice(odd_classes)
            odd_indices = self.class_to_indices[odd_class]
            odd_sample = random.choice(odd_indices)
        
        return pos_samples + [odd_sample]
    
    def __iter__(self):
        groups = []
        
        # Generate groups for the epoch
        num_batches = max(1, self.groups_per_epoch // self.groups_per_batch)
        
        for _ in range(num_batches * self.groups_per_batch):
            group = self._create_group()
            groups.extend(group)
        
        if self.shuffle:
            # Shuffle groups while maintaining group structure
            group_indices = list(range(0, len(groups), self.group_size))
            random.shuffle(group_indices)
            
            shuffled_groups = []
            for start_idx in group_indices:
                end_idx = min(start_idx + self.group_size, len(groups))
                shuffled_groups.extend(groups[start_idx:end_idx])
            groups = shuffled_groups
        
        return iter(groups)
    
    def __len__(self):
        num_batches = max(1, self.groups_per_epoch // self.groups_per_batch)
        return num_batches * self.groups_per_batch * self.group_size


def build_groups_from_batch(batch_indices: List[int], 
                           batch_labels: List[int], 
                           m: int = 4) -> List[List[int]]:
    """
    Build groups from a regular batch (fallback method)
    Args:
        batch_indices: List of sample indices in the batch
        batch_labels: List of corresponding labels
        m: Number of positives per group
    Returns:
        List of groups, each group is a list of indices
    """
    # Group by label
    label_to_indices = {}
    for idx, label in zip(batch_indices, batch_labels):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)
    
    groups = []
    available_labels = list(label_to_indices.keys())
    
    for pos_label in available_labels:
        pos_indices = label_to_indices[pos_label]
        
        if len(pos_indices) >= m:
            # Select m positives
            pos_samples = pos_indices[:m]
            
            # Select odd from different class
            odd_labels = [l for l in available_labels if l != pos_label]
            if odd_labels:
                odd_label = random.choice(odd_labels)
                odd_indices = label_to_indices[odd_label]
                odd_sample = random.choice(odd_indices)
                
                group = pos_samples + [odd_sample]
                groups.append(group)
    
    return groups


def collate_groups(batch: List[Tuple[torch.Tensor, int]]) -> Dict[str, Any]:
    """
    Custom collate function for group-based training
    Organizes batch into groups of (m positives + 1 odd)
    """
    # Standard collation first
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    
    batch_size = len(batch)
    group_size = 5  # Default K = 5 (4 positives + 1 odd)
    
    # Organize into groups
    groups = []
    group_labels = []
    
    for i in range(0, batch_size, group_size):
        end_idx = min(i + group_size, batch_size)
        if end_idx - i == group_size:  # Only complete groups
            group_images = images[i:end_idx]
            group_lbls = labels[i:end_idx]
            
            groups.append(group_images)
            group_labels.append(group_lbls)
    
    return {
        'images': images,
        'labels': labels,
        'groups': groups,
        'group_labels': group_labels,
        'batch_size': batch_size,
        'num_groups': len(groups)
    }


def create_group_dataloader(dataset,
                           group_size: int = 5,
                           groups_per_batch: int = 4,
                           num_workers: int = 4,
                           shuffle: bool = True) -> DataLoader:
    """
    Create a DataLoader with group-based sampling for WallyCL
    """
    sampler = GroupSampler(
        dataset=dataset,
        group_size=group_size,
        groups_per_batch=groups_per_batch,
        shuffle=shuffle
    )
    
    return DataLoader(
        dataset,
        batch_sampler=None,
        sampler=sampler,
        batch_size=group_size * groups_per_batch,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_groups
    )


class GroupBatcher:
    """Helper class to manage group construction and batching"""
    
    def __init__(self, m: int = 4, group_size: int = 5):
        self.m = m  # Number of positives
        self.K = group_size  # Total group size (m + 1)
    
    def organize_batch_into_groups(self, 
                                  images: torch.Tensor, 
                                  labels: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Organize a batch into groups for WallyCL training
        Args:
            images: [batch_size, C, H, W]
            labels: [batch_size]
        Returns:
            group_images: List of [K, C, H, W] tensors
            group_labels: List of [K] tensors
        """
        batch_size = images.size(0)
        
        group_images = []
        group_labels = []
        
        # Simple grouping: assume batch is already organized by the sampler
        for i in range(0, batch_size, self.K):
            end_idx = min(i + self.K, batch_size)
            if end_idx - i == self.K:  # Only complete groups
                group_img = images[i:end_idx]
                group_lbl = labels[i:end_idx]
                
                group_images.append(group_img)
                group_labels.append(group_lbl)
        
        return group_images, group_labels
    
    def get_positive_aggregate(self, group_tokens: torch.Tensor) -> torch.Tensor:
        """
        Aggregate positive samples' tokens in a group
        Args:
            group_tokens: [K, embed_dim] group token embeddings
        Returns:
            pos_aggregate: [embed_dim] averaged positive tokens
        """
        # Positives are first m samples
        pos_tokens = group_tokens[:self.m]  # [m, embed_dim]
        return pos_tokens.mean(dim=0)  # [embed_dim]
    
    def sample_negative_tokens(self, 
                             all_token_embeddings: List[torch.Tensor],
                             exclude_group_idx: int,
                             num_negatives: int = 3) -> List[torch.Tensor]:
        """
        Sample negative token embeddings from other groups
        Args:
            all_token_embeddings: List of token embeddings from all groups
            exclude_group_idx: Index of current group to exclude
            num_negatives: Number of negative samples
        Returns:
            negative_tokens: List of negative token embeddings
        """
        available_groups = [i for i in range(len(all_token_embeddings)) 
                          if i != exclude_group_idx]
        
        if len(available_groups) == 0:
            return []
        
        selected_groups = random.choices(available_groups, 
                                       k=min(num_negatives, len(available_groups)))
        
        negative_tokens = []
        for group_idx in selected_groups:
            # Use the odd sample from the selected group as negative
            group_tokens = all_token_embeddings[group_idx]
            if len(group_tokens) >= self.K:
                odd_token = group_tokens[-1]  # Last is odd
                negative_tokens.append(odd_token)
        
        return negative_tokens
