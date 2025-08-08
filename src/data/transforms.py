from torchvision import transforms
import torch


def get_train_transforms(input_size: int = 224,
                        augmentation_strength: str = 'medium') -> transforms.Compose:
    """
    Get training transforms for WallyCL
    Args:
        input_size: Input image size
        augmentation_strength: 'weak', 'medium', or 'strong'
    """
    
    if augmentation_strength == 'weak':
        transform_list = [
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    elif augmentation_strength == 'medium':
        transform_list = [
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    elif augmentation_strength == 'strong':
        transform_list = [
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1)),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    else:
        raise ValueError(f"Unknown augmentation strength: {augmentation_strength}")
    
    return transforms.Compose(transform_list)


def get_val_transforms(input_size: int = 224) -> transforms.Compose:
    """Get validation/test transforms"""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_structure_aware_transforms(input_size: int = 224, 
                                 mask_ratio: float = 0.1,
                                 shuffle_ratio: float = 0.1) -> transforms.Compose:
    """
    Get structure-aware transforms similar to CLE-ViT
    These are applied consistently across positives in a group
    """
    
    class StructureAwareAugment:
        def __init__(self, mask_ratio: float, shuffle_ratio: float):
            self.mask_ratio = mask_ratio
            self.shuffle_ratio = shuffle_ratio
        
        def __call__(self, img):
            # For now, implement basic random erasing as structure-aware augmentation
            # In a full implementation, this would include patch-level operations
            if torch.rand(1) < self.mask_ratio:
                # Random erasing with small patches
                img = transforms.RandomErasing(p=1.0, scale=(0.01, 0.05), ratio=(0.3, 3.3))(img)
            
            return img
    
    return transforms.Compose([
        transforms.Resize((input_size + 32, input_size + 32)),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        StructureAwareAugment(mask_ratio, shuffle_ratio),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# Define standard configurations
TRANSFORM_CONFIGS = {
    'train_weak': lambda size=224: get_train_transforms(size, 'weak'),
    'train_medium': lambda size=224: get_train_transforms(size, 'medium'),
    'train_strong': lambda size=224: get_train_transforms(size, 'strong'),
    'val': lambda size=224: get_val_transforms(size),
    'structure_aware': lambda size=224: get_structure_aware_transforms(size)
}
