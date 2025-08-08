import torch
import torch.nn as nn
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class CheckpointManager:
    """Manage model checkpoints during training"""
    
    def __init__(self, save_dir: str, max_checkpoints: int = 5):
        """
        Args:
            save_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(self,
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       epoch: int,
                       metrics: Dict[str, float],
                       is_best: bool = False,
                       extra_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a checkpoint
        
        Args:
            model: The model to save
            optimizer: The optimizer state
            scheduler: The scheduler state (optional)
            epoch: Current epoch
            metrics: Dictionary of metrics
            is_best: Whether this is the best checkpoint
            extra_data: Additional data to save
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'is_best': is_best
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if extra_data is not None:
            checkpoint.update(extra_data)
        
        # Save checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch:03d}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint: {best_path}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(self,
                       model: nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       checkpoint_path: Optional[str] = None,
                       load_best: bool = False) -> Dict[str, Any]:
        """
        Load a checkpoint
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            checkpoint_path: Specific checkpoint path (optional)
            load_best: Whether to load the best checkpoint
        
        Returns:
            Dictionary containing checkpoint metadata
        """
        if load_best:
            checkpoint_path = self.save_dir / 'best_checkpoint.pth'
        elif checkpoint_path is None:
            # Load latest checkpoint
            checkpoint_path = self._get_latest_checkpoint()
        
        if checkpoint_path is None or not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Loaded checkpoint: {checkpoint_path}")
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'is_best': checkpoint.get('is_best', False)
        }
    
    def _get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint"""
        checkpoint_files = list(self.save_dir.glob('checkpoint_epoch_*.pth'))
        
        if not checkpoint_files:
            return None
        
        # Sort by epoch number
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return str(checkpoint_files[-1])
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit"""
        checkpoint_files = list(self.save_dir.glob('checkpoint_epoch_*.pth'))
        
        if len(checkpoint_files) <= self.max_checkpoints:
            return
        
        # Sort by epoch number
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        # Remove oldest checkpoints
        for checkpoint_file in checkpoint_files[:-self.max_checkpoints]:
            checkpoint_file.unlink()
            self.logger.info(f"Removed old checkpoint: {checkpoint_file}")
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints"""
        checkpoint_files = list(self.save_dir.glob('checkpoint_epoch_*.pth'))
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return [str(f) for f in checkpoint_files]
    
    def has_checkpoints(self) -> bool:
        """Check if any checkpoints exist"""
        return len(list(self.save_dir.glob('checkpoint_epoch_*.pth'))) > 0
    
    def get_best_checkpoint_path(self) -> Optional[str]:
        """Get path to best checkpoint if it exists"""
        best_path = self.save_dir / 'best_checkpoint.pth'
        return str(best_path) if best_path.exists() else None
