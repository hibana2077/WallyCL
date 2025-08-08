import torch
import torch.nn as nn
from typing import Optional
import copy


class EMA:
    """Exponential Moving Average for model parameters"""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Args:
            model: The model to track
            decay: EMA decay factor
        """
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval()
        
        # Disable gradients for EMA model
        for param in self.model.parameters():
            param.requires_grad_(False)
    
    def update(self, model: Optional[nn.Module] = None):
        """Update EMA parameters"""
        if model is None:
            return
        
        with torch.no_grad():
            for ema_param, model_param in zip(self.model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)
    
    def set_decay(self, decay: float):
        """Update decay factor"""
        self.decay = decay
    
    def state_dict(self):
        """Get EMA model state dict"""
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load EMA model state dict"""
        self.model.load_state_dict(state_dict)
