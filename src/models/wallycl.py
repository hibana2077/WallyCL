import torch
import torch.nn as nn
import timm
from typing import Tuple, Optional, List
from .token_utils import GumbelTopK, TokenScorer, ProjectionHead, compute_token_average


class WallyClModel(nn.Module):
    """WallyCL model with token-based backbone and group contrastive learning"""
    
    def __init__(self,
                 model_name: str = 'vit_base_patch16_224',
                 num_classes: int = 1000,
                 embed_dim: int = 128,
                 hidden_dim: int = 512,
                 k_tokens: int = 8,
                 tau_gumbel: float = 1.0,
                 pretrained: bool = True):
        super().__init__()
        
        self.k_tokens = k_tokens
        self.embed_dim = embed_dim
        
        # Load backbone (ViT or Swin)
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling to get tokens
        )
        
        # Get backbone feature dimensions
        if hasattr(self.backbone, 'embed_dim'):
            self.backbone_dim = self.backbone.embed_dim
        elif hasattr(self.backbone, 'num_features'):
            self.backbone_dim = self.backbone.num_features
        else:
            # Fallback: run a dummy forward to get dimensions
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                dummy_output = self.backbone(dummy_input)
                if len(dummy_output.shape) == 3:  # [1, num_tokens, dim]
                    self.backbone_dim = dummy_output.shape[-1]
                else:
                    self.backbone_dim = dummy_output.shape[-1]
        
        # Token-level components
        self.token_scorer = TokenScorer(self.backbone_dim)
        self.gumbel_topk = GumbelTopK(k=k_tokens, tau=tau_gumbel)
        
        # Projection heads
        self.projection_head = ProjectionHead(self.backbone_dim, hidden_dim, embed_dim)
        self.token_projection_head = ProjectionHead(self.backbone_dim, hidden_dim, embed_dim)
        
        # Classification head
        self.classifier = nn.Linear(self.backbone_dim, num_classes)
        
        # Global pooling for image-level features
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    
    def extract_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract token embeddings and pooled features from backbone
        Args:
            x: [batch_size, 3, H, W] input images
        Returns:
            tokens: [batch_size, num_tokens, backbone_dim] token embeddings
            pooled: [batch_size, backbone_dim] globally pooled features
        """
        # Forward through backbone
        features = self.backbone(x)
        
        if len(features.shape) == 3:  # [batch_size, num_tokens, dim]
            tokens = features
            # Global average pooling over tokens (excluding CLS if present)
            if hasattr(self.backbone, 'cls_token') and self.backbone.cls_token is not None:
                cls_token = tokens[:, 0:1, :]  # CLS token
                patch_tokens = tokens[:, 1:, :]  # Patch tokens
                pooled = patch_tokens.mean(dim=1)  # Average patch tokens
            else:
                pooled = tokens.mean(dim=1)  # Average all tokens
        else:  # Already pooled features
            batch_size = features.shape[0]
            pooled = features
            # Create dummy tokens (shouldn't happen with proper ViT/Swin)
            tokens = pooled.unsqueeze(1).expand(-1, 196, -1)  # Assume 14x14 patches
        
        return tokens, pooled
    
    def select_tokens(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k tokens using Gumbel-Top-K
        Args:
            tokens: [batch_size, num_tokens, backbone_dim]
        Returns:
            selected_tokens: [batch_size, backbone_dim] averaged selected tokens
            weights: [batch_size, num_tokens] selection weights
        """
        batch_size, num_tokens, _ = tokens.shape
        
        # Score tokens
        scores = self.token_scorer(tokens)  # [batch_size, num_tokens]
        
        # Select top-k tokens with Gumbel
        weights = self.gumbel_topk(scores)  # [batch_size, num_tokens]
        
        # Compute weighted average
        selected_tokens = compute_token_average(tokens, weights)
        
        return selected_tokens, weights
    
    def forward(self, x: torch.Tensor, return_tokens: bool = False) -> dict:
        """
        Forward pass through WallyCL model
        Args:
            x: [batch_size, 3, H, W] input images
            return_tokens: whether to return token-level information
        Returns:
            dict with embeddings, logits, and optionally token info
        """
        batch_size = x.size(0)
        
        # Extract tokens and pooled features
        tokens, pooled = self.extract_tokens(x)
        
        # Image-level embeddings (L2 normalized)
        embeddings = self.projection_head(pooled)
        
        # Classification logits
        logits = self.classifier(pooled)
        
        outputs = {
            'embeddings': embeddings,
            'logits': logits,
            'pooled_features': pooled
        }
        
        if return_tokens:
            # Token selection and projection
            selected_tokens, token_weights = self.select_tokens(tokens)
            token_embeddings = self.token_projection_head(selected_tokens)
            
            outputs.update({
                'tokens': tokens,
                'selected_tokens': selected_tokens,
                'token_embeddings': token_embeddings,
                'token_weights': token_weights
            })
        
        return outputs
    
    def get_token_attention_map(self, x: torch.Tensor, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Get attention map showing which tokens were selected
        Args:
            x: [batch_size, 3, H, W] input images
            target_size: optional (H, W) to resize attention map
        Returns:
            attention_maps: [batch_size, H, W] attention maps
        """
        with torch.no_grad():
            tokens, _ = self.extract_tokens(x)
            _, token_weights = self.select_tokens(tokens)
            
            batch_size = x.size(0)
            
            # Reshape token weights to spatial dimensions
            # Assuming ViT with patch size 16 and 224x224 input -> 14x14 patches
            if hasattr(self.backbone, 'patch_embed'):
                patch_size = self.backbone.patch_embed.patch_size[0]
                input_size = x.shape[-1]  # Assume square input
                grid_size = input_size // patch_size
                
                # Skip CLS token if present
                if hasattr(self.backbone, 'cls_token') and self.backbone.cls_token is not None:
                    spatial_weights = token_weights[:, 1:]  # Skip CLS token
                else:
                    spatial_weights = token_weights
                
                # Reshape to spatial grid
                attention_maps = spatial_weights.view(batch_size, grid_size, grid_size)
                
                # Interpolate to target size if provided
                if target_size is not None:
                    attention_maps = torch.nn.functional.interpolate(
                        attention_maps.unsqueeze(1),  # Add channel dim
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)  # Remove channel dim
                
                return attention_maps
            else:
                # Fallback for non-patch models
                return token_weights


def create_wallycl_model(model_name: str = 'vit_base_patch16_224',
                        num_classes: int = 1000,
                        **kwargs) -> WallyClModel:
    """
    Factory function to create WallyCL model with different backbones
    """
    return WallyClModel(
        model_name=model_name,
        num_classes=num_classes,
        **kwargs
    )
