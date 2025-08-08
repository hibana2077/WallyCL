import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import yaml
import os

# Import our modules
from src.models.wallycl import WallyClModel
from src.data.transforms import TRANSFORM_CONFIGS


class WallyClInference:
    """Inference and visualization for WallyCL model"""
    
    def __init__(self, config: dict, checkpoint_path: str):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = WallyClModel(
            model_name=config['model']['backbone'],
            num_classes=config['data']['num_classes'],
            embed_dim=config['model']['embed_dim'],
            hidden_dim=config['model']['hidden_dim'],
            k_tokens=config['model']['k_tokens'],
            tau_gumbel=config['model']['tau_gumbel'],
            pretrained=False
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Transform
        self.transform = TRANSFORM_CONFIGS['val'](config['data']['input_size'])
        
        print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    def predict_single_image(self, image_path: str) -> dict:
        """Predict class and get attention for a single image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Transform image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(input_tensor, return_tokens=True)
            
            # Get predictions
            logits = outputs['logits']
            probs = F.softmax(logits, dim=1)
            pred_class = torch.argmax(logits, dim=1).item()
            confidence = probs[0, pred_class].item()
            
            # Get attention map
            attention_map = self.model.get_token_attention_map(
                input_tensor, target_size=original_size
            )[0].cpu().numpy()
            
            # Get top-k predictions
            top_k_probs, top_k_indices = torch.topk(probs[0], k=min(5, probs.size(1)))
            
            return {
                'predicted_class': pred_class,
                'confidence': confidence,
                'attention_map': attention_map,
                'top_k_predictions': list(zip(top_k_indices.cpu().numpy(), 
                                            top_k_probs.cpu().numpy())),
                'original_image': image,
                'input_tensor': input_tensor
            }
    
    def visualize_attention(self, image_path: str, save_path: str = None):
        """Visualize attention map for an image"""
        result = self.predict_single_image(image_path)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(result['original_image'])
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Attention map
        axes[1].imshow(result['attention_map'], cmap='hot', interpolation='bilinear')
        axes[1].set_title('Token Attention Map')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(result['original_image'])
        overlay = axes[2].imshow(result['attention_map'], cmap='hot', alpha=0.6, 
                               interpolation='bilinear')
        axes[2].set_title(f'Predicted: Class {result["predicted_class"]} '
                         f'(Conf: {result["confidence"]:.3f})')
        axes[2].axis('off')
        
        # Add colorbar
        plt.colorbar(overlay, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()
        
        # Print predictions
        print(f"\nPrediction: Class {result['predicted_class']} (Confidence: {result['confidence']:.3f})")
        print("Top-5 predictions:")
        for i, (class_idx, prob) in enumerate(result['top_k_predictions']):
            print(f"  {i+1}. Class {class_idx}: {prob:.3f}")
    
    def compare_group_images(self, image_paths: list, save_path: str = None):
        """Compare multiple images and predict which one is odd"""
        if len(image_paths) != self.config['data']['group_size']:
            print(f"Expected {self.config['data']['group_size']} images, got {len(image_paths)}")
            return
        
        # Process all images
        images = []
        input_tensors = []
        
        for path in image_paths:
            image = Image.open(path).convert('RGB')
            images.append(image)
            
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            input_tensors.append(input_tensor)
        
        # Combine inputs
        batch_input = torch.cat(input_tensors, dim=0)
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(batch_input)
            embeddings = outputs['embeddings']
            
            # Compute inconsistency scores
            scores = self.compute_inconsistency_scores(embeddings)
            predicted_odd = torch.argmax(scores).item()
        
        # Visualize
        num_images = len(images)
        fig, axes = plt.subplots(2, num_images, figsize=(3*num_images, 6))
        if num_images == 1:
            axes = axes.reshape(2, 1)
        
        for i, (image, score) in enumerate(zip(images, scores)):
            # Original image
            axes[0, i].imshow(image)
            title = f"Image {i+1}"
            if i == predicted_odd:
                title += " (PREDICTED ODD)"
                axes[0, i].set_title(title, color='red', fontweight='bold')
            else:
                axes[0, i].set_title(title)
            axes[0, i].axis('off')
            
            # Score bar
            axes[1, i].bar(0, score.item(), color='red' if i == predicted_odd else 'blue')
            axes[1, i].set_title(f'Score: {score.item():.3f}')
            axes[1, i].set_xlim(-0.5, 0.5)
            axes[1, i].set_xticks([])
        
        plt.suptitle(f'Odd-One-Out Prediction: Image {predicted_odd + 1}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Group comparison saved to: {save_path}")
        else:
            plt.show()
        
        return predicted_odd
    
    def compute_inconsistency_scores(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute inconsistency scores for odd-one-out detection"""
        K = embeddings.size(0)
        m = K - 1  # Assume last one is the potential odd
        
        scores = []
        for k in range(K):
            distances = []
            for i in range(m):  # Compare with first m samples
                if i != k:
                    dist = 1.0 - F.cosine_similarity(embeddings[k], embeddings[i], dim=0)
                    distances.append(dist)
            
            if distances:
                avg_distance = torch.stack(distances).mean()
            else:
                avg_distance = torch.tensor(0.0, device=embeddings.device)
            
            scores.append(avg_distance)
        
        return torch.stack(scores)


def main():
    parser = argparse.ArgumentParser(description="WallyCL Inference")
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                       help='Single image path for classification')
    parser.add_argument('--group', nargs='+', default=None,
                       help='Multiple image paths for odd-one-out detection')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for visualization')
    
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file not found: {args.config}")
        return
    
    # Set a default number of classes if not specified
    if 'num_classes' not in config['data']:
        config['data']['num_classes'] = 100  # Default
    
    # Create inference engine
    inference = WallyClInference(config, args.checkpoint)
    
    if args.image:
        # Single image classification
        print(f"Analyzing single image: {args.image}")
        inference.visualize_attention(args.image, args.output)
        
    elif args.group:
        # Group odd-one-out detection
        print(f"Analyzing group of {len(args.group)} images")
        predicted_odd = inference.compare_group_images(args.group, args.output)
        print(f"Predicted odd image: {args.group[predicted_odd]}")
        
    else:
        print("Please specify either --image for single image analysis or --group for multiple images")


if __name__ == '__main__':
    main()
