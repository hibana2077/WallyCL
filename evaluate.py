import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import argparse
import yaml
import os
from tqdm import tqdm
from pathlib import Path
import json

# Import our modules
from src.dataset.ufgvc import UFGVCDataset
from src.models.wallycl import WallyClModel
from src.data.transforms import TRANSFORM_CONFIGS
from src.utils.metrics import WallyClMetrics, CalibrationMetrics
from src.utils.checkpoint import CheckpointManager


class WallyClEvaluator:
    """Evaluator for WallyCL model"""
    
    def __init__(self, config: Dict, checkpoint_path: str):
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
            pretrained=False  # We'll load trained weights
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Checkpoint metrics: {checkpoint.get('metrics', {})}")
        
        # Metrics
        self.metrics = WallyClMetrics()
    
    def setup_data(self):
        """Setup evaluation datasets"""
        config = self.config['data']
        
        # Transforms
        val_transform = TRANSFORM_CONFIGS['val'](config['input_size'])
        
        # Datasets
        splits = ['val', 'test']
        self.datasets = {}
        self.dataloaders = {}
        
        for split in splits:
            try:
                dataset = UFGVCDataset(
                    dataset_name=config['dataset_name'],
                    root=config['data_root'],
                    split=split,
                    transform=val_transform,
                    download=False
                )
                
                dataloader = DataLoader(
                    dataset,
                    batch_size=config['val_batch_size'],
                    shuffle=False,
                    num_workers=config['num_workers'],
                    pin_memory=True
                )
                
                self.datasets[split] = dataset
                self.dataloaders[split] = dataloader
                
                print(f"{split.upper()} dataset: {len(dataset)} samples")
                
            except ValueError as e:
                print(f"Warning: Could not load {split} split: {e}")
                continue
        
        # Update num_classes
        if 'val' in self.datasets:
            self.config['data']['num_classes'] = len(self.datasets['val'].classes)
        elif 'test' in self.datasets:
            self.config['data']['num_classes'] = len(self.datasets['test'].classes)
    
    def evaluate_classification(self, split: str = 'val') -> Dict[str, float]:
        """Evaluate classification performance"""
        if split not in self.dataloaders:
            print(f"Split {split} not available")
            return {}
        
        dataloader = self.dataloaders[split]
        
        all_preds = []
        all_labels = []
        all_logits = []
        
        print(f"Evaluating classification on {split} split...")
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=f"Evaluating {split}"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                logits = outputs['logits']
                
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.append(logits.cpu())
        
        # Concatenate logits
        all_logits = torch.cat(all_logits, dim=0)
        
        # Compute metrics
        metrics = self.metrics.compute_classification_metrics(all_preds, all_labels)
        
        # Top-5 accuracy
        if all_logits.size(1) >= 5:
            top5_acc = self.metrics.compute_top_k_accuracy(
                all_logits, torch.tensor(all_labels), k=5
            )
            metrics['top5_accuracy'] = top5_acc
        
        # Calibration metrics
        cal_metrics = CalibrationMetrics()
        ece = cal_metrics.compute_ece(all_logits, torch.tensor(all_labels))
        metrics['ece'] = ece
        
        return metrics
    
    def evaluate_odd_one_out(self, split: str = 'val', num_groups: int = 200) -> Dict[str, float]:
        """Evaluate odd-one-out detection performance"""
        if split not in self.datasets:
            print(f"Split {split} not available")
            return {}
        
        dataset = self.datasets[split]
        
        print(f"Evaluating odd-one-out detection on {split} split...")
        
        correct_predictions = 0
        total_groups = 0
        
        # Group configuration
        m = self.config['data']['m']
        group_size = self.config['data']['group_size']
        
        with torch.no_grad():
            for _ in tqdm(range(num_groups), desc="Odd-one-out evaluation"):
                # Sample a group
                group_indices = self.sample_evaluation_group(dataset, m)
                if len(group_indices) != group_size:
                    continue
                
                # Load images
                group_images = []
                for idx in group_indices:
                    img, _ = dataset[idx]
                    group_images.append(img)
                
                group_images = torch.stack(group_images).to(self.device)
                
                # Forward pass
                outputs = self.model(group_images)
                embeddings = outputs['embeddings']
                
                # Compute inconsistency scores
                scores = self.compute_inconsistency_scores(embeddings, m)
                pred_odd = torch.argmax(scores)
                
                # True odd is the last sample
                true_odd = group_size - 1
                
                if pred_odd.item() == true_odd:
                    correct_predictions += 1
                total_groups += 1
        
        odd_accuracy = correct_predictions / max(total_groups, 1)
        
        return {
            'odd_accuracy': odd_accuracy,
            'total_groups_evaluated': total_groups
        }
    
    def sample_evaluation_group(self, dataset, m: int) -> List[int]:
        """Sample a group for odd-one-out evaluation"""
        import random
        
        # Group by class
        class_to_indices = {}
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)
        
        # Filter classes with enough samples
        valid_classes = [cls for cls, indices in class_to_indices.items() 
                        if len(indices) >= m]
        
        if len(valid_classes) < 2:
            return []
        
        # Sample positive class
        pos_class = random.choice(valid_classes)
        pos_indices = random.sample(class_to_indices[pos_class], m)
        
        # Sample odd class
        odd_classes = [cls for cls in valid_classes if cls != pos_class]
        odd_class = random.choice(odd_classes)
        odd_index = random.choice(class_to_indices[odd_class])
        
        return pos_indices + [odd_index]
    
    def compute_inconsistency_scores(self, embeddings: torch.Tensor, m: int) -> torch.Tensor:
        """Compute inconsistency scores for odd-one-out detection"""
        K = embeddings.size(0)
        
        scores = []
        for k in range(K):
            distances = []
            for i in range(m):  # Compare with positives only
                if i != k:
                    dist = 1.0 - torch.cosine_similarity(embeddings[k], embeddings[i], dim=0)
                    distances.append(dist)
            
            if distances:
                avg_distance = torch.stack(distances).mean()
            else:
                avg_distance = torch.tensor(0.0, device=embeddings.device)
            
            scores.append(avg_distance)
        
        return torch.stack(scores)
    
    def evaluate_token_attribution(self, split: str = 'val', num_samples: int = 50) -> Dict[str, float]:
        """Evaluate token attribution quality"""
        if split not in self.datasets:
            print(f"Split {split} not available")
            return {}
        
        dataset = self.datasets[split]
        
        print(f"Evaluating token attribution on {split} split...")
        
        attribution_scores = []
        
        with torch.no_grad():
            for i in tqdm(range(min(num_samples, len(dataset))), desc="Token attribution"):
                img, label = dataset[i]
                img = img.unsqueeze(0).to(self.device)
                
                # Get token weights
                outputs = self.model(img, return_tokens=True)
                token_weights = outputs['token_weights']  # [1, num_tokens]
                
                # Compute sparsity (how concentrated the selection is)
                weights = token_weights.squeeze(0)
                sparsity = torch.sum(weights > 0.1).float() / weights.size(0)
                attribution_scores.append(sparsity.item())
        
        return {
            'mean_sparsity': np.mean(attribution_scores),
            'std_sparsity': np.std(attribution_scores),
            'num_samples_evaluated': len(attribution_scores)
        }
    
    def run_full_evaluation(self) -> Dict[str, Dict[str, float]]:
        """Run complete evaluation suite"""
        print("Setting up data...")
        self.setup_data()
        
        results = {}
        
        # Evaluate on available splits
        for split in self.dataloaders.keys():
            print(f"\n=== Evaluating {split.upper()} split ===")
            
            # Classification metrics
            cls_metrics = self.evaluate_classification(split)
            
            # Odd-one-out metrics
            odd_metrics = self.evaluate_odd_one_out(split)
            
            # Token attribution metrics
            attr_metrics = self.evaluate_token_attribution(split)
            
            # Combine results
            results[split] = {
                **cls_metrics,
                **odd_metrics,
                **attr_metrics
            }
        
        return results
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_path}")
    
    def print_results(self, results: Dict):
        """Print formatted results"""
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        for split, metrics in results.items():
            print(f"\n{split.upper()} Split:")
            print("-" * 20)
            
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"{metric:20s}: {value:.4f}")
                else:
                    print(f"{metric:20s}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate WallyCL model")
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name (override config)')
    parser.add_argument('--output', type=str, default='./results/evaluation_results.json',
                       help='Output path for results')
    parser.add_argument('--split', type=str, default='all',
                       help='Split to evaluate: val, test, or all')
    
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file not found: {args.config}")
        return
    
    # Override dataset if specified
    if args.dataset is not None:
        config['data']['dataset_name'] = args.dataset
    
    # Create evaluator
    evaluator = WallyClEvaluator(config, args.checkpoint)
    
    # Run evaluation
    if args.split == 'all':
        results = evaluator.run_full_evaluation()
    else:
        evaluator.setup_data()
        if args.split in ['val', 'test']:
            cls_metrics = evaluator.evaluate_classification(args.split)
            odd_metrics = evaluator.evaluate_odd_one_out(args.split)
            attr_metrics = evaluator.evaluate_token_attribution(args.split)
            results = {args.split: {**cls_metrics, **odd_metrics, **attr_metrics}}
        else:
            print(f"Invalid split: {args.split}")
            return
    
    # Print and save results
    evaluator.print_results(results)
    evaluator.save_results(results, args.output)


if __name__ == '__main__':
    main()
