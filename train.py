import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import wandb
import os
from tqdm import tqdm
import argparse
from pathlib import Path
import yaml

# Import our modules
from src.dataset.ufgvc import UFGVCDataset
from src.models.wallycl import WallyClModel
from src.models.losses import WallyClLoss
from src.data.group_sampler import create_group_dataloader, GroupBatcher
from src.data.transforms import TRANSFORM_CONFIGS
from src.utils.metrics import WallyClMetrics
from src.utils.checkpoint import CheckpointManager
from src.utils.ema import EMA


class WallyClTrainer:
    """Main trainer class for WallyCL"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize wandb in offline mode
        os.environ['WANDB_MODE'] = 'offline'
        wandb.init(
            project="wallycl",
            config=config,
            mode="offline"
        )
        
        # Set up model
        self.model = WallyClModel(
            model_name=config['model']['backbone'],
            num_classes=config['data']['num_classes'],
            embed_dim=config['model']['embed_dim'],
            hidden_dim=config['model']['hidden_dim'],
            k_tokens=config['model']['k_tokens'],
            tau_gumbel=config['model']['tau_gumbel'],
            pretrained=config['model']['pretrained']
        ).to(self.device)
        
        # Set up loss function
        self.criterion = WallyClLoss(
            lambda_sup=config['loss']['lambda_sup'],
            lambda_tok=config['loss']['lambda_tok'],
            lambda_ce=config['loss']['lambda_ce'],
            tau_sup=config['loss']['tau_sup'],
            tau_tok=config['loss']['tau_tok'],
            alpha=config['loss']['alpha'],
            distance_type=config['loss']['distance_type']
        )
        
        # Set up optimizer with different learning rates
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'classifier' in name:
                head_params.append(param)
                print(f"Classifier param: {name}, shape: {param.shape}")
            else:
                backbone_params.append(param)
                # 可選：凍結backbone進行快速測試
                # param.requires_grad = False
        
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': config['optimizer']['lr'] * 0.01},  # backbone用很小學習率
            {'params': head_params, 'lr': config['optimizer']['lr'] * 20}  # 分類頭用很大學習率
        ], weight_decay=config['optimizer']['weight_decay'])
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Optimizer setup: backbone params={len(backbone_params)}, head params={len(head_params)}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Set up scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['optimizer']['lr'] * 0.2
        )
        
        # EMA
        if config['training']['use_ema']:
            self.ema = EMA(self.model, decay=config['training']['ema_decay'])
        else:
            self.ema = None
        
        # Group batcher
        self.group_batcher = GroupBatcher(
            m=config['data']['m'],
            group_size=config['data']['group_size']
        )
        
        # Metrics and checkpoint manager
        self.metrics = WallyClMetrics()
        self.checkpoint_manager = CheckpointManager(
            save_dir=config['training']['save_dir'],
            max_checkpoints=config['training']['max_checkpoints']
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_odd_acc = 0.0
    
    def setup_data(self):
        """Setup datasets and dataloaders"""
        config = self.config['data']
        
        # Transforms
        train_transform = TRANSFORM_CONFIGS[config['train_transform']](config['input_size'])
        val_transform = TRANSFORM_CONFIGS['val'](config['input_size'])
        
        # Datasets
        self.train_dataset = UFGVCDataset(
            dataset_name=config['dataset_name'],
            root=config['data_root'],
            split='train',
            transform=train_transform,
            download=True
        )
        
        self.val_dataset = UFGVCDataset(
            dataset_name=config['dataset_name'],
            root=config['data_root'],
            split='val',
            transform=val_transform,
            download=False
        )
        
        # Update num_classes in config
        self.config['data']['num_classes'] = len(self.train_dataset.classes)
        
        # Dataloaders
        self.train_loader = create_group_dataloader(
            self.train_dataset,
            group_size=config['group_size'],
            groups_per_batch=config['groups_per_batch'],
            num_workers=config['num_workers'],
            shuffle=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['val_batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        print(f"Train dataset: {len(self.train_dataset)} samples, {len(self.train_dataset.classes)} classes")
        print(f"Val dataset: {len(self.val_dataset)} samples")
        print(f"Train batches per epoch: {len(self.train_loader)}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {'total': 0, 'odd': 0, 'supcon': 0, 'tokcon': 0, 'ce': 0}
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress_bar:
            # Extract batch data
            images = batch['images'].to(self.device)
            labels = batch['labels'].to(self.device)
            groups = batch['groups']
            group_labels = batch['group_labels']
            
            if not groups:  # Skip if no complete groups
                continue
            
            self.optimizer.zero_grad()
            
            # Forward pass for all images
            outputs = self.model(images, return_tokens=True)
            embeddings = outputs['embeddings']
            logits = outputs['logits']
            token_embeddings = outputs['token_embeddings']
            
            # Organize into groups
            group_embeddings = []
            group_label_tensors = []
            token_contrastive_pairs = []
            
            start_idx = 0
            for i, group in enumerate(groups):
                group_size = group.size(0)
                end_idx = start_idx + group_size
                
                # Group embeddings and labels
                group_emb = embeddings[start_idx:end_idx]
                group_lbl = group_labels[i].to(self.device)
                
                group_embeddings.append(group_emb)
                group_label_tensors.append(group_lbl)
                
                # Token contrastive pairs
                group_token_emb = token_embeddings[start_idx:end_idx]
                odd_tokens = group_token_emb[-1]  # Last is odd
                pos_tokens = self.group_batcher.get_positive_aggregate(group_token_emb)
                
                # Sample negative tokens from other groups
                all_group_tokens = [token_embeddings[j*self.config['data']['group_size']:(j+1)*self.config['data']['group_size']] 
                                  for j in range(len(groups))]
                neg_tokens = self.group_batcher.sample_negative_tokens(
                    all_group_tokens, exclude_group_idx=i
                )
                
                token_contrastive_pairs.append((odd_tokens, pos_tokens, neg_tokens))
                
                start_idx = end_idx
            
            # Compute loss
            total_loss, loss_dict = self.criterion(
                group_embeddings=group_embeddings,
                group_labels=group_label_tensors,
                cls_logits=logits,
                cls_labels=labels,
                token_contrastive_pairs=token_contrastive_pairs
            )
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update EMA
            if self.ema is not None:
                self.ema.update()
            
            # Accumulate losses
            for key, value in loss_dict.items():
                epoch_losses[key] += value
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'odd': f"{loss_dict['odd']:.4f}",
                'supcon': f"{loss_dict['supcon']:.4f}",
                'ce': f"{loss_dict['ce']:.4f}"
            })
        
        # Average losses
        if num_batches > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        model_to_eval = self.ema.model if self.ema is not None else self.model
        model_to_eval.eval()
        
        all_preds = []
        all_labels = []
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 簡化的前向傳播，避免復雜的token處理
                outputs = model_to_eval(images, return_tokens=False)
                logits = outputs['logits']
                
                # Classification loss
                loss = nn.CrossEntropyLoss()(logits, labels)
                val_loss += loss.item()
                num_batches += 1
                
                # Predictions
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        metrics = self.metrics.compute_classification_metrics(all_preds, all_labels)
        metrics['val_loss'] = val_loss / max(num_batches, 1)
        
        return metrics
    
    def validate_odd_one_out(self) -> float:
        """Validate odd-one-out task on validation set"""
        model_to_eval = self.ema.model if self.ema is not None else self.model
        model_to_eval.eval()
        
        # Create pseudo-groups from validation set
        correct_odd_predictions = 0
        total_groups = 0
        
        with torch.no_grad():
            # Sample groups from validation set
            num_val_groups = min(100, len(self.val_dataset) // self.config['data']['group_size'])
            
            for _ in range(num_val_groups):
                # Sample m positives + 1 odd
                group_indices = self.sample_validation_group()
                if len(group_indices) != self.config['data']['group_size']:
                    continue
                
                group_images = []
                for idx in group_indices:
                    img, _ = self.val_dataset[idx]
                    group_images.append(img)
                
                group_images = torch.stack(group_images).to(self.device)
                
                # Forward pass
                outputs = model_to_eval(group_images)
                embeddings = outputs['embeddings']
                
                # Compute inconsistency scores
                scores = self.compute_inconsistency_scores(embeddings)
                pred_odd = torch.argmax(scores)
                
                # True odd is the last one (index -1)
                true_odd = len(group_indices) - 1
                
                if pred_odd.item() == true_odd:
                    correct_odd_predictions += 1
                total_groups += 1
        
        odd_accuracy = correct_odd_predictions / max(total_groups, 1)
        return odd_accuracy
    
    def sample_validation_group(self) -> List[int]:
        """Sample a group from validation set"""
        import random
        
        # Get class distribution
        class_to_indices = {}
        for idx in range(len(self.val_dataset)):
            _, label = self.val_dataset[idx]
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)
        
        # Sample positive class with enough samples
        valid_classes = [cls for cls, indices in class_to_indices.items() 
                        if len(indices) >= self.config['data']['m']]
        
        if len(valid_classes) < 2:
            return []
        
        pos_class = random.choice(valid_classes)
        pos_indices = random.sample(class_to_indices[pos_class], self.config['data']['m'])
        
        # Sample odd class
        odd_classes = [cls for cls in valid_classes if cls != pos_class]
        odd_class = random.choice(odd_classes)
        odd_index = random.choice(class_to_indices[odd_class])
        
        return pos_indices + [odd_index]
    
    def compute_inconsistency_scores(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute inconsistency scores for odd-one-out detection"""
        K = embeddings.size(0)
        m = K - 1
        
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
    
    def train(self):
        """Main training loop"""
        print("Setting up data...")
        self.setup_data()
        
        print(f"Starting training for {self.config['training']['epochs']} epochs")
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Odd-one-out validation (every 5 epochs to save time)
            if epoch % 5 == 0:
                odd_acc = self.validate_odd_one_out()
            else:
                odd_acc = 0.0
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            wandb.log({
                'epoch': epoch,
                'train/loss_total': train_losses['total'],
                'train/loss_odd': train_losses['odd'],
                'train/loss_supcon': train_losses['supcon'],
                'train/loss_tokcon': train_losses['tokcon'],
                'train/loss_ce': train_losses['ce'],
                'val/accuracy': val_metrics['accuracy'],
                'val/f1_macro': val_metrics['f1_macro'],
                'val/loss': val_metrics['val_loss'],
                'val/odd_accuracy': odd_acc,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Print epoch summary
            print(f"Epoch {epoch}: "
                  f"Train Loss: {train_losses['total']:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}, "
                  f"Odd Acc: {odd_acc:.4f}")
            
            # Save checkpoints
            is_best_val = val_metrics['accuracy'] > self.best_val_acc
            is_best_odd = odd_acc > self.best_odd_acc
            
            if is_best_val:
                self.best_val_acc = val_metrics['accuracy']
            if is_best_odd:
                self.best_odd_acc = odd_acc
            
            self.checkpoint_manager.save_checkpoint(
                model=self.ema.model if self.ema else self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metrics={
                    'val_acc': val_metrics['accuracy'],
                    'odd_acc': odd_acc,
                    'train_loss': train_losses['total']
                },
                is_best=is_best_val or is_best_odd
            )
        
        print(f"Training completed. Best val acc: {self.best_val_acc:.4f}, Best odd acc: {self.best_odd_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train WallyCL model")
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--dataset', type=str, default='cotton80',
                       help='Dataset name')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        # Default config
        config = {
            'model': {
                'backbone': 'vit_base_patch16_224',
                'embed_dim': 128,
                'hidden_dim': 512,
                'k_tokens': 8,
                'tau_gumbel': 1.0,
                'pretrained': True
            },
            'data': {
                'dataset_name': args.dataset,
                'data_root': './data',
                'input_size': 224,
                'group_size': 5,
                'm': 4,
                'groups_per_batch': 4,
                'val_batch_size': 32,
                'num_workers': 4,
                'train_transform': 'train_medium'
            },
            'loss': {
                'lambda_sup': 1.0,
                'lambda_tok': 0.5,
                'lambda_ce': 1.0,
                'tau_sup': 0.1,
                'tau_tok': 0.07,
                'alpha': 5.0,
                'distance_type': 'cosine'
            },
            'optimizer': {
                'lr': 2e-4,
                'weight_decay': 0.05
            },
            'training': {
                'epochs': 100,
                'use_ema': True,
                'ema_decay': 0.999,
                'save_dir': './checkpoints',
                'max_checkpoints': 5
            }
        }
    
    # Override dataset if specified
    config['data']['dataset_name'] = args.dataset
    
    # Create trainer and start training
    trainer = WallyClTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
