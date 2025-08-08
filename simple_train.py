#!/usr/bin/env python3
"""
簡化的WallyCL訓練腳本 - 只用CE loss，按照診斷結果實現
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from pathlib import Path
import yaml
from tqdm import tqdm
import random
import wandb

# Import our modules
from src.dataset.ufgvc import UFGVCDataset
from src.models.wallycl import WallyClModel
from src.data.transforms import TRANSFORM_CONFIGS
from src.utils.metrics import WallyClMetrics


def load_config():
    """Load default config"""
    config_path = Path("configs/default.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def simple_train():
    """簡化的訓練函數 - 只用CE loss"""
    print("="*60)
    print("WallyCL 簡化訓練 - 只用CE loss (修復版)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config()
    
    # 設置隨機種子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 初始化wandb
    os.environ['WANDB_MODE'] = 'offline'
    wandb.init(
        project="wallycl-fixed",
        config=config,
        mode="offline"
    )
    
    # 創建數據集
    simple_transform = TRANSFORM_CONFIGS['val'](config['data']['input_size'])
    
    train_dataset = UFGVCDataset(
        dataset_name=config['data']['dataset_name'],
        root=config['data']['data_root'],
        split='train',
        transform=simple_transform,
        download=False
    )
    
    val_dataset = UFGVCDataset(
        dataset_name=config['data']['dataset_name'],
        root=config['data']['data_root'],
        split='val',
        transform=simple_transform,
        download=False
    )
    
    print(f"數據集: Train={len(train_dataset)}, Val={len(val_dataset)}, Classes={len(train_dataset.classes)}")
    
    # 更新配置
    config['data']['num_classes'] = len(train_dataset.classes)
    
    # 創建模型
    model = WallyClModel(
        model_name=config['model']['backbone'],
        num_classes=config['data']['num_classes'],
        embed_dim=config['model']['embed_dim'],
        hidden_dim=config['model']['hidden_dim'],
        k_tokens=config['model']['k_tokens'],
        tau_gumbel=config['model']['tau_gumbel'],
        pretrained=config['model']['pretrained']
    ).to(device)
    
    print(f"模型參數: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 設置優化器 - 分層學習率，重點訓練分類頭
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier_params.append(param)
            print(f"分類頭參數: {name}, shape: {param.shape}")
        else:
            backbone_params.append(param)
    
    # 高學習率分類頭，低學習率backbone
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config['optimizer']['lr'] * 0.01},  # backbone: 5e-6
        {'params': classifier_params, 'lr': config['optimizer']['lr'] * 20}   # classifier: 1e-2
    ], weight_decay=config['optimizer']['weight_decay'])
    
    print(f"優化器設置:")
    print(f"  Backbone學習率: {config['optimizer']['lr'] * 0.01:.2e} ({len(backbone_params)}參數)")
    print(f"  分類頭學習率: {config['optimizer']['lr'] * 20:.2e} ({len(classifier_params)}參數)")
    
    # 只用CE loss
    criterion = nn.CrossEntropyLoss()
    
    # 創建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    print(f"DataLoader: Train batches={len(train_loader)}, Val batches={len(val_loader)}")
    
    # 訓練
    print(f"\\n開始訓練 {config['training']['epochs']} epochs...")
    
    best_val_acc = 0.0
    
    for epoch in range(config['training']['epochs']):
        # 訓練階段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} Train")
        
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # 簡化的前向傳播 - 不使用token機制
            outputs = model(images, return_tokens=False)
            logits = outputs['logits']
            
            loss = criterion(logits, labels)
            loss.backward()
            
            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 統計
            train_loss += loss.item()
            _, preds = torch.max(logits, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            # 更新進度條
            train_acc = train_correct / train_total
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{train_acc:.4f}",
                'grad': f"{grad_norm:.2e}"
            })
            
            # 記錄第一個batch的詳細信息
            if i == 0:
                print(f"\\n  第一個batch詳情:")
                print(f"    Loss: {loss.item():.4f}")
                print(f"    Logits範圍: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
                print(f"    梯度範數: {grad_norm:.2e}")
                print(f"    預測樣本: {preds[:5].cpu().numpy()}")
                print(f"    真實樣本: {labels[:5].cpu().numpy()}")
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # 驗證階段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} Val")
            
            for images, labels in pbar:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images, return_tokens=False)
                logits = outputs['logits']
                
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(logits, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                # 更新進度條
                val_acc = val_correct / val_total
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{val_acc:.4f}"
                })
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        # 記錄結果
        random_acc = 1.0 / config['data']['num_classes']
        
        print(f"\\nEpoch {epoch+1} 結果:")
        print(f"  Train: Loss={avg_train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"  Val:   Loss={avg_val_loss:.4f}, Acc={val_acc:.4f}")
        print(f"  隨機準確率: {random_acc:.4f}")
        print(f"  提升倍數: {val_acc/random_acc:.1f}x")
        
        # 更新最佳結果
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  ✅ 新的最佳驗證準確率！")
        
        # wandb記錄
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'val_loss': avg_val_loss,
            'val_acc': val_acc,
            'val_acc_vs_random': val_acc / random_acc
        })
        
        # 檢查是否顯著好於隨機
        if val_acc > random_acc * 3:
            print(f"  🎉 成功！驗證準確率顯著高於隨機水平")
    
    print(f"\\n訓練完成！")
    print(f"最佳驗證準確率: {best_val_acc:.4f}")
    print(f"vs 隨機水平: {random_acc:.4f} (提升 {best_val_acc/random_acc:.1f}x)")
    
    if best_val_acc > random_acc * 2:
        print("\\n✅ 修復成功！現在可以添加其他loss組件了")
        return True
    else:
        print("\\n❌ 仍需要進一步調試")
        return False


def main():
    """主函數"""
    print("WallyCL 簡化訓練腳本")
    
    success = simple_train()
    
    if success:
        print("\\n📋 下一步計劃:")
        print("1. 逐步增加其他loss權重")
        print("2. 修復odd-one-out評估方向")
        print("3. 恢復完整的訓練流程")
    else:
        print("\\n需要進一步診斷模型或數據問題")


if __name__ == "__main__":
    main()
