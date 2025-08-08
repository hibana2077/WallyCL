"""
Quick demo script for WallyCL training
This script demonstrates how to run a minimal training example
"""

import torch
import os

# Set wandb to offline mode
os.environ['WANDB_MODE'] = 'offline'

# Configuration for quick demo
demo_config = {
    'model': {
        'backbone': 'vit_base_patch16_224',
        'embed_dim': 128,
        'hidden_dim': 512,
        'k_tokens': 8,
        'tau_gumbel': 1.0,
        'pretrained': False  # Faster download for demo
    },
    'data': {
        'dataset_name': 'cotton80',
        'data_root': './data',
        'input_size': 224,
        'group_size': 5,
        'm': 4,
        'groups_per_batch': 2,  # Smaller for demo
        'val_batch_size': 16,   # Smaller for demo
        'num_workers': 2,       # Fewer workers for demo
        'train_transform': 'train_medium',
        'num_classes': 80  # Will be updated automatically
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
        'epochs': 2,  # Very short demo
        'use_ema': True,
        'ema_decay': 0.999,
        'save_dir': './demo_checkpoints',
        'max_checkpoints': 2
    }
}

def run_demo():
    """Run a quick demo of WallyCL training"""
    print("🚀 Starting WallyCL Demo")
    print("=" * 50)
    
    # Import here to avoid issues if modules have problems
    try:
        from train import WallyClTrainer
        
        # Create trainer
        trainer = WallyClTrainer(demo_config)
        
        print("✅ Trainer created successfully")
        print("✅ Starting demo training...")
        print("📝 Note: This is a minimal demo with 2 epochs")
        print("📝 For full training, use: python train.py --dataset cotton80")
        print("")
        
        # Run training
        trainer.train()
        
        print("🎉 Demo completed successfully!")
        print("📁 Check ./demo_checkpoints/ for saved models")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("💡 Make sure you have all dependencies installed:")
        print("   pip install -r requirements.txt")
        print("")
        print("💡 You can also run the test script instead:")
        print("   python test_implementation.py")

if __name__ == '__main__':
    run_demo()
