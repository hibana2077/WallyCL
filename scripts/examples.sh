#!/bin/bash

# WallyCL Training Script Examples

echo "WallyCL Training Examples"
echo "========================="

# Example 1: Basic training on Cotton80
echo "1. Training on Cotton80 dataset:"
echo "python train.py --dataset cotton80"
echo ""

# Example 2: Training with custom configuration
echo "2. Training with custom config:"
echo "python train.py --config configs/default.yaml --dataset soybean"
echo ""

# Example 3: Resume training
echo "3. Resume training from checkpoint:"
echo "python train.py --resume checkpoints/best_checkpoint.pth"
echo ""

# Example 4: Training with different parameters
echo "4. Training with specific parameters:"
echo "python train.py \\"
echo "    --dataset cotton80 \\"
echo "    --config configs/default.yaml"
echo ""

# Example 5: Evaluation
echo "5. Evaluate trained model:"
echo "python evaluate.py \\"
echo "    --checkpoint checkpoints/best_checkpoint.pth \\"
echo "    --dataset cotton80 \\"
echo "    --output results/cotton80_results.json"
echo ""

# Example 6: Inference on single image
echo "6. Single image inference:"
echo "python inference.py \\"
echo "    --checkpoint checkpoints/best_checkpoint.pth \\"
echo "    --image path/to/leaf_image.jpg \\"
echo "    --output visualization.png"
echo ""

# Example 7: Odd-one-out detection
echo "7. Group odd-one-out detection:"
echo "python inference.py \\"
echo "    --checkpoint checkpoints/best_checkpoint.pth \\"
echo "    --group img1.jpg img2.jpg img3.jpg img4.jpg img5.jpg \\"
echo "    --output group_analysis.png"
echo ""

# Example 8: Test implementation
echo "8. Test implementation:"
echo "python test_implementation.py"
echo ""

echo "Available datasets:"
echo "- cotton80"
echo "- soybean"
echo "- soy_ageing_r1"
echo "- soy_ageing_r3"
echo "- soy_ageing_r4"
echo "- soy_ageing_r5"
echo "- soy_ageing_r6"
