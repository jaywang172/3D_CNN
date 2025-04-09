import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import numpy as np

# 導入自定義模塊
from models.face3dcnn import OptimizedFace3DCNN
from models.model_optimization import prune_model, quantize_model, get_model_size, export_model
from data.dataset import create_data_loaders
from utils.training import train_model
from utils.evaluation import evaluate_model

def main():
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='Train 3D Face Recognition Model')
    parser.add_argument('--batch-size', type=int, default=16, help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 20)')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay (default: 1e-5)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--prune-amount', type=float, default=0.3, help='pruning amount (default: 0.3)')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience (default: 3)')
    parser.add_argument('--output-dir', type=str, default='results', help='output directory (default: results)')
    args = parser.parse_args()
    
    # 確保輸出目錄存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 設置隨機種子以確保結果可重現
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 設置設備
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # 創建數據集和數據加載器
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=args.batch_size,
        train_samples=800,
        val_samples=200,
        test_samples=100
    )
    
    # 創建模型
    model = OptimizedFace3DCNN(num_classes=2).to(device)
    print(f"Original model size: {get_model_size(model):.2f} MB")
    
    # 設置損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # 訓練模型
    print("\nTraining model...")
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        patience=args.patience,
        output_dir=args.output_dir
    )
    
    # 評估原始模型
    print("\nEvaluating original model...")
    original_metrics = evaluate_model(trained_model, test_loader, device, args.output_dir)
    print("Original model metrics:")
    for metric, value in original_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 模型剪枝
    print("\nPruning model...")
    pruned_model = prune_model(trained_model, amount=args.prune_amount)
    print(f"Pruned model size: {get_model_size(pruned_model):.2f} MB")
    
    # 評估剪枝後的模型
    print("\nEvaluating pruned model...")
    pruned_metrics = evaluate_model(pruned_model, test_loader, device, args.output_dir)
    print("Pruned model metrics:")
    for metric, value in pruned_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 模型量化（CPU）
    final_model = pruned_model
    if device.type == 'cpu':
        print("\nQuantizing model...")
        quantized_model = quantize_model(pruned_model)
        print(f"Quantized model size: {get_model_size(quantized_model):.2f} MB")
        
        # 評估量化後的模型
        print("\nEvaluating quantized model...")
        quantized_metrics = evaluate_model(quantized_model, test_loader, device, args.output_dir)
        print("Quantized model metrics:")
        for metric, value in quantized_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        final_model = quantized_model
    else:
        print("\nQuantization is only supported on CPU. Skipping quantization evaluation.")
    
    # 保存最終模型
    print("\nSaving optimize