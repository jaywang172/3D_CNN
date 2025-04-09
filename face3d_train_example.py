import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from optimized_face3dcnn import OptimizedFace3DCNN, train_model, evaluate_model, prune_model, quantize_model, get_model_size

# 創建一個簡單的3D人臉數據集示例
class Face3DDataset(Dataset):
    def __init__(self, num_samples=100, is_train=True):
        self.num_samples = num_samples
        self.is_train = is_train
        
        # 生成隨機數據作為示例
        # 在實際應用中，這裡應該載入真實的3D人臉數據
        self.data = np.random.randn(num_samples, 1, 32, 32, 32).astype(np.float32)
        
        # 生成隨機標籤（0: 非人臉, 1: 人臉）
        self.labels = np.random.randint(0, 2, num_samples).astype(np.int64)
        
        # 如果是訓練集，進行簡單的數據增強
        if is_train:
            # 添加一些噪聲作為數據增強的示例
            self.data += 0.1 * np.random.randn(num_samples, 1, 32, 32, 32).astype(np.float32)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), torch.tensor(self.labels[idx])

# 主函數：訓練和評估模型
def main():
    # 設置隨機種子以確保結果可重現
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 設置設備
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 創建數據集和數據加載器
    train_dataset = Face3DDataset(num_samples=800, is_train=True)
    val_dataset = Face3DDataset(num_samples=200, is_train=False)
    test_dataset = Face3DDataset(num_samples=100, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 創建模型
    model = OptimizedFace3DCNN(num_classes=2).to(device)
    print(f"Original model size: {get_model_size(model):.2f} MB")
    
    # 設置損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # 添加權重衰減以減少過擬合
    
    # 訓練模型
    print("\nTraining model...")
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=20,  # 減少訓練輪數以加快訓練速度
        patience=3      # 早停機制的耐心值
    )
    
    # 評估原始模型
    print("\nEvaluating original model...")
    original_metrics = evaluate_model(trained_model, test_loader, device)
    print("Original model metrics:")
    for metric, value in original_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 模型剪枝
    print("\nPruning model...")
    pruned_model = prune_model(trained_model, amount=0.3)
    print(f"Pruned model size: {get_model_size(pruned_model):.2f} MB")
    
    # 評估剪枝後的模型
    print("\nEvaluating pruned model...")
    pruned_metrics = evaluate_model(pruned_model, test_loader, device)
    print("Pruned model metrics:")
    for metric, value in pruned_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 模型量化（僅在CPU上支持）
    if device.type == 'cpu':
        print("\nQuantizing model...")
        quantized_model = quantize_model(pruned_model)
        print(f"Quantized model size: {get_model_size(quantized_model):.2f} MB")
        
        # 評估量化後的模型
        print("\nEvaluating quantized model...")
        quantized_metrics = evaluate_model(quantized_model, test_loader, device)
        print("Quantized model metrics:")
        for metric, value in quantized_metrics.items():
            print(f"{metric}: {value:.4f}")
    else:
        print("\nQuantization is only supported on CPU. Skipping quantization evaluation.")
    
    # 保存最終模型
    print("\nSaving optimized model...")
    if device.type == 'cpu':
        final_model = quantized_model
    else:
        final_model = pruned_model
    
    torch.save(final_model.state_dict(), "optimized_face3d_model.pth")
    
    # 導出為TorchScript格式（適用於邊緣裝置部署）
    example_input = torch.randn(1, 1, 32, 32, 32).to(device)
    traced_model = torch.jit.trace(final_model, example_input)
    torch.jit.save(traced_model, "optimized_face3d_model.pt")
    
    print("\nOptimization complete! The model has been saved in both .pth and .pt formats.")
    print("The .pt format (TorchScript) is recommended for deployment on edge devices.")

if __name__ == "__main__":
    main()