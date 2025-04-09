import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

# 優化的3D CNN模型，針對邊緣裝置
class OptimizedFace3DCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(OptimizedFace3DCNN, self).__init__()
        
        # 使用深度可分離卷積來減少參數量和計算量
        # 第一個卷積塊 - 使用更小的濾波器和通道數
        self.conv1_depthwise = nn.Conv3d(1, 1, kernel_size=3, padding=1, groups=1)
        self.conv1_pointwise = nn.Conv3d(1, 8, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(8)  # 批量正規化提高訓練穩定性
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # 第二個卷積塊
        self.conv2_depthwise = nn.Conv3d(8, 8, kernel_size=3, padding=1, groups=8)
        self.conv2_pointwise = nn.Conv3d(8, 16, kernel_size=1)
        self.bn2 = nn.BatchNorm3d(16)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # 第三個卷積塊
        self.conv3_depthwise = nn.Conv3d(16, 16, kernel_size=3, padding=1, groups=16)
        self.conv3_pointwise = nn.Conv3d(16, 32, kernel_size=1)
        self.bn3 = nn.BatchNorm3d(32)
        self.pool3 = nn.AdaptiveAvgPool3d(1)  # 全局平均池化減少參數
        
        # 全連接層 - 減少神經元數量
        self.fc1 = nn.Linear(32, 16)
        self.dropout = nn.Dropout(0.2)  # 添加dropout防止過擬合
        self.fc2 = nn.Linear(16, num_classes)
    
    def forward(self, x):
        # 第一個卷積塊
        x = self.conv1_depthwise(x)
        x = self.conv1_pointwise(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # 第二個卷積塊
        x = self.conv2_depthwise(x)
        x = self.conv2_pointwise(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # 第三個卷積塊
        x = self.conv3_depthwise(x)
        x = self.conv3_pointwise(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # 全連接層
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 模型量化函數 - 將模型轉換為INT8格式以減少內存佔用和提高推理速度
def quantize_model(model):
    # 靜態量化
    model_quantized = torch.quantization.quantize_dynamic(
        model,  # 原始模型
        {nn.Linear, nn.Conv3d},  # 要量化的層類型
        dtype=torch.qint8  # 量化數據類型
    )
    return model_quantized

# 模型剪枝函數 - 移除不重要的權重以減少模型大小
def prune_model(model, amount=0.3):
    # 這裡使用簡單的全局剪枝作為示例
    # 在實際應用中，可能需要更複雜的剪枝策略
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear):
            torch.nn.utils.prune.l1_unstructured(module, 'weight', amount=amount)
    return model

# 計算模型大小（MB）
def get_model_size(model):
    torch_params = sum(p.numel() for p in model.parameters())
    torch_size = torch_params * 4 / (1024 * 1024)  # 假設每個參數為4字節（float32）
    return torch_size

# 評估模型性能並繪製ROC曲線和混淆矩陣
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # 獲取預測結果
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # 假設正類的概率在索引1
    
    # 計算混淆矩陣
    cm = confusion_matrix(all_labels, all_preds)
    
    # 繪製混淆矩陣
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # 計算並繪製ROC曲線
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve.png')
    plt.close()
    
    # 計算準確率、精確率、召回率和F1分數
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    
    return metrics

# 訓練函數，包含早停機制
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, patience=5):
    best_val_loss = float('inf')
    counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 訓練階段
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向傳播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向傳播和優化
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # 驗證階段
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {epoch_train_loss:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}')
        
        # 早停機制
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # 繪製訓練和驗證損失曲線
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curves.png')
    plt.close()
    
    # 載入最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    return model

# 主函數示例
if __name__ == "__main__":
    # 設置設備
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 創建模型
    original_model = OptimizedFace3DCNN(num_classes=2).to(device)
    print(f"Original model size: {get_model_size(original_model):.2f} MB")
    
    # 示例輸入和標籤: (batch_size, channels, depth, height, width)
    sample_input = torch.randn(8, 1, 32, 32, 32).to(device)
    labels = torch.randint(0, 2, (8,)).to(device)
    
    # 前向傳播測試
    outputs = original_model(sample_input)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    print(f'Initial loss: {loss.item():.4f}')
    
    # 模型剪枝
    pruned_model = prune_model(original_model)
    print(f"Pruned model size: {get_model_size(pruned_model):.2f} MB")
    
    # 模型量化 (僅在CPU上支持)
    if device.type == 'cpu':
        quantized_model = quantize_model(pruned_model)
        print(f"Quantized model size: {get_model_size(quantized_model):.2f} MB")
    else:
        print("Quantization is only supported on CPU")
        quantized_model = pruned_model
    
    # 注意：在實際應用中，您需要準備真實的數據集和數據加載器
    # 這裡僅作為示例，使用隨機生成的數據
    
    print("\nNote: This is a demonstration with random data.")
    print("In a real application, you would need to:")
    print("1. Prepare your 3D face dataset")
    print("2. Create proper data loaders")
    print("3. Train the model with real data")
    print("4. Evaluate on a test set")
    print("5. Export the model for edge devices using TorchScript or ONNX")
    
    print("\nTo deploy on edge devices, consider:")
    print("1. Using TensorRT for NVIDIA devices")
    print("2. Using ONNX Runtime for cross-platform deployment")
    print("3. Using TFLite or CoreML for mobile devices")
    print("4. Further optimizing based on specific hardware constraints")