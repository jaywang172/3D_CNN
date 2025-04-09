import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 創建一個簡單的3D人臉數據集
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

# 創建數據加載器
def create_data_loaders(batch_size=16, train_samples=800, val_samples=200, test_samples=100):
    """
    創建訓練、驗證和測試數據加載器
    
    參數:
        batch_size (int): 批次大小
        train_samples (int): 訓練樣本數量
        val_samples (int): 驗證樣本數量
        test_samples (int): 測試樣本數量
        
    返回:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 設置隨機種子以確保結果可重現
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 創建數據集
    train_dataset = Face3DDataset(num_samples=train_samples, is_train=True)
    val_dataset = Face3DDataset(num_samples=val_samples, is_train=False)
    test_dataset = Face3DDataset(num_samples=test_samples, is_train=False)
    
    # 創建數據加載器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader