# 3D人臉辨識系統 (Face3D)

## 專案概述

本專案實現了一個針對邊緣裝置優化的3D人臉辨識系統，結合了深度學習（3D CNN）和電腦視覺技術（OpenCV）。系統能夠高效地處理3D人臉數據，並在資源受限的邊緣裝置上進行部署。

### 主要特點

- **輕量化3D CNN模型**：使用深度可分離卷積等技術，減少模型參數量和計算量
- **模型優化**：實現了模型剪枝和量化，顯著減少模型大小和推理時間
- **OpenCV整合**：利用OpenCV進行圖像預處理和人臉檢測
- **邊緣裝置部署**：針對資源受限環境進行優化，支援TorchScript導出
- **完整評估指標**：提供ROC曲線、混淆矩陣等多種性能評估方法

## 安裝指南

### 環境需求

- Python 3.7+
- PyTorch 1.7+
- OpenCV 4.5+
- NumPy
- Matplotlib
- scikit-learn
- seaborn

### 安裝步驟

1. 克隆此專案到本地：

```bash
git clone https://github.com/yourusername/face3d-recognition.git
cd face3d-recognition
```

2. 安裝所需依賴：

```bash
pip install torch torchvision numpy matplotlib scikit-learn seaborn opencv-python
```

## 使用方法

### 模型訓練

使用`face3d_train_example.py`腳本進行模型訓練：

```bash
python face3d_train_example.py
```

此腳本將：
1. 創建模擬的3D人臉數據集（在實際應用中應替換為真實數據）
2. 訓練基本的3D CNN模型
3. 應用模型剪枝和量化優化
4. 評估模型性能並生成評估圖表
5. 導出優化後的模型為TorchScript格式

### 結合OpenCV進行人臉檢測和辨識

以下是結合OpenCV進行人臉檢測和3D人臉辨識的基本流程：

```python
import cv2
import torch
import numpy as np
from optimized_face3dcnn import OptimizedFace3DCNN

# 載入預訓練模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = OptimizedFace3DCNN(num_classes=2).to(device)
model.load_state_dict(torch.load("optimized_face3d_model.pth"))
model.eval()

# 載入OpenCV人臉檢測器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 開啟攝像頭
cap = cv2.VideoCapture(0)

while True:
    # 讀取影像幀
    ret, frame = cap.read()
    if not ret:
        break
        
    # 轉換為灰度圖像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 檢測人臉
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        # 繪製人臉框
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # 提取人臉區域並進行預處理
        face_roi = frame[y:y+h, x:x+w]
        
        # 在實際應用中，這裡需要將2D人臉轉換為3D表示
        # 這可能需要深度相機或其他3D重建技術
        # 以下僅為示例，實際應用需要替換
        face_3d = np.random.randn(1, 1, 32, 32, 32).astype(np.float32)  # 模擬3D數據
        face_tensor = torch.from_numpy(face_3d).to(device)
        
        # 模型推理
        with torch.no_grad():
            output = model(face_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            pred_class = torch.argmax(prob, dim=1).item()
            confidence = prob[0][pred_class].item()
        
        # 顯示結果
        label = "Face" if pred_class == 1 else "Not Face"
        cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    # 顯示結果
    cv2.imshow('Face Detection & Recognition', frame)
    
    # 按'q'退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
```

## 模型架構

### OptimizedFace3DCNN

本專案使用了針對邊緣裝置優化的3D CNN模型，主要特點包括：

1. **深度可分離卷積**：將標準卷積分解為深度卷積和逐點卷積，大幅減少參數量
2. **批量正規化**：提高訓練穩定性和泛化能力
3. **全局平均池化**：減少全連接層的參數量
4. **Dropout正則化**：防止過擬合

模型架構如下：

```
OptimizedFace3DCNN(
  (conv1_depthwise): Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
  (conv1_pointwise): Conv3d(1, 8, kernel_size=(1, 1, 1), stride=(1, 1, 1))
  (bn1): BatchNorm3d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool1): MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2_depthwise): Conv3d(8, 8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=8)
  (conv2_pointwise): Conv3d(8, 16, kernel_size=(1, 1, 1), stride=(1, 1, 1))
  (bn2): BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool2): MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv3_depthwise): Conv3d(16, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=16)
  (conv3_pointwise): Conv3d(16, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1))
  (bn3): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool3): AdaptiveAvgPool3d(output_size=1)
  (fc1): Linear(in_features=32, out_features=16, bias=True)
  (dropout): Dropout(p=0.2, inplace=False)
  (fc2): Linear(in_features=16, out_features=2, bias=True)
)
```

## 模型優化技術

### 1. 模型剪枝

剪枝是一種通過移除不重要的權重來減少模型大小的技術。本專案使用L1正則化剪枝，移除約30%的權重，同時保持模型性能。

```python
# 模型剪枝示例
pruned_model = prune_model(trained_model, amount=0.3)
```

### 2. 模型量化

量化是將模型的權重從32位浮點數（float32）轉換為8位整數（int8）的過程，可顯著減少模型大小和推理時間。

```python
# 模型量化示例
quantized_model = quantize_model(pruned_model)
```

### 3. TorchScript導出

為了在邊緣裝置上高效部署，本專案支援將模型導出為TorchScript格式。

```python
# TorchScript導出示例
example_input = torch.randn(1, 1, 32, 32, 32)
traced_model = torch.jit.trace(model, example_input)
torch.jit.save(traced_model, "optimized_face3d_model.pt")
```

## 性能評估

### 評估指標

本專案提供多種評估指標來衡量模型性能：

- **準確率（Accuracy）**：正確預測的比例
- **精確率（Precision）**：正確識別為人臉的比例
- **召回率（Recall）**：成功識別出的真實人臉比例
- **F1分數**：精確率和召回率的調和平均
- **ROC曲線下面積（AUC）**：模型區分能力的綜合指標

### 視覺化評估

#### 混淆矩陣

混淆矩陣顯示了模型預測的四種可能結果：
- 真正例（TP）：正確識別為人臉
- 真負例（TN）：正確識別為非人臉
- 假正例（FP）：錯誤識別為人臉
- 假負例（FN）：錯誤識別為非人臉

#### ROC曲線

ROC曲線展示了不同閾值下真正例率（TPR）與假正例率（FPR）的關係，曲線下面積（AUC）越大表示模型性能越好。

## 邊緣裝置部署

### 資源需求

優化後的模型具有以下特點：
- 模型大小：< 1MB
- 推理時間：在典型邊緣裝置上 < 100ms/幀
- 記憶體使用：< 50MB

### 部署步驟

1. 在目標裝置上安裝PyTorch和OpenCV
2. 複製優化後的模型文件（.pt格式）到目標裝置
3. 使用上述示例代碼進行部署和推理

## 常見問題

**Q: 如何使用真實的3D人臉數據？**

A: 您可以使用深度相機（如Intel RealSense、Microsoft Kinect）獲取3D人臉數據，或使用專業的3D掃描設備。另外，也可以使用基於單目相機的3D重建技術。

**Q: 模型在邊緣裝置上運行緩慢怎麼辦？**

A: 可以嘗試以下優化方法：
- 進一步減少模型層數和通道數
- 增加剪枝比例（如0.5）
- 考慮使用半精度浮點數（FP16）
- 使用ONNX或TensorRT進行進一步優化

**Q: 如何提高人臉檢測的準確率？**

A: 可以考慮以下方法：
- 使用更先進的人臉檢測器（如MTCNN、RetinaFace）
- 增加訓練數據的多樣性
- 使用數據增強技術
- 調整模型架構和超參數

## 許可證

本專案採用MIT許可證。詳情請參閱LICENSE文件。

## 聯絡方式

如有任何問題或建議，請聯絡：38661797jay@gmail.com
