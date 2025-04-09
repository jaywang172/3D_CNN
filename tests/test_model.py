import unittest
import torch
import sys
import os
import numpy as np

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 導入自定義模塊
from models.face3dcnn import OptimizedFace3DCNN
from models.model_optimization import prune_model, get_model_size

class TestFace3DCNN(unittest.TestCase):
    def setUp(self):
        # 設置測試環境
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = OptimizedFace3DCNN(num_classes=2).to(self.device)
        self.batch_size = 4
        self.input_shape = (self.batch_size, 1, 32, 32, 32)
        
    def test_model_forward(self):
        # 測試模型前向傳播
        x = torch.randn(self.input_shape).to(self.device)
        output = self.model(x)
        
        # 檢查輸出形狀
        self.assertEqual(output.shape, (self.batch_size, 2))
        
    def test_model_parameters(self):
        # 測試模型參數數量
        param_count = sum(p.numel() for p in self.model.parameters())
        # 確保參數數量在合理範圍內（針對輕量化模型）
        self.assertLess(param_count, 100000)  # 參數數量應小於10萬
        
    def test_model_size(self):
        # 測試模型大小
        model_size = get_model_size(self.model)
        # 確保模型大小在合理範圍內（針對邊緣裝置）
        self.assertLess(model_size, 1.0)  # 模型大小應小於1MB
        
    def test_model_pruning(self):
        # 測試模型剪枝
        original_size = get_model_size(self.model)
        pruned_model = prune_model(self.model, amount=0.3)
        pruned_size = get_model_size(pruned_model)
        
        # 確保剪枝後模型大小減小
        self.assertLess(pruned_size, original_size)
        
        # 確保剪枝後模型仍能正常運行
        x = torch.randn(self.input_shape).to(self.device)
        output = pruned_model(x)
        self.assertEqual(output.shape, (self.batch_size, 2))
        
    def test_model_inference(self):
        # 測試模型推理
        self.model.eval()
        x = torch.randn(1, 1, 32, 32, 32).to(self.device)
        
        with torch.no_grad():
            # 測量推理時間
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            output = self.model(x)
            end.record()
            
            # 等待GPU完成
            torch.cuda.synchronize()