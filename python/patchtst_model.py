#!/usr/bin/env python3
"""
PatchTST模型实现
用于时间序列预测，支持多变量时间序列
基于Transformer的Patch-based时间序列预测模型
"""

import json
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from typing import List
import pickle
from pathlib import Path


class PatchEmbedding(nn.Module):
    """Patch嵌入层"""
    def __init__(self, d_model, patch_len=16, stride=8):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = nn.Parameter(torch.randn(1000, d_model))
        
    def forward(self, x):
        # x: [Batch, seq_len, n_vars]
        n_vars = x.shape[2]
        # Padding
        x = self.padding_patch_layer(x)
        # 创建patches
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # x: [Batch, n_patches, n_vars, patch_len]
        batch_size, n_patches, n_vars, patch_len = x.shape
        # 展平并转置: [Batch, n_patches, n_vars, patch_len] -> [Batch, n_patches, n_vars * patch_len]
        x = x.reshape(batch_size, n_patches, -1)
        # 重新reshape为: [Batch, n_patches * n_vars, patch_len]
        x = x.reshape(batch_size, n_patches * n_vars, patch_len)
        # 嵌入
        x = self.value_embedding(x)
        # 添加位置编码
        x = x + self.position_embedding[:x.shape[1], :].unsqueeze(0)
        return x


class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, n_layers=3):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
    def forward(self, x):
        return self.transformer_encoder(x)


class PatchTSTModel(nn.Module):
    """PatchTST模型"""
    def __init__(self, seq_len, pred_len, enc_in, d_model=128, n_heads=8, e_layers=3, 
                 d_ff=256, dropout=0.1, patch_len=16, stride=8):
        super(PatchTSTModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = d_model
        
        # Patch嵌入
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride)
        
        # Transformer编码器
        self.encoder = TransformerEncoder(d_model, n_heads, d_ff, dropout, e_layers)
        
        # 输出层
        self.head = nn.Linear(d_model, pred_len)
        
    def forward(self, x):
        # x: [Batch, seq_len, n_vars]
        # Patch嵌入
        x = self.patch_embedding(x)
        # x: [Batch, n_patches * n_vars, d_model]
        
        # Transformer编码
        x = self.encoder(x)
        # x: [Batch, n_patches * n_vars, d_model]
        
        # 取最后一个patch的输出（或平均池化）
        # 对于多变量，我们需要对每个变量分别处理
        n_patches = x.shape[1] // self.enc_in
        # 重新reshape: [Batch, n_patches * n_vars, d_model] -> [Batch, n_patches, n_vars, d_model]
        x = x.reshape(x.shape[0], n_patches, self.enc_in, self.d_model)
        # 取每个变量的最后一个patch
        x = x[:, -1, :, :]  # [Batch, n_vars, d_model]
        
        # 预测
        x = self.head(x)  # [Batch, n_vars, pred_len]
        x = x.transpose(1, 2)  # [Batch, pred_len, n_vars]
        
        return x


class PatchTSTPredictor:
    """PatchTST预测器封装类"""
    def __init__(self, seq_len=96, pred_len=1, model_dir='./models', 
                 d_model=128, n_heads=8, e_layers=3, d_ff=256, dropout=0.1, 
                 patch_len=16, stride=8):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.enc_in = None
        
        # 自动调整patch_len和stride，确保patch_len <= seq_len
        original_patch_len = patch_len
        if patch_len > seq_len:
            # 如果patch_len大于seq_len，调整为seq_len的一半（向下取整）
            patch_len = max(1, seq_len // 2)
            stride = max(1, patch_len // 2)
            print(f"Warning: patch_len ({original_patch_len}) > seq_len ({seq_len}). "
                  f"Auto-adjusted to patch_len={patch_len}, stride={stride}", file=sys.stderr)
        
        # 确保stride <= patch_len
        if stride > patch_len:
            stride = max(1, patch_len // 2)
            print(f"Warning: stride ({stride}) > patch_len ({patch_len}). "
                  f"Auto-adjusted to stride={stride}", file=sys.stderr)
        
        # 将调整后的值赋给实例变量
        self.patch_len = patch_len
        self.stride = stride
        
        # 设备选择：优先使用GPU，然后是M1的MPS，最后是CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("✓ CUDA acceleration enabled", file=sys.stderr)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')  # M1/M2芯片的Metal加速
            print("✓ M1/M2 MPS (Metal) acceleration enabled", file=sys.stderr)
        else:
            self.device = torch.device('cpu')
            print("⚠ Using CPU (no GPU/MPS acceleration available)", file=sys.stderr)
        print(f"Device: {self.device}", file=sys.stderr)
        sys.stderr.flush()
        # 数据归一化参数
        self.data_mean = None
        self.data_std = None
        # 模型参数
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.dropout = dropout
        
    def fit(self, data: List[List[float]], epochs=10, learning_rate=0.001):
        """
        训练模型
        data: 训练数据，每个内层列表代表一个时间点的多变量数据
        """
        # 转换为numpy数组
        data_array = np.array(data, dtype=np.float32)
        n_samples, n_features = data_array.shape
        self.enc_in = n_features
        
        # 数据归一化：计算均值和标准差（按特征维度）
        # 使用训练数据的统计量进行标准化
        self.data_mean = np.mean(data_array, axis=0, keepdims=True)  # [1, n_features]
        self.data_std = np.std(data_array, axis=0, keepdims=True)  # [1, n_features]
        # 避免除零，如果std为0则设为1
        self.data_std = np.where(self.data_std < 1e-8, 1.0, self.data_std)
        
        # 标准化数据
        data_normalized = (data_array - self.data_mean) / self.data_std
        
        # 输出设备信息（确保在训练时显示）
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and str(self.device) == 'mps':
            print("✓ M1/M2 MPS (Metal) acceleration enabled for training", file=sys.stderr)
        elif torch.cuda.is_available() and str(self.device) == 'cuda':
            print("✓ CUDA acceleration enabled for training", file=sys.stderr)
        else:
            print("⚠ Using CPU for training (no GPU/MPS acceleration)", file=sys.stderr)
        print(f"Training device: {self.device}", file=sys.stderr)
        
        print(f"Data statistics - Mean range: [{np.min(self.data_mean):.2f}, {np.max(self.data_mean):.2f}], "
              f"Std range: [{np.min(self.data_std):.2f}, {np.max(self.data_std):.2f}]", file=sys.stderr)
        print(f"Normalized data range: [{np.min(data_normalized):.2f}, {np.max(data_normalized):.2f}]", file=sys.stderr)
        sys.stderr.flush()
        
        # 创建模型
        self.model = PatchTSTModel(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            enc_in=self.enc_in,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            patch_len=self.patch_len,
            stride=self.stride
        ).to(self.device)
        
        # 准备训练数据
        if n_samples < self.seq_len + self.pred_len:
            raise ValueError(f"Data length {n_samples} must be >= seq_len + pred_len ({self.seq_len + self.pred_len})")
        
        # 创建训练样本（使用归一化后的数据）
        # 限制训练样本数量以避免内存溢出（最多50000个样本）
        max_samples = 50000
        total_possible_samples = n_samples - self.seq_len - self.pred_len + 1
        
        print(f"Creating training samples: {total_possible_samples} possible samples...", file=sys.stderr)
        sys.stderr.flush()
        
        if total_possible_samples > max_samples:
            # 均匀采样
            indices = np.linspace(0, total_possible_samples - 1, max_samples, dtype=int)
            print(f"Sampling {max_samples} samples from {total_possible_samples} possible samples to reduce memory usage", file=sys.stderr)
            sys.stderr.flush()
        else:
            indices = np.arange(total_possible_samples)
        
        print(f"Building {len(indices)} training samples...", file=sys.stderr)
        sys.stderr.flush()
        
        X_train = []
        y_train = []
        for idx, i in enumerate(indices):
            if (idx + 1) % 10000 == 0:
                print(f"  Processed {idx + 1}/{len(indices)} samples...", file=sys.stderr)
                sys.stderr.flush()
            X_train.append(data_normalized[i:i+self.seq_len])
            y_train.append(data_normalized[i+self.seq_len:i+self.seq_len+self.pred_len])
        
        print(f"Converting to numpy arrays...", file=sys.stderr)
        sys.stderr.flush()
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        print(f"Training data prepared: X_train shape={X_train.shape}, y_train shape={y_train.shape}", file=sys.stderr)
        sys.stderr.flush()
        
        # 转换为torch tensor（先不移动到GPU，使用批次训练）
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        
        # 训练参数
        batch_size = 32  # 批次大小，可以根据GPU内存调整
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        n_batches = (len(X_train) + batch_size - 1) // batch_size
        print(f"Starting training: {len(X_train)} samples, {n_features} features, {epochs} epochs, batch_size={batch_size}, {n_batches} batches per epoch", file=sys.stderr)
        sys.stderr.flush()
        
        print(f"Starting training loop...", file=sys.stderr)
        sys.stderr.flush()
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            # 批次训练
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(X_train))
                
                X_batch = X_train[start_idx:end_idx].to(self.device)
                y_batch = y_train[start_idx:end_idx].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # 释放GPU内存
                del X_batch, y_batch, outputs, loss
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = epoch_loss / n_batches
            
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}", file=sys.stderr)
                sys.stderr.flush()
        
        print(f"Training completed. Final loss: {avg_loss:.6f}", file=sys.stderr)
        sys.stderr.flush()
        
        # 保存模型（包含归一化参数）
        print(f"Saving model with normalization parameters (mean shape: {self.data_mean.shape}, std shape: {self.data_std.shape})", file=sys.stderr)
        sys.stderr.flush()
        self.save_model()
        print(f"Model saved successfully to {self.model_dir / 'patchtst_model.pkl'}", file=sys.stderr)
        sys.stderr.flush()
        
    def predict(self, window: List[List[float]]) -> List[float]:
        """
        预测下一个时间点
        window: 滑动窗口数据，每个内层列表代表一个时间点的多变量数据
        """
        if self.model is None:
            raise ValueError("Model not trained, please call fit() first")
        
        # 转换为numpy数组
        window_array = np.array(window, dtype=np.float32)
        
        # 检查窗口大小
        if window_array.shape[0] != self.seq_len:
            raise ValueError(f"Window size {window_array.shape[0]} must equal seq_len {self.seq_len}")
        
        # 检查归一化参数是否已加载
        if self.data_mean is None or self.data_std is None:
            raise ValueError("Normalization parameters not loaded. Please train or load the model first.")
        
        # 输出设备信息（仅在第一次预测时输出，避免过多输出）
        if not hasattr(self, '_prediction_device_shown'):
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and str(self.device) == 'mps':
                print("✓ M1/M2 MPS (Metal) acceleration enabled for prediction", file=sys.stderr)
            elif torch.cuda.is_available() and str(self.device) == 'cuda':
                print("✓ CUDA acceleration enabled for prediction", file=sys.stderr)
            else:
                print("⚠ Using CPU for prediction (no GPU/MPS acceleration)", file=sys.stderr)
            print(f"Prediction device: {self.device}", file=sys.stderr)
            sys.stderr.flush()
            self._prediction_device_shown = True
        
        # 归一化输入窗口
        window_normalized = (window_array - self.data_mean) / self.data_std
        
        # 转换为torch tensor并添加batch维度
        window_tensor = torch.from_numpy(window_normalized).float().unsqueeze(0).to(self.device)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            output = self.model(window_tensor)
            # output shape: [1, pred_len, n_features]
            prediction_normalized = output[0, 0, :].cpu().numpy()  # 取第一个预测时间点的所有特征
        
        # 反归一化：将预测结果转换回原始尺度
        prediction = prediction_normalized * self.data_std[0] + self.data_mean[0]
        
        return prediction.tolist()
    
    def save_model(self):
        """保存模型"""
        model_path = self.model_dir / 'patchtst_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model_state': self.model.state_dict(),
                'seq_len': self.seq_len,
                'pred_len': self.pred_len,
                'enc_in': self.enc_in,
                'd_model': self.d_model,
                'n_heads': self.n_heads,
                'e_layers': self.e_layers,
                'd_ff': self.d_ff,
                'dropout': self.dropout,
                'patch_len': self.patch_len,
                'stride': self.stride,
                'data_mean': self.data_mean,
                'data_std': self.data_std
            }, f)
    
    def load_model(self):
        """加载模型"""
        model_path = self.model_dir / 'patchtst_model.pkl'
        if not model_path.exists():
            return False
        
        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.seq_len = checkpoint['seq_len']
        self.pred_len = checkpoint['pred_len']
        self.enc_in = checkpoint['enc_in']
        self.d_model = checkpoint.get('d_model', 128)
        self.n_heads = checkpoint.get('n_heads', 8)
        self.e_layers = checkpoint.get('e_layers', 3)
        self.d_ff = checkpoint.get('d_ff', 256)
        self.dropout = checkpoint.get('dropout', 0.1)
        patch_len = checkpoint.get('patch_len', 16)
        stride = checkpoint.get('stride', 8)
        
        # 自动调整patch_len和stride，确保patch_len <= seq_len
        if patch_len > self.seq_len:
            # 如果patch_len大于seq_len，调整为seq_len的一半（向下取整）
            patch_len = max(1, self.seq_len // 2)
            stride = max(1, patch_len // 2)
            print(f"Warning: Loaded patch_len ({checkpoint.get('patch_len', 16)}) > seq_len ({self.seq_len}). "
                  f"Auto-adjusted to patch_len={patch_len}, stride={stride}", file=sys.stderr)
        
        # 确保stride <= patch_len
        if stride > patch_len:
            stride = max(1, patch_len // 2)
            print(f"Warning: Loaded stride ({checkpoint.get('stride', 8)}) > patch_len ({patch_len}). "
                  f"Auto-adjusted to stride={stride}", file=sys.stderr)
        
        self.patch_len = patch_len
        self.stride = stride
        
        # 加载归一化参数（兼容旧模型）
        self.data_mean = checkpoint.get('data_mean', None)
        self.data_std = checkpoint.get('data_std', None)
        
        self.model = PatchTSTModel(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            enc_in=self.enc_in,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            patch_len=self.patch_len,
            stride=self.stride
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        return True


def main():
    """主函数，处理命令行输入"""
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Missing command argument"}), file=sys.stderr)
        sys.exit(1)
    
    command = sys.argv[1]
    
    # 读取输入：支持从文件或stdin读取
    input_file = sys.argv[2] if len(sys.argv) > 2 else None
    try:
        if input_file:
            # 从文件读取
            with open(input_file, 'r') as f:
                input_str = f.read()
        else:
            # 从stdin读取
            input_str = sys.stdin.read()
        
        if not input_str.strip():
            print(json.dumps({"error": "Input data is empty"}), file=sys.stderr)
            sys.exit(1)
        input_data = json.loads(input_str)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"JSON parsing failed: {str(e)}"}), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"Failed to read input: {str(e)}"}), file=sys.stderr)
        sys.exit(1)
    
    if command == "fit":
        # 训练模型
        try:
            data = input_data['data']
            seq_len = input_data.get('seq_len', 96)
            pred_len = input_data.get('pred_len', 1)
            epochs = input_data.get('epochs', 10)
            learning_rate = input_data.get('learning_rate', 0.001)
            model_dir = input_data.get('model_dir', './models')
            d_model = input_data.get('d_model', 128)
            n_heads = input_data.get('n_heads', 8)
            e_layers = input_data.get('e_layers', 3)
            
            predictor = PatchTSTPredictor(seq_len=seq_len, pred_len=pred_len, 
                                        model_dir=model_dir, d_model=d_model,
                                        n_heads=n_heads, e_layers=e_layers)
            predictor.fit(data, epochs=epochs, learning_rate=learning_rate)
            
            result = {"status": "success", "message": "Model training completed"}
            print(json.dumps(result))
            sys.stdout.flush()  # 确保输出被刷新
        except KeyError as e:
            error_msg = json.dumps({"error": f"Missing required parameter: {str(e)}"})
            print(error_msg, file=sys.stderr)
            sys.stderr.flush()
            sys.exit(1)
        except Exception as e:
            import traceback
            error_msg = json.dumps({"error": f"Training failed: {str(e)}", "traceback": traceback.format_exc()})
            print(error_msg, file=sys.stderr)
            sys.stderr.flush()
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)
        
    elif command == "predict":
        # 预测
        try:
            window = input_data['window']
            model_dir = input_data.get('model_dir', './models')
            seq_len = input_data.get('seq_len', 96)
            pred_len = input_data.get('pred_len', 1)
            d_model = input_data.get('d_model', 128)
            n_heads = input_data.get('n_heads', 8)
            e_layers = input_data.get('e_layers', 3)
            
            predictor = PatchTSTPredictor(seq_len=seq_len, pred_len=pred_len,
                                        model_dir=model_dir, d_model=d_model,
                                        n_heads=n_heads, e_layers=e_layers)
            
            if not predictor.load_model():
                error_msg = json.dumps({"error": "Model not found, please train the model first"})
                print(error_msg, file=sys.stderr)
                sys.stderr.flush()
                sys.exit(1)
            
            # 检查归一化参数是否已加载
            if predictor.data_mean is None or predictor.data_std is None:
                error_msg = json.dumps({
                    "error": "Model file does not contain normalization parameters. "
                             "This appears to be an old model file created before normalization was added. "
                             "Please delete the old model file and retrain the model."
                })
                print(error_msg, file=sys.stderr)
                sys.stderr.flush()
                sys.exit(1)
            
            prediction = predictor.predict(window)
            result = {"prediction": prediction}
            print(json.dumps(result))
            sys.stdout.flush()  # 确保输出被刷新
        except KeyError as e:
            error_msg = json.dumps({"error": f"Missing required parameter: {str(e)}"})
            print(error_msg, file=sys.stderr)
            sys.stderr.flush()
            sys.exit(1)
        except Exception as e:
            import traceback
            error_msg = json.dumps({"error": f"Prediction failed: {str(e)}", "traceback": traceback.format_exc()})
            print(error_msg, file=sys.stderr)
            sys.stderr.flush()
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)
        
    else:
        print(json.dumps({"error": f"Unknown command: {command}"}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
