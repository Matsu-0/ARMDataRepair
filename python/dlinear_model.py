#!/usr/bin/env python3
"""
DLinear模型实现
用于时间序列预测，支持多变量时间序列
"""

import json
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple
import pickle
from pathlib import Path


class MovingAverage(nn.Module):
    """移动平均模块，用于分解趋势"""
    def __init__(self, kernel_size, stride):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        if stride != 0:
            self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
        else:
            self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomposition(nn.Module):
    """序列分解模块"""
    def __init__(self, kernel_size):
        super(SeriesDecomposition, self).__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinearModel(nn.Module):
    """DLinear模型"""
    def __init__(self, seq_len, pred_len, individual, enc_in):
        super(DLinearModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        
        # 分解
        kernel_size = 25
        self.decompsition = SeriesDecomposition(kernel_size)
        self.enc_in = enc_in
        
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.enc_in):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        # x: [Batch, Channel, Input length]
        
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.enc_in):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]


class DLinearPredictor:
    """DLinear预测器封装类"""
    def __init__(self, seq_len=96, pred_len=1, individual=True, model_dir='./models'):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.enc_in = None
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
        
    def fit(self, data: List[List[float]], epochs=100, learning_rate=0.001):
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
        self.model = DLinearModel(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            individual=self.individual,
            enc_in=self.enc_in
        ).to(self.device)
        
        # 准备训练数据
        if n_samples < self.seq_len + self.pred_len:
            raise ValueError(f"Data length {n_samples} must be >= seq_len + pred_len ({self.seq_len + self.pred_len})")
        
        # 创建训练样本（使用归一化后的数据）
        X_train = []
        y_train = []
        for i in range(n_samples - self.seq_len - self.pred_len + 1):
            X_train.append(data_normalized[i:i+self.seq_len])
            y_train.append(data_normalized[i+self.seq_len:i+self.seq_len+self.pred_len])
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # 转换为torch tensor
        X_train = torch.from_numpy(X_train).float().to(self.device)
        y_train = torch.from_numpy(y_train).float().to(self.device)
        
        # 训练
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        print(f"Starting training: {n_samples} samples, {n_features} features, {epochs} epochs", file=sys.stderr)
        sys.stderr.flush()
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}", file=sys.stderr)
                sys.stderr.flush()
        
        print(f"Training completed. Final loss: {loss.item():.6f}", file=sys.stderr)
        sys.stderr.flush()
        
        # 保存模型（包含归一化参数）
        print(f"Saving model with normalization parameters (mean shape: {self.data_mean.shape}, std shape: {self.data_std.shape})", file=sys.stderr)
        sys.stderr.flush()
        self.save_model()
        print(f"Model saved successfully to {self.model_dir / 'dlinear_model.pkl'}", file=sys.stderr)
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
        model_path = self.model_dir / 'dlinear_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model_state': self.model.state_dict(),
                'seq_len': self.seq_len,
                'pred_len': self.pred_len,
                'individual': self.individual,
                'enc_in': self.enc_in,
                'data_mean': self.data_mean,
                'data_std': self.data_std
            }, f)
    
    def load_model(self):
        """加载模型"""
        model_path = self.model_dir / 'dlinear_model.pkl'
        if not model_path.exists():
            return False
        
        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.seq_len = checkpoint['seq_len']
        self.pred_len = checkpoint['pred_len']
        self.individual = checkpoint['individual']
        self.enc_in = checkpoint['enc_in']
        
        # 加载归一化参数（兼容旧模型）
        self.data_mean = checkpoint.get('data_mean', None)
        self.data_std = checkpoint.get('data_std', None)
        
        # 检查归一化参数是否存在
        if self.data_mean is None or self.data_std is None:
            print("Warning: Model file does not contain normalization parameters. "
                  "This may be an old model file. Please retrain the model.", file=sys.stderr)
            sys.stderr.flush()
            # 不返回False，让调用者决定如何处理
        
        self.model = DLinearModel(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            individual=self.individual,
            enc_in=self.enc_in
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
    
    # 交互模式：长期运行，等待预测命令
    if command == "interactive":
        try:
            model_dir = sys.argv[2] if len(sys.argv) > 2 else './models'
            seq_len = int(sys.argv[3]) if len(sys.argv) > 3 else 96
            enc_in = int(sys.argv[4]) if len(sys.argv) > 4 else 1
            
            # 创建预测器并加载模型
            predictor = DLinearPredictor(seq_len=seq_len, pred_len=1, individual=True, model_dir=model_dir)
            
            # 尝试加载模型
            if not predictor.load_model():
                print(json.dumps({"error": "Model not found, please train the model first"}), file=sys.stderr)
                sys.stderr.flush()
                sys.exit(1)
            
            # 检查归一化参数
            if predictor.data_mean is None or predictor.data_std is None:
                print(json.dumps({"error": "Model file does not contain normalization parameters"}), file=sys.stderr)
                sys.stderr.flush()
                sys.exit(1)
            
            # 发送就绪信号
            print("READY", flush=True)
            sys.stderr.flush()
            
            # 进入交互循环
            while True:
                try:
                    line = sys.stdin.readline()
                    if not line:
                        break  # EOF
                    
                    line = line.strip()
                    if line == "EXIT":
                        break
                    elif line == "PREDICT":
                        # 读取JSON输入（直到END标记）
                        json_lines = []
                        while True:
                            json_line = sys.stdin.readline()
                            if not json_line:
                                break
                            json_line = json_line.strip()
                            if json_line == "END":
                                break
                            json_lines.append(json_line)
                        
                        json_input = "\n".join(json_lines)
                        if not json_input.strip():
                            result = json.dumps({"error": "Empty input"})
                            print(result, flush=True)
                            continue
                        
                        try:
                            input_data = json.loads(json_input)
                            window = input_data['window']
                            prediction = predictor.predict(window)
                            result = json.dumps({"prediction": prediction})
                            print(result, flush=True)
                        except Exception as e:
                            import traceback
                            error_msg = json.dumps({"error": f"Prediction failed: {str(e)}", "traceback": traceback.format_exc()})
                            print(error_msg, flush=True)
                            sys.stderr.flush()
                    else:
                        # 未知命令，忽略
                        pass
                except Exception as e:
                    import traceback
                    error_msg = json.dumps({"error": f"Interactive mode error: {str(e)}", "traceback": traceback.format_exc()})
                    print(error_msg, flush=True)
                    sys.stderr.flush()
                    break
            
        except Exception as e:
            import traceback
            error_msg = json.dumps({"error": f"Interactive mode failed: {str(e)}", "traceback": traceback.format_exc()})
            print(error_msg, file=sys.stderr)
            sys.stderr.flush()
            sys.exit(1)
        
        sys.exit(0)
    
    # 读取输入：支持从文件或stdin读取
    try:
        if len(sys.argv) >= 3:
            # 从文件读取
            input_file = sys.argv[2]
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
            individual = input_data.get('individual', True)
            epochs = input_data.get('epochs', 100)
            learning_rate = input_data.get('learning_rate', 0.001)
            model_dir = input_data.get('model_dir', './models')
            
            print(f"开始训练模型: seq_len={seq_len}, pred_len={pred_len}, epochs={epochs}, data_size={len(data)}", file=sys.stderr)
            
            predictor = DLinearPredictor(seq_len=seq_len, pred_len=pred_len, 
                                        individual=individual, model_dir=model_dir)
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
            individual = input_data.get('individual', True)
            
            predictor = DLinearPredictor(seq_len=seq_len, pred_len=pred_len,
                                        individual=individual, model_dir=model_dir)
            
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
