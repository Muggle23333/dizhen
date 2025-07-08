import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

def get_model_size(model):
    """
    计算并打印模型参数数量
    支持Keras模型和scikit-learn模型
    """
    try:
        # 尝试作为Keras/TensorFlow模型处理
        if hasattr(model, 'count_params'):
            total_params = model.count_params()
            trainable_params = sum([w.numpy().size for w in model.trainable_weights])
            non_trainable_params = total_params - trainable_params
            
            print("\n模型参数统计:")
            print(f"总参数数量: {total_params/1e6:.2f}M")
            print(f"可训练参数: {trainable_params/1e6:.2f}M")
            print(f"不可训练参数: {non_trainable_params/1e6:.2f}M")
            return total_params
            
        # 处理scikit-learn模型(如MLPClassifier)
        elif hasattr(model, 'coefs_'):
            total_params = 0
            if hasattr(model, 'coefs_'):
                for layer in model.coefs_:
                    total_params += layer.size
            if hasattr(model, 'intercepts_'):
                for intercept in model.intercepts_:
                    total_params += intercept.size
                    
            print(f"\n模型参数数量: {total_params/1e6:.2f}M")
            return total_params
            
        # 处理PyTorch模型(如您提供的示例)
        elif hasattr(model, 'parameters'):
            n_params = sum(p.numel() for p in model.parameters())
            print(f"\n模型参数数量: {n_params/1e6:.2f}M")
            return n_params
            
        else:
            print("\n警告: 无法确定模型参数数量 - 模型类型不被支持")
            return 0
            
    except Exception as e:
        print(f"\n参数计算错误: {str(e)}")
        return 0

def load_model_with_size(fpath):
    """
    加载模型并计算其大小
    """
    try:
        # 尝试加载Keras模型
        if fpath.endswith('.h5') or fpath.endswith('.keras'):
            model = load_model(fpath)
            print(f"成功加载Keras模型: {fpath}")
        # 尝试加载scikit-learn模型
        elif fpath.endswith('.pkl') or fpath.endswith('.joblib'):
            model = joblib.load(fpath)
            print(f"成功加载scikit-learn模型: {fpath}")
        else:
            raise ValueError("不支持的模型格式")
            
        # 计算并显示模型大小
        get_model_size(model)
        return model
        
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return None

def main():
    # [原有数据加载和预处理代码保持不变...]
    
    # 修改后的模型加载部分
    model_path = 'best_model.h5'  # 可以是.pkl或.h5
    
    # 加载模型并获取大小信息
    model = load_model_with_size(model_path)
    if model is None:
        print("无法加载模型，程序终止")
        return
    
    # [原有预测和评估代码保持不变...]
    
    # 在最终输出中添加模型大小信息
    print("\n=== 性能总结 ===")
    # [原有性能指标输出...]
    print("模型参数数量已在上方显示")

if __name__ == "__main__":
    main()