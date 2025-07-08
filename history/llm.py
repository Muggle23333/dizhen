import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score
import joblib
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

#test_folder_path = r'/home/lyj/100'  # 替换为您的实际路径
test_folder_path = r'test'
def load_csv_files(folder_path):
    """加载文件夹中的所有CSV文件路径"""
    csv_files = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            csv_files.append(os.path.join(folder_path, file))
    return csv_files

def read_csv_file(file_path):
    """读取CSV文件并返回数据"""
    data = pd.read_csv(file_path, header=None)
    return data.values

def preprocess_data(data, max_length=1000):
    """预处理数据：归一化并填充到相同长度"""
    X = []  # 特征
    y = []  # 标签

    for file_data in data:
        # 确保数据至少有两列
        if file_data.shape[1] < 2:
            print(f"警告: 文件数据只有{file_data.shape[1]}列，跳过")
            continue
            
        amplitudes = file_data[:, 0]
        labels = file_data[:, 1]
        
        X.append(amplitudes)
        y.append(labels)

    # 填充序列到相同长度
    X_padded = pad_sequences(X, maxlen=max_length, padding='post', truncating='post', dtype='float32')
    y_padded = pad_sequences(y, maxlen=max_length, padding='post', truncating='post', dtype='float32')
    
    # 特征数据归一化
    scaler = StandardScaler()
    
    # 重塑数据以适应StandardScaler
    original_shape = X_padded.shape
    X_reshaped = X_padded.reshape(-1, original_shape[1])
    X_normalized = scaler.fit_transform(X_reshaped)
    X_normalized = X_normalized.reshape(original_shape)
    
    return X_normalized, y_padded

def calculate_metrics(y_true, y_pred):
    """计算准确率、精确率等指标"""
    # 展平数组以便计算
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # 计算指标
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    
    return accuracy, precision

def plot_sample(sample_data, true_labels, pred_labels, index, metrics):
    """绘制单个样本的可视化"""
    # 绘制原始数据与预测事件
    plt.figure(figsize=(14, 7))
    plt.plot(sample_data, label='地震振幅数据')
    
    # 绘制预测事件
    predicted_events = np.where(pred_labels == 1)[0]
    plt.plot(predicted_events, sample_data[predicted_events], 'ro', markersize=3, label='预测事件')
    
    # 绘制真实事件
    true_events = np.where(true_labels == 1)[0]
    plt.scatter(true_events, sample_data[true_events], color='green', marker='x', label='真实事件')
    
    plt.xlabel('时间点')
    plt.ylabel('归一化振幅')
    plt.title(f'样本 {index} - 地震事件检测 (准确率: {metrics[0]:.2f}, 精确率: {metrics[1]:.2f})')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 绘制标签对比
    plt.figure(figsize=(14, 7))
    plt.plot(true_labels, 'g-', label='真实标签')
    plt.plot(pred_labels, 'r--', label='预测标签')
    plt.xlabel('时间点')
    plt.ylabel('事件标签 (0/1)')
    plt.title(f'样本 {index} - 预测与实际标签对比')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # 加载测试数据
    csv_files = load_csv_files(test_folder_path)
    if not csv_files:
        print(f"错误: 在 {test_folder_path} 中没有找到CSV文件")
        return
    
    test_data = [read_csv_file(file) for file in csv_files]
    if not test_data:
        print("错误: 没有加载到有效数据")
        return
    
    # 预处理数据
    try:
        X_test, y_test = preprocess_data(test_data)
        print(f"数据预处理完成，形状: X_test={X_test.shape}, y_test={y_test.shape}")
    except Exception as e:
        print(f"预处理错误: {e}")
        return
    
    # 加载模型
    try:
        # 尝试加载Keras模型
        model = load_model('LSTM_model.h5')
        print("Keras模型加载成功")
    except:
        try:
            # 尝试加载sklearn模型
            model = joblib.load('LSTM_model.pkl')
            print("Scikit-learn模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            return
    
    # 模型预测
    try:
        # 根据模型类型调整预测方法
        if hasattr(model, 'predict_classes'):  # 旧版Keras
            y_pred = model.predict_classes(X_test)
        elif hasattr(model, 'predict'):  # 新版Keras或sklearn
            if hasattr(model, 'predict_proba'):  # 概率输出模型
                y_pred_proba = model.predict_proba(X_test)
                if len(y_pred_proba.shape) > 1:  # 二分类问题
                    y_pred = (y_pred_proba[:, 1] > 0.42).astype(int)
                else:  # 单输出
                    y_pred = (y_pred_proba > 0.42).astype(int)
            else:  # 直接输出类别的模型
                y_pred = model.predict(X_test)
        else:
            print("错误: 无法识别的模型类型")
            return
            
        # 确保预测结果形状正确
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
    except Exception as e:
        print(f"预测错误: {e}")
        return
    
    # 计算每个样本的指标
    accuracies = []
    precisions = []
    
    for i in range(len(y_test)):
        # 确保样本长度匹配
        min_length = min(len(y_test[i]), len(y_pred[i]))
        sample_y_test = y_test[i][:min_length]
        sample_y_pred = y_pred[i][:min_length]
        
        accuracy, precision = calculate_metrics(sample_y_test, sample_y_pred)
        accuracies.append(accuracy)
        precisions.append(precision)
    
    # 找出准确率最高的20个样本
    if len(accuracies) < 20:
        top_indices = range(len(accuracies))
        print(f"警告: 样本数量不足20，只显示{len(accuracies)}个样本")
    else:
        top_indices = np.argsort(accuracies)[-20:]
    
    # 可视化最佳样本
    for i, index in enumerate(top_indices):
        # 确保样本长度匹配
        min_length = min(len(X_test[index]), len(y_test[index]), len(y_pred[index]))
        sample_data = X_test[index][:min_length]
        sample_true_labels = y_test[index][:min_length]
        sample_pred_labels = y_pred[index][:min_length]
        
        metrics = (accuracies[index], precisions[index])
        plot_sample(sample_data, sample_true_labels, sample_pred_labels, i, metrics)
    
    # 计算平均指标
    avg_accuracy = np.mean([accuracies[i] for i in top_indices])
    avg_precision = np.mean([precisions[i] for i in top_indices])
    
    print("\n整体性能摘要:")
    print(f"平均准确率: {avg_accuracy:.4f}")
    print(f"平均精确率: {avg_precision:.4f}")

if __name__ == "__main__":
    main()