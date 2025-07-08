import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import random
from matplotlib.font_manager import FontProperties

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
test_folder_path = r'test'
def load_csv_files(folder_path):
    csv_files = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            csv_files.append(os.path.join(folder_path, file))
    return csv_files

# 读取文件
def read_csv_file(file_path):
    data = pd.read_csv(file_path, header=None)
    return data.values
def preprocess_data(data):
    X = []  # 特征
    y = []  # 标签

    for file_data in data:
        amplitudes = file_data[:, 0]
        labels = file_data[:, 1]
        
        X.append(amplitudes)
        y.append(labels)

    # 特征数据归一化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y
# 从指定文件夹加载数据
def load_data(folder_path):
    csv_files = load_csv_files(folder_path)
    data = [read_csv_file(file) for file in csv_files]
    return data
test_data = load_data(test_folder_path)
# 对测试集进行预处理
X_test, y_test = preprocess_data(test_data)
import joblib

# 加载模型
model =joblib.load('LSTM_model.pkl')
model.summary()
# 模型评估
y_pred = (model.predict(X_test) > 0.42).astype("int32")

import numpy as np
import matplotlib.pyplot as plt

# 函数用于计算单个样本的准确率
def calculate_accuracy(y_true, y_pred):
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    true_negatives = np.sum((y_pred == 0) & (y_true == 0))
    false_positives = np.sum((y_pred == 1) & (y_true == 0))
    false_negatives = np.sum((y_pred == 0) & (y_true == 1))
    
    accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
    return accuracy

# 计算每个样本的准确率
accuracies = [calculate_accuracy(y_test[i], y_pred[i]) for i in range(len(y_test))]

# 找出准确率最高的20个样本的索引
top_indices = np.argsort(accuracies)[-20:]

# 可视化准确率最高的20个样本
for index in top_indices:
    sample_data = X_test[index].flatten()
    sample_labels = y_test[index]
    sample_pred = y_pred[index]

    # 绘制原始数据与去噪后的数据
    plt.figure(figsize=(14, 7))
    plt.plot(sample_data, label='处理后数据')

    predicted_events = np.where(sample_pred == 1)[0]
    plt.plot(predicted_events, sample_data[predicted_events], 'o', markersize=3, color='red', label='预测事件')

    true_events = np.where(sample_labels == 1)[0]
    plt.scatter(true_events, sample_data[true_events], color='green', label='真实事件')

    plt.legend()
    plt.title('地震迹线标签图像')
    plt.show()

    # 绘制预测与真实标签
    plt.figure(figsize=(14, 7))
    plt.scatter(range(len(sample_labels)), sample_labels, c='g', label='真实值')

    predicted_one_indices = np.where(sample_pred == 1)[0]
    predicted_zero_indices = np.where(sample_pred == 0)[0]

    plt.scatter(predicted_one_indices, sample_pred[predicted_one_indices], c='r', s=0.7, label='预测标签 (1)')
    plt.scatter(predicted_zero_indices, sample_pred[predicted_zero_indices], c='y', s=0.7, label='预测标签 (0)')

    plt.legend()
    plt.title('预测与实际标签的对比')
    plt.show()

    # 计算精度和准确率
    predicted_positives = (sample_pred == 1)
    true_positives = np.sum((sample_pred == 1) & (sample_labels == 1))
    actual_positives = np.sum(sample_labels == 1)
    true_negatives = np.sum((sample_pred == 0) & (sample_labels == 0))
    false_positives = np.sum(predicted_positives) - true_positives
    false_negatives = actual_positives - true_positives

    precision = true_positives / (true_positives + false_positives)
    accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)

    print(f"样本 {index} Accuracy: {accuracy:.4f}")
    print(f"样本 {index} Precision: {precision:.4f}\n")
# 计算平均精确率
average_precision = np.mean(precision)

# 计算平均准确率
average_accuracy = np.mean([accuracies[i] for i in top_indices])

print(f"二十个样本的精确率: {average_precision:.4f}")
print(f"二十个样本的准确率: {average_accuracy:.4f}")
