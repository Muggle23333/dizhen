# 集合分割不太自然
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import joblib
from collections import Counter

# 设置字体和样式
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径配置
TRAIN_FOLDER = r'test'
TEST_FOLDER = 'test'
MAX_LENGTH = 10000  # 最大序列长度

def load_and_preprocess_data(folder_path, mode='train'):
    """加载并预处理数据（支持训练/测试模式）"""
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not csv_files:
        raise ValueError(f"在 {folder_path} 中没有找到CSV文件")
    
    data = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, header=None).values
        if df.shape[1] >= 2:  # 确保有特征和标签
            data.append(df)
    
    # 填充序列
    X = pad_sequences([d[:, 0] for d in data], 
                     maxlen=MAX_LENGTH, 
                     padding='post', 
                     truncating='post',
                     dtype='float32')
    y = pad_sequences([d[:, 1] for d in data], 
                     maxlen=MAX_LENGTH, 
                     padding='post', 
                     truncating='post',
                     dtype='float32')
    
    # 训练模式下进行归一化并保存scaler
    if mode == 'train':
        scaler = StandardScaler()
        X = scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
        joblib.dump(scaler, 'scaler.pkl')
    else:
        scaler = joblib.load('scaler.pkl')
        X = scaler.transform(X.reshape(-1, 1)).reshape(X.shape)
    
    return X, y

def split_dataset(X, y, strategy='sample', test_size=0.2, random_state=42):
    """
    数据集分割策略
    :param strategy: 'sample'按样本分割 / 'timestep'按时间步分割
    :return: X_train, X_val, y_train, y_val
    """
    if strategy == 'sample':
        # 常规样本级分割（保持样本完整）
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    elif strategy == 'timestep':
        # 时间步级分割（适合长时间序列）
        split_idx = int(X.shape[1] * (1 - test_size))
        X_train, X_val = X[:, :split_idx], X[:, split_idx:]
        y_train, y_val = y[:, :split_idx], y[:, split_idx:]
        return X_train, X_val, y_train, y_val
    
    else:
        raise ValueError("请选择'sample'或'timestep'分割策略")

def plot_data_distribution(y_train, y_val):
    """可视化训练集/验证集的标签分布"""
    train_counts = Counter(y_train.flatten())
    val_counts = Counter(y_val.flatten())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 训练集分布
    ax1.bar(train_counts.keys(), train_counts.values(), color='royalblue')
    ax1.set_title('训练集标签分布')
    ax1.set_xlabel('类别')
    ax1.set_ylabel('数量')
    
    # 验证集分布
    ax2.bar(val_counts.keys(), val_counts.values(), color='orange')
    ax2.set_title('验证集标签分布')
    ax2.set_xlabel('类别')
    
    plt.tight_layout()
    plt.show()

def build_lstm_model(input_shape):
    """构建双向LSTM模型"""
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall')])
    return model

def train_model():
    # 1. 数据加载与预处理
    X, y = load_and_preprocess_data(TRAIN_FOLDER)
    print(f"数据加载完成 - 特征形状: {X.shape}, 标签形状: {y.shape}")
    
    # 2. 数据集分割（可选择策略）
    X_train, X_val, y_train, y_val = split_dataset(
        X, y, 
        strategy='sample',  # 尝试改为'timestep'对比效果
        test_size=0.2
    )
    print(f"\n数据集分割结果:")
    print(f"训练集: {X_train.shape} | 验证集: {X_val.shape}")
    
    # 3. 可视化数据分布
    plot_data_distribution(y_train, y_val)
    
    # 4. 模型构建
    model = build_lstm_model((X_train.shape[1], 1))
    model.summary()
    
    # 5. 训练配置
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ModelCheckpoint('best_lstm_model.h5', monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
    ]
    
    # 6. 模型训练
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )
    
    # 7. 训练过程可视化
    plot_training_curves(history)
    
    # 8. 模型评估
    evaluate_model(model, X_val, y_val)

def plot_training_curves(history):
    """绘制训练指标曲线"""
    metrics = ['loss', 'accuracy', 'precision', 'recall']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        ax.plot(history.history[metric], label=f'训练{metric}')
        ax.plot(history.history[f'val_{metric}'], label=f'验证{metric}')
        ax.set_title(f'{metric}曲线')
        ax.set_xlabel('Epoch')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_val, y_val):
    """模型性能评估"""
    print("\n验证集评估:")
    results = model.evaluate(X_val, y_val, verbose=0)
    metrics = {
        'loss': results[0],
        'accuracy': results[1],
        'precision': results[2],
        'recall': results[3],
        'f1_score': 2 * (results[2] * results[3]) / (results[2] + results[3] + 1e-7)
    }
    
    for name, value in metrics.items():
        print(f"{name:>10}: {value:.4f}")
    
    # 保存评估结果
    pd.DataFrame([metrics]).to_csv('model_performance.csv', index=False)

if __name__ == "__main__":
    import tensorflow as tf
    print(f"TensorFlow版本: {tf.__version__}")
    print(f"GPU可用: {tf.config.list_physical_devices('GPU')}")
    
    train_model()