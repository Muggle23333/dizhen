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

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 训练数据路径
train_folder_path = 'train'  # 替换为您的训练数据路径

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
    
    # 保存归一化器，测试时使用
    joblib.dump(scaler, 'scaler.pkl')
    
    return X_normalized, y_padded

def build_lstm_model(input_shape):
    """构建LSTM模型"""
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
                 metrics=['accuracy'])
    
    return model

def plot_training_history(history):
    """绘制训练过程中的准确率和损失曲线"""
    plt.figure(figsize=(12, 5))
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('模型准确率')
    plt.ylabel('准确率')
    plt.xlabel('训练轮次')
    plt.legend()
    
    # 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.ylabel('损失值')
    plt.xlabel('训练轮次')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # 加载训练数据
    csv_files = load_csv_files(train_folder_path)
    if not csv_files:
        print(f"错误: 在 {train_folder_path} 中没有找到CSV文件")
        return
    
    train_data = [read_csv_file(file) for file in csv_files]
    if not train_data:
        print("错误: 没有加载到有效数据")
        return
    
    # 预处理数据
    try:
        X, y = preprocess_data(train_data)
        print(f"数据预处理完成，形状: X={X.shape}, y={y.shape}")
    except Exception as e:
        print(f"预处理错误: {e}")
        return
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 构建模型
    input_shape = (X_train.shape[1], 1)  # (时间步长, 特征维度)
    model = build_lstm_model(input_shape)
    model.summary()
    
    # 设置回调函数
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, verbose=1),
        ModelCheckpoint('LSTM_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    # 训练模型
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 评估模型
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n验证集评估结果 - 损失: {loss:.4f}, 准确率: {accuracy:.4f}")

if __name__ == "__main__":
    main()