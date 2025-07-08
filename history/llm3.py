# 少了最大长度，但是也一点可视化都没有了
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import Sequence
import matplotlib.pyplot as plt
import joblib
from collections import Counter
import tensorflow as tf

# 配置
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class EarthquakeDataGenerator(Sequence):
    """自定义数据生成器处理变长序列"""
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
        batch_X = [self.X[i] for i in batch_indices]
        batch_y = [self.y[i] for i in batch_indices]
        
        # 动态批处理：按当前批次最大长度处理
        max_len = max(len(x) for x in batch_X)
        X_batch = np.zeros((len(batch_X), max_len, 1))
        y_batch = np.zeros((len(batch_y), max_len, 1))
        
        for i, (x, y) in enumerate(zip(batch_X, batch_y)):
            X_batch[i, :len(x), 0] = x
            y_batch[i, :len(y), 0] = y
            
        return X_batch, y_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def load_raw_sequences(folder_path):
    """加载原始变长序列数据"""
    sequences = []
    labels = []
    
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            data = pd.read_csv(os.path.join(folder_path, file), header=None).values
            sequences.append(data[:, 0].astype('float32'))
            labels.append(data[:, 1].astype('float32'))
    
    return sequences, labels

def normalize_sequences(sequences, mode='train'):
    """对变长序列进行归一化"""
    if mode == 'train':
        # 合并所有序列点计算统计量
        all_values = np.concatenate(sequences)
        scaler = StandardScaler().fit(all_values.reshape(-1, 1))
        joblib.dump(scaler, 'scaler.pkl')
    else:
        scaler = joblib.load('scaler.pkl')
    
    return [scaler.transform(seq.reshape(-1, 1)).flatten() for seq in sequences]

def build_dynamic_lstm_model():
    """构建支持变长输入的LSTM模型"""
    model = Sequential([
        Masking(mask_value=0., input_shape=(None, 1)),  # 自动处理变长序列
        LSTM(128, return_sequences=True),
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

def train_with_variable_length():
    # 1. 加载原始变长数据
    X, y = load_raw_sequences(r'test')
    print(f"加载完成 - 样本数: {len(X)}")
    print(f"序列长度示例: {[len(x) for x in X[:5]]}")
    
    # 2. 数据归一化
    X_normalized = normalize_sequences(X, mode='train')
    
    # 3. 分割数据集
    X_train, X_val, y_train, y_val = train_test_split(
        X_normalized, y, test_size=0.2, random_state=42
    )
    
    # 4. 创建数据生成器
    train_generator = EarthquakeDataGenerator(X_train, y_train, batch_size=32)
    val_generator = EarthquakeDataGenerator(X_val, y_val, batch_size=32, shuffle=False)
    
    # 5. 构建模型
    model = build_dynamic_lstm_model()
    model.summary()
    
    # 6. 训练配置
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('dynamic_lstm_model.h5', monitor='val_loss', save_best_only=True)
    ]
    
    # 7. 模型训练
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=5,
        callbacks=callbacks,
        verbose=1
    )
    
    # 8. 评估
    evaluate_variable_length_model(model, val_generator)

def evaluate_variable_length_model(model, generator):
    """评估变长序列模型"""
    print("\n验证集评估:")
    results = model.evaluate(generator, verbose=0)
    metrics = {
        'loss': results[0],
        'accuracy': results[1],
        'precision': results[2],
        'recall': results[3],
        'f1_score': 2 * (results[2] * results[3]) / (results[2] + results[3] + 1e-7)
    }
    
    for name, value in metrics.items():
        print(f"{name:>10}: {value:.4f}")

if __name__ == "__main__":
    train_with_variable_length()