import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import joblib
from matplotlib.gridspec import GridSpec
from collections import Counter
from sklearn.preprocessing import StandardScaler
from keras.utils import Sequence

def plot_dataset_split(X_train, X_val, y_train, y_val):
    """可视化数据集划分情况"""
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # 样本长度分布
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist([len(x) for x in X_train], bins=30, alpha=0.7, label='训练集')
    ax1.hist([len(x) for x in X_val], bins=30, alpha=0.7, label='验证集')
    ax1.set_title('序列长度分布对比')
    ax1.set_xlabel('序列长度')
    ax1.set_ylabel('频数')
    ax1.legend()
    
    # 标签分布
    ax2 = fig.add_subplot(gs[0, 1:])
    train_counts = Counter(np.concatenate(y_train).flatten())
    val_counts = Counter(np.concatenate(y_val).flatten())
    ax2.bar(train_counts.keys(), train_counts.values(), alpha=0.7, label='训练集')
    ax2.bar(val_counts.keys(), val_counts.values(), alpha=0.7, label='验证集')
    ax2.set_title('标签类别分布对比')
    ax2.set_xlabel('类别')
    ax2.set_ylabel('数量')
    ax2.legend()
    
    # 样本示例可视化
    ax3 = fig.add_subplot(gs[1, :])
    #sample_idx = np.random.randint(0, len(X_train))
    sample_idx = 10
    ax3.plot(X_train[sample_idx], label='地震波形')
    event_points = np.where(y_train[sample_idx] == 1)[0]
    ax3.scatter(event_points, X_train[sample_idx][event_points], 
               color='red', label='地震事件')
    ax3.set_title(f'训练集样本示例 (ID: {sample_idx})')
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('振幅')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('dataset_split.png', dpi=300)
    plt.show()


def load_raw_sequences(folder_path):
    """加载原始变长序列数据"""
    sequences = []
    labels = []
    
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            data = pd.read_csv(os.path.join(folder_path, file), header=None).values
            numeric_data = pd.to_numeric(data[:, 0], errors='coerce')  # 非数值转为 NaN
            sequences.append(numeric_data[~pd.isna(numeric_data)].astype('float32'))  # 只保留有效数值
            numeric_data = pd.to_numeric(data[:, 1], errors='coerce')  # 非数值转为 NaN
            labels.append(numeric_data[~pd.isna(numeric_data)].astype('float32'))  # 只保留有效数值
    
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