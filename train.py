import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import joblib
from collections import Counter
import tensorflow as tf
from matplotlib.gridspec import GridSpec
from keras.utils import Sequence

from dataset import plot_dataset_split,load_raw_sequences,normalize_sequences,EarthquakeDataGenerator
from model import build_dynamic_lstm_model
# 配置
plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_training_curves(history):
    """绘制训练过程中的各项指标曲线"""
    metrics = ['loss', 'accuracy', 'precision', 'recall']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        ax.plot(history.history[metric], linewidth=2, label=f'训练{metric}')
        ax.plot(history.history[f'val_{metric}'], linewidth=2, label=f'验证{metric}')
        ax.set_title(f'{metric}变化曲线', fontsize=14)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=12)
        
        # 标记最佳值
        if 'val_' + metric in history.history:
            best_epoch = np.argmin(history.history['val_loss']) if metric == 'loss' else np.argmax(history.history[f'val_{metric}'])
            best_value = history.history[f'val_{metric}'][best_epoch]
            ax.axvline(best_epoch, color='r', linestyle='--', alpha=0.3)
            ax.annotate(f'最佳: {best_value:.4f}', 
                       xy=(best_epoch, best_value),
                       xytext=(10, 10), 
                       textcoords='offset points',
                       bbox=dict(boxstyle='round', fc='w'))
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    plt.show()
    

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


def train_model():
    # 1. 数据加载与预处理
    sequences, labels = load_raw_sequences(r'train_data')
    X = normalize_sequences(sequences)
    y = labels
    
    # 2. 数据集划分
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=[l[0] for l in labels]  # 按初始标签分层
    )
    
    # 3. 可视化数据集划分
    print("\n数据集划分统计:")
    print(f"训练集样本数: {len(X_train)} | 验证集样本数: {len(X_val)}")
    print(f"训练集平均长度: {np.mean([len(x) for x in X_train]):.1f}±{np.std([len(x) for x in X_train]):.1f}")
    print(f"验证集平均长度: {np.mean([len(x) for x in X_val]):.1f}±{np.std([len(x) for x in X_val]):.1f}")
    plot_dataset_split(X_train, X_val, y_train, y_val)
    
    # 4. 创建数据生成器
    train_gen = EarthquakeDataGenerator(X_train, y_train, batch_size=32)
    val_gen = EarthquakeDataGenerator(X_val, y_val, batch_size=32, shuffle=False)
    
    # 5. 模型构建
    model = build_dynamic_lstm_model()
    
    # 6. 训练配置
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, verbose=1),
        ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.CSVLogger('training_log.csv')
    ]
    
    # 7. 模型训练
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=2,
        callbacks=callbacks,
        verbose=1
    )
    
    # 8. 绘制训练曲线
    plot_training_curves(history)
    
    # 9. 最终评估
    print("\n最终模型评估:")
    model = tf.keras.models.load_model('best_model.h5')
    evaluate_variable_length_model(model, val_gen)

if __name__ == "__main__":
    # 检查GPU可用性
    print("="*50)
    print(f"TensorFlow 版本: {tf.__version__}")
    print(f"GPU 可用: {len(tf.config.list_physical_devices('GPU')) > 0}")
    if tf.config.list_physical_devices('GPU'):
        print(f"GPU 设备: {tf.test.gpu_device_name()}")
    print("="*50)
    
    train_model()