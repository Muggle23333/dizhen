
import os
import pandas as pd

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

a = load_raw_sequences('train_data')
print(a)
