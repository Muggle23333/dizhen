from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking
import tensorflow as tf
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