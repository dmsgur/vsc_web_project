#lstm_and_conv.py
import sys

import matplotlib.pyplot as plt
import numpy as np  # numpy 1.26.4
import tensorflow as tf  # tensorflow-cpu 2.10.0
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, LSTM, ConvLSTM1D, Dropout, Reshape, MaxPool1D, BatchNormalization

#from train_model import receive_data,preData,split_xyData,recovery_info

np.random.seed(123)
tf.random.set_seed(123)
# == 동일 버전 ==
print("python -v ",sys.version)
print("tensorflow-cpu -v",tf.__version__)
print("numpy -v ",np.__version__)
def createModel_conv(pred_step):
    if pred_step=="short":outputsize=30
    elif pred_step=="long":outputsize=90
    elif pred_step=="llong":outputsize=180
    else : outputsize=60
    conv_model = Sequential()
    conv_model.add(Input((outputsize,5)))
    conv_model.add(Reshape((outputsize,5,1)))
    conv_model.add(BatchNormalization())
    conv_model.add(ConvLSTM1D(
    16,3, strides=1,padding='same',return_sequences=True,go_backwards=True))
    conv_model.add(ConvLSTM1D(
    32,5, strides=1,padding='same',dropout=0.3,recurrent_dropout=0.2,
    return_sequences=True,go_backwards=True))
    conv_model.add(ConvLSTM1D(
    64,5, strides=1,padding='same',dropout=0.3,recurrent_dropout=0.2,
    return_sequences=False))
    conv_model.add(MaxPool1D(pool_size=4,strides=1,padding="same"))
    conv_model.add(BatchNormalization())
    conv_model.add(Dropout(0.4))
    conv_model.add(Dense(512,activation="relu"))
    conv_model.add(Dense(256,activation="relu"))
    conv_model.add(Dropout(0.3))
    conv_model.add(Dense(64,activation="relu"))
    conv_model.add(Dense(1,activation="sigmoid"))
    conv_model.add(Reshape((-1,)))
    conv_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                       optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005,beta_1=0.5),
                       metrics=["MAE"])
    return conv_model


#모델 구성 및 훈련 최적화 모델 저장
def createModel_lstm(pred_step):
    if pred_step=="short":outputsize=30
    elif pred_step=="long":outputsize=90
    elif pred_step=="llong":outputsize=180
    else : outputsize=60
    lstm_model = Sequential()
    lstm_model.add(Input((outputsize,5)))
    lstm_model.add(LSTM(
        256,
        dropout=0.4,
        recurrent_dropout=0.3,
        seed=123,
        return_sequences=True,
        go_backwards=True,
    ))
    lstm_model.add(LSTM(
        128,
        dropout=0.4,
        recurrent_dropout=0.3,
        seed=123,
        return_sequences=True,
        go_backwards=True,
    ))
    lstm_model.add(LSTM(
        64,
        seed=123
    ))
    lstm_model.add(Dense(128,activation="relu"))
    lstm_model.add(Dropout(0.4))
    lstm_model.add(Dense(32,activation="relu"))
    lstm_model.add(Dropout(0.3))
    lstm_model.add(Dense(5,activation="sigmoid"))
    lstm_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                       optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005,beta_1=0.5),
                       metrics=["MAE"])
    return lstm_model
def createCallback(coinname,modeltype=None):
    #'./lstmsave/BTC/2025-05-13'
    #./lstmsave/BTC/2025-05-13/lstm_{epoch:02d}-{val_loss:.2f}.keras
    # paths ="./%s/%s/%s"%(modeltype+"save",coinname,date.today())
    # if not os.path.exists(paths):
    #     os.makedirs(paths)#여러개의 디렉토리 생성
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',verbose=1,\
                                          patience=50,mode='min', \
                                          restore_best_weights=True)
    # mcp = tf.keras.callbacks.ModelCheckpoint(
    #     "./%s/%s/%s/%s_{epoch:02d}-{val_loss:.2f}.keras"%\
    #     (modeltype+"save",coinname,date.today(),modeltype),
    #     monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
    return [es]
