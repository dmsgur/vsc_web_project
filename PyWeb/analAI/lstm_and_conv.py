import os
import sys
from datetime import date
import tensorflow as tf # tensorflow-cpu 2.10.0
import numpy as np # numpy 1.26.4
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential,Model,Input
from tensorflow.keras.layers import Dense,LSTM,ConvLSTM1D
from train_model import receive_data,preData,split_xyData,recovery_info
# == 동일 버전 ==
print("python -v ",sys.version)
print("tensorflow-cpu -v",tf.__version__)
print("numpy -v ",np.__version__)
#모델 구성 및 훈련 최적화 모델 저장
def createModel_lstm(outputsize):
    lstm_model = Sequential()
    lstm_model.add(Input((5,5)))
    lstm_model.add(LSTM(
        128,
        dropout=0.3,
        recurrent_dropout=0.2,
        seed=123,
        return_sequences=True,
        go_backwards=True,
    ))
    lstm_model.add(LSTM(
        64,
        dropout=0.2,
        seed=123
    ))
    lstm_model.add(Dense(128,activation="relu"))
    lstm_model.add(Dense(64,activation="relu"))
    lstm_model.add(Dense(32,activation="relu"))
    lstm_model.add(Dense(outputsize,activation="sigmoid"))
    lstm_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                       optimizer=tf.keras.optimizers.Adam())
    return lstm_model
def createCallback(coinname,modeltype):
    #'./lstmsave/BTC/2025-05-13'
    #./lstmsave/BTC/2025-05-13/lstm_{epoch:02d}-{val_loss:.2f}.keras
    paths ="./%s/%s/%s"%(modeltype+"save",coinname,date.today())
    if not os.path.exists(paths):
        os.makedirs(paths)#여러개의 디렉토리 생성
    print("./%s/%s/%s/%s_{epoch:02d}-{val_loss:.2f}.keras"%\
        (modeltype+"save",coinname,date.today(),modeltype))

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=30,
                                                      mode='min')
    mcp = tf.keras.callbacks.ModelCheckpoint(
        "./%s/%s/%s_{epoch:02d}-{val_loss:.2f}.keras"%\
        (modeltype+"save",coinname,modeltype),
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        mode='auto')
    return [es,mcp]
def drawGraph(losses,val_loss):
    pass

if "__main__"==__name__:
    user_count=1000
    user_timestep=30
    data_sets, target_name, req_time = receive_data(req_time=1, getcnt=user_count)
    print("수신된 데이터: 수량", len(data_sets), \
          " 이름:", target_name, " 시간대:", req_time)
    print("현재가격:", data_sets[-1]["trade_price"])
    pre_datasets, recovery_price = preData(data_sets)  # 정규화가격정보,복구가격편차및평균
    x_data, y_data = split_xyData(pre_datasets, user_timestep)
    print(x_data.max())
    print(y_data.max())
    print(x_data.min())
    print(y_data.min())
    # 데이터 정합성 검증
    print(x_data.shape, y_data.shape)
    print(y_data[0][0] == x_data[1][4][0])
    print(y_data[-2][0] == x_data[-1][4][0])
    # 가격복구 검증
    recprice = recovery_info(y_data[:1], recovery_price)
    print("가격복구정보", recprice)
    lstm_model = createModel_lstm(user_timestep)
    cbs = createCallback(target_name, "lstm")
    fhist = lstm_model.fit(x_data,y_data,epochs=100,validation_split=0.1,batch_size=user_count//30,
                           callbacks=cbs)
    # print("==========",os.getcwd())


