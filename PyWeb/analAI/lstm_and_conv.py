import os
import sys
from datetime import date
import tensorflow as tf # tensorflow-cpu 2.10.0
import numpy as np # numpy 1.26.4
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential,Model,Input
from tensorflow.keras.layers import Dense,LSTM,ConvLSTM1D,Dropout,Reshape,MaxPool1D,BatchNormalization
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
    conv_model.add(ConvLSTM1D(
    16,3, strides=1,padding='same',dropout=0.3,recurrent_dropout=0.2,
    return_sequences=True,go_backwards=True))
    # conv_model.add(ConvLSTM1D(
    # 32,5, strides=1,padding='same',dropout=0.3,recurrent_dropout=0.2,
    # return_sequences=True,go_backwards=True))
    conv_model.add(ConvLSTM1D(
    64,5, strides=1,padding='same',dropout=0.3,recurrent_dropout=0.2,
    return_sequences=False))
    conv_model.add(MaxPool1D(pool_size=2,strides=1,padding="same"))
    conv_model.add(BatchNormalization())
    conv_model.add(Dropout(0.5))
    conv_model.add(Dense(256,activation="relu"))
    conv_model.add(Dropout(0.5))
    conv_model.add(Dense(64,activation="relu"))
    conv_model.add(Dropout(0.4))
    conv_model.add(Dense(32,activation="relu"))
    conv_model.add(BatchNormalization())
    conv_model.add(Dropout(0.3))
    conv_model.add(Dense(1,activation="linear"))
    conv_model.add(Reshape((-1,)))
    conv_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                       optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005,beta_1=0.5),
                       metrics=["acc"])
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
    lstm_model.add(Dense(5,activation="linear"))
    lstm_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                       optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005,beta_1=0.5),
                       metrics=["acc"])
    return lstm_model
def createCallback(coinname,modeltype=None):
    #'./lstmsave/BTC/2025-05-13'
    #./lstmsave/BTC/2025-05-13/lstm_{epoch:02d}-{val_loss:.2f}.keras
    # paths ="./%s/%s/%s"%(modeltype+"save",coinname,date.today())
    # if not os.path.exists(paths):
    #     os.makedirs(paths)#여러개의 디렉토리 생성
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',\
                                          patience=30,mode='min', \
                                          restore_best_weights=True)
    # mcp = tf.keras.callbacks.ModelCheckpoint(
    #     "./%s/%s/%s/%s_{epoch:02d}-{val_loss:.2f}.keras"%\
    #     (modeltype+"save",coinname,date.today(),modeltype),
    #     monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
    return [es]
def drawGraph(losses,val_losses):
    plt.figure(figsize=(3,3))
    plt.plot(losses,label="loss")
    plt.plot(val_losses, label="valid_loss")
    plt.legend()
    plt.title("LOSSES")
    plt.show()
def drawPredict(y_predict,y_true):
    plt.scatter(y_true,y_predict)
    plt.plot(y_true,y_true)
    plt.xlabel("true")
    plt.ylabel("pred")
    plt.show()

if "__main__"==__name__:
    pass
    # user_count=1000
    # user_timestep=60
    # epoch=10
    # data_sets, target_name, req_time = receive_data(req_time=1, getcnt=user_count)
    # print("수신된 데이터: 수량", len(data_sets), \
    #       " 이름:", target_name, " 시간대:", req_time)
    # print("현재가격:", data_sets[-1]["trade_price"])
    # pre_datasets, recovery_price = preData(data_sets)  # 정규화가격정보,복구가격편차및평균
    # print("pre_datasets==")
    # print(pre_datasets.max())
    # x_data, y_data = split_xyData(pre_datasets, user_timestep)
    # print("pre_datasets==")
    # print(pre_datasets.max())
    #
    # # 데이터 정합성 검증
    # print(x_data.shape, y_data.shape)
    # print(y_data[0][0] == x_data[1][4][0])
    # print(y_data[-2][0] == x_data[-1][4][0])
    # print(x_data.max())
    # print(y_data.max())
    # print(x_data.min())
    # print(y_data.min())
    # # lstm_model = createModel_lstm(user_timestep)
    # # cbs = createCallback(target_name, "lstm")
    # print("데이터정보")
    # print(x_data.max())
    # print((y_data>1).sum())
    # print(x_data.min())
    # print(y_data.min())
    # # fhist = lstm_model.fit(x_data,y_data,epochs=epoch,validation_data=(x_data,y_data),batch_size=user_count//30,
    # #                        callbacks=cbs)
    # # drawGraph(fhist.history["loss"], fhist.history["val_loss"])
    # # y_pred = lstm_model.predict(x_data)
    # # drawPredict(y_pred, y_data)
    # # print("==========",os.getcwd())
    # conv_model=createModel_conv(user_timestep)
    # cbs = createCallback(target_name,"conv")
    # fhist = conv_model.fit(x_data,y_data,epochs=epoch,validation_data=(x_data,y_data),batch_size=user_count//30,
    #                         callbacks=cbs)
    # drawGraph(fhist.history["loss"], fhist.history["val_loss"])
    # y_pred = conv_model.predict(x_data)
    # drawPredict(y_pred, y_data)
    # recprice=recovery_info(y_pred[-5:],recovery_price)
    # print(recprice)
    # #['opening_price'],['high_price'],['low_price'],['candle_acc_trade_volume'],['trade_price']
    # print(recprice[-1][0])
    # print(recprice[-1][1])
    # print(recprice[-1][2])
    # print(recprice[-1][3])
    # print(recprice[-1][4])
    # loss,acc=conv_model.evaluate(x_data,y_data)
    # print(acc,loss)



