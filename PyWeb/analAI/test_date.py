# from datetime import datetime, timedelta
# from dateutil.relativedelta import relativedelta
# #yyyy-MM-dd HH:mm:ss
# curdatetime =datetime.now().replace(microsecond=0)
# print(curdatetime)#현재날짜
# pre_datetime=curdatetime-timedelta(days=200)
# print(pre_datetime)
# ppre_datetime=pre_datetime-timedelta(hours=240)
# print(ppre_datetime)
# pppre_datetime=ppre_datetime-timedelta(minutes=1*10)
# print(pppre_datetime)
# pre_week = pppre_datetime-relativedelta(weeks=500)
# print(pre_week)
# pre_month = pre_week-relativedelta(months=500)
# print(pre_month)

import os
import sys
from datetime import date
import tensorflow as tf # tensorflow-cpu 2.10.0
import numpy as np # numpy 1.26.4
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential,Model,Input
from tensorflow.keras.layers import Dense,LSTM,ConvLSTM1D,Dropout,Reshape
from train_model import receive_data,preData,split_xyData,recovery_info
user_count=1000
user_timestep=60
epoch=10
data_sets, target_name, req_time = receive_data(req_time=1, getcnt=user_count)
print("수신된 데이터: 수량", len(data_sets), \
      " 이름:", target_name, " 시간대:", req_time)
print("현재가격:", data_sets[-1]["trade_price"])
pre_datasets, recovery_price = preData(data_sets)  # 정규화가격정보,복구가격편차및평균
print("pre_datasets==")
print(pre_datasets.max())
x_data, y_data = split_xyData(pre_datasets, user_timestep)
print("pre_datasets==")
print(pre_datasets.max())

conv_model = Sequential()
conv_model.add(Input((user_timestep,5)))
conv_model.add(Reshape((user_timestep,5,1)))
conv_model.add(ConvLSTM1D(
8,3, strides=1,padding='same',dropout=0.3,recurrent_dropout=0.2,
return_sequences=True,go_backwards=True))#(939, 60, 5, 8)

conv_model.add(ConvLSTM1D(
32,5, strides=1,padding='same',dropout=0.3,recurrent_dropout=0.2,
return_sequences=True,go_backwards=True))#(939, 60, 5, 32)

conv_model.add(ConvLSTM1D(
128,5, strides=1,padding='same',dropout=0.3,recurrent_dropout=0.2,
return_sequences=False))#(939, 5, 128)

conv_model.add(Dense(256,activation="relu"))

conv_model.add(Dropout(0.4))
conv_model.add(Dense(64,activation="relu"))
conv_model.add(Dense(32,activation="relu"))
conv_model.add(Dense(1,activation="sigmoid"))#(939, 5, 1)
conv_model.add(Reshape((-1,)))#(939, 5)
res = conv_model(x_data)
print(res.shape)
#
# conv_model.compile(loss=tf.keras.losses.MeanSquaredError(),
#                     optimizer=tf.keras.optimizers.Adam())
# conv_model.fit(x_data,y_data,epochs=1)
#
