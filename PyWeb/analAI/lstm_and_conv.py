import sys
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

if "__main__"==__name__:
    data_sets, target_name, req_time = receive_data(req_time=1, getcnt=1000)
    print("수신된 데이터: 수량", len(data_sets), \
          " 이름:", target_name, " 시간대:", req_time)
    print("현재가격:", data_sets[-1]["trade_price"])
    pre_datasets, recovery_price = preData(data_sets)  # 정규화가격정보,복구가격편차및평균
    x_data, y_data = split_xyData(pre_datasets, 5)
    # 데이터 정합성 검증
    print(x_data.shape, y_data.shape)
    print("현재가격복구:",
          y_data[-1][0] * recovery_price[0]["std"] + recovery_price[0]["mean"])
    print(y_data[0][0] == x_data[1][4][0])
    print(y_data[-2][0] == x_data[-1][4][0])
    # 가격복구 검증
    recprice = recovery_info(y_data[:1], recovery_price)
    print("가격복구정보", recprice)