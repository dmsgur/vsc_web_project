# py version 3.9
import sys
import tensorflow as tf # tensorflow-cpu 2.10.0
import numpy as np # numpy 1.26.4
import requests
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential,Model,Input
from tensorflow.keras.layers import Dense,LSTM,ConvLSTM1D
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
# == 동일 버전 ==
print("python -v ",sys.version)
print("tensorflow-cpu -v",tf.__version__)
print("numpy -v ",np.__version__)
NAME_URL = r"https://api.bithumb.com/v1/market/all"
MAIN_URL = r"https://api.bithumb.com/v1/candles/"
# https://api.bithumb.com/v1/candles/minutes/{unit}
# https://api.bithumb.com/v1/candles/days
# https: // api.bithumb.com / v1 / candles / weeks
# https://api.bithumb.com/v1/candles/months
resobj = requests.get(NAME_URL).json()# ret [ {dict} ]
#[{'market': 'KRW-BTC', 'korean_name': '비트코인', 'english_name': 'Bitcoin'}, {'market': 'KRW-ETH', 'korea
# => 변경 {BTC:["Bitcoin":"비트코인"],ETC:["ether..":"이더리움"]}
names={}
for unit in resobj:
    n = unit["market"].split("-")[1]
    names[n]=[unit["english_name"],unit["korean_name"]]
print(names)
def receive_data(target_name="BTC",req_time="days",getcnt=200):
    # <--- 끝날짜에서 전방으로 200개씩
    dt=None
    data_sets=[]# 최근데이터를 맨 뒤로 보냄
    date_datas = []# 수신된 날짜 데이터
    minutetime =0
    if type(req_time)==int:
        minutetime=req_time
        req_time="minutes/"+str(req_time)
    dt = datetime.now().replace(microsecond=0)
    while(True):
        #yyyy-MM-dd HH:mm:ss
        params = {"market":"KRW-"+target_name,"to":dt,"count":getcnt}
        result = requests.get(MAIN_URL+req_time,params)
        res = result.json()
        res.reverse()
        if not res:
            break
        dt = datetime.strptime(res[0]["candle_date_time_kst"], "%Y-%m-%dT%H:%M:%S")
        data_sets=[o for o in res if not o["candle_date_time_kst"] in date_datas]+data_sets
        #minutes , days, weeks, months
        if minutetime:
            dt = dt - timedelta(minutes=minutetime-1)
        date_datas.extend([ o["candle_date_time_kst"] for o in res ])
        if len(data_sets)>=getcnt:break
    return data_sets,target_name,req_time
    # requests.get()
    #pass #주소로부터 데이터 수신
def preData(data_sets):
    pdata_sets = np.array([[d['trade_price'],d['opening_price'],d['high_price'],d['low_price'],d['candle_acc_trade_volume']]\
            for d in data_sets])
    #정규분포로 스케일 z = (x - u) / s
    for i in range(pdata_sets.shape[1]):
        pdata_sets[:,i] = (pdata_sets[:,i]-pdata_sets[:,i].mean())/pdata_sets[:,i].std()
    print(pdata_sets.shape)
    return pdata_sets
def split_xyData(pre_datasets,time_step=20):
    x_data = []
    y_data = []
    # for t in range(time_step):
if "__main__"==__name__:
    # months, weeks,days, minutes 분 단위 : 1, 3, 5, 10, 15, 30, 60, 240
    #receive_data()
    data_sets,target_name,req_time = receive_data(req_time=1,getcnt=1000)
    #data_sets,target_name,req_time = receive_data(req_time="months",getcnt=500)
    print("수신된 데이터: 수량",len(data_sets),\
          " 이름:",target_name," 시간대:",req_time)
    print("현재가격:",data_sets[-1]["trade_price"])
    pre_datasets = preData(data_sets)
    split_xyData(pre_datasets, 5)