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
    dt=None
    data_sets=[]# 최근데이터를 맨 뒤로 보냄
    date_datas = []# 수신된 날짜 데이터
    minutetime =0
    if type(req_time)==int:
        minutetime=req_time
        req_time="minutes/"+str(req_time)
    dt = datetime.now().replace(microsecond=0)
    while(True):
        print("=========")
        #yyyy-MM-dd HH:mm:ss
        params = {"market":"KRW-"+target_name,"to":dt,"count":getcnt}
        result = requests.get(MAIN_URL+req_time,params)
        res = result.json()
        res.reverse()
        if not res:
            break
        data_sets.extend(\
            [o for o in res if not o["candle_date_time_kst"] in date_datas])
        #minutes , days, weeks, months
        if req_time=="days":
            dt = dt-timedelta(days=getcnt)
        elif req_time=="months":
            dt = dt-relativedelta(months=getcnt)
        elif req_time=="weeks":
            dt = dt-relativedelta(weeks=getcnt)
        else:
            dt = dt - timedelta(minutes=getcnt*minutetime)
        date_datas.extend([ o["candle_date_time_kst"] for o in res ])
        if len(data_sets)>=getcnt:break
    return data_sets,target_name,req_time
    # requests.get()
    #pass #주소로부터 데이터 수신

if "__main__"==__name__:
    # months, weeks,days, minutes 분 단위 : 1, 3, 5, 10, 15, 30, 60, 240
    #receive_data()
    data_sets,target_name,req_time = receive_data(req_time=3,getcnt=1000)
    #data_sets,target_name,req_time = receive_data(req_time="months",getcnt=500)
    print("수신된 데이터: 수량",len(data_sets),\
          " 이름:",target_name," 시간대:",req_time)