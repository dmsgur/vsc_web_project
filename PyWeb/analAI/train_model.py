# py version 3.9
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as  np
import requests
import pickle
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
    return data_sets,target_name
    # requests.get()
    #pass #주소로부터 데이터 수신
def createScaler(coinname,pdata_sets):
    scalers = []
    for i in range(pdata_sets.shape[1]):
        scalers.append({"max":pdata_sets[:,i].max(),"min": pdata_sets[:,i].min()})
    with open(f"./config/{coinname}_scaler","wb") as fp:
        pickle.dump(scalers,fp)
    return scalers
def callScaler(coinname):
    scalers=None
    with open(f"./config/{coinname}_scaler","wb") as fp:
        scalers=pickle.load(fp)
    return scalers
def preData(data_sets,coinname,cre_scaler=False):
    pdata_sets = np.array([[d['opening_price'],d['high_price'],d['low_price'],d['candle_acc_trade_volume'],d['trade_price']]\
            for d in data_sets])
    #min-max 스케일 X_scaled = X_std * (max - min) + min
    scalers = None
    if cre_scaler:
        scalers=createScaler(pdata_sets,coinname)
    else :
        scalers=callScaler(coinname)
    for i in range(pdata_sets.shape[1]):
        pdata_sets[:,i] = (pdata_sets[:,i]-scalers[i]["min"])/(scalers[i]["max"]-scalers[i]["min"])
    print(pdata_sets.shape)
    return pdata_sets
def split_xyData(pre_datasets,step="middle"):
    time_step=0
    if step=="short":time_step=30
    elif step=="long":time_step=90
    elif step=="llong":time_step=180
    else : time_step=60
    x_data = []
    y_data = []
    for t in range(len(pre_datasets)-time_step-1):
        x_data.append(pre_datasets[t:time_step+t])
        y_data.append(pre_datasets[time_step+t])#다중선형회귀
    return np.array(x_data),np.array(y_data)
def recovery_info(pred_data,coinname):
    scalers = callScaler(coinname)
    for dic in range(len(scalers)):
        #X_scaled = X_std * (max - min) + min
        pred_data[:,dic] = pred_data[:,dic]*(scalers[dic]["max"]-scalers[dic]["min"])+scalers[dic]["min"]
    return pred_data
def predict_service(target_name="BTC",req_time="days",pred_step="middle"):
    pred_timestep=0
    if pred_step=="short":pred_timestep=30
    elif pred_step=="long":pred_timestep=90
    elif pred_step=="llong":pred_timestep=180
    else : pred_timestep=60
    data_sets = []  # 최근데이터를 맨 뒤로 보냄
    date_datas = []  # 수신된 날짜 데이터
    minutetime = 0
    if type(req_time) == int:
        minutetime = req_time
        req_time = "minutes/" + str(req_time)
    # yyyy-MM-dd HH:mm:ss
    params = {"market": "KRW-" + target_name,  "count": pred_timestep}
    result = requests.get(MAIN_URL + req_time, params)
    res = result.json()
    res.reverse()
    data_sets = [o for o in res if not o["candle_date_time_kst"] in date_datas] + data_sets
    # minutes , days, weeks, months
    date_datas.extend([o["candle_date_time_kst"] for o in res])
    return data_sets, target_name
class ConfingData():
    def __init__(self,coinname="BTC",timestepstr="middle",req_time="days"):
        self.coinname=coinname
        self.timestepstr = timestepstr
        self.req_time=req_time
    def init_train(self):
        passwd = input("최초 훈련을 시작합니다. 모든 모델은 초기화 됩니다. 비밀번호를 입력해주세요")
        if passwd != "1234":
            return
    def upgrade_train(self):
        passwd = input("모델의 추가 훈련데이터를 수신하여 기존모델을 업그레이드 합니다. 비밀번호를 입력해주세요")
        if passwd != "5678":
            return
class UserService():
    def __init__(self,coinname="BTC",timestepstr="middle",req_time="days"):
        self.coinname=coinname
        self.timestepstr = timestepstr
        self.req_time=req_time
    def pred_service(self):
        print("예측을 시작합니다.")



if "__main__"==__name__:
    # print("전처리 main 실행")
    # # months, weeks,days, minutes 분 단위 : 1, 3, 5, 10, 15, 30, 60, 240
    # #receive_data()
    # data_sets,target_name,req_time = receive_data(req_time=1,getcnt=1000)
    # #data_sets,target_name,req_time = receive_data(req_time="months",getcnt=500)
    # print("수신된 데이터: 수량",len(data_sets),\
    #       " 이름:",target_name," 시간대:",req_time)
    # print("현재가격:",data_sets[-1]["trade_price"])
    # pre_datasets,recovery_price = preData(data_sets)#정규화가격정보,복구가격편차및평균
    # x_data,y_data = split_xyData(pre_datasets, 5)
    # #데이터 정합성 검증
    # print(x_data.shape,y_data.shape)
    # print(y_data[0][-1]==x_data[1][4][-1])
    # print(y_data[-2][-1] == x_data[-1][4][-1])
    # #가격복구 테스트
    # recprice = recovery_info(y_data[:5], recovery_price)
    # print("가격복구정보",recprice)
    #타입스텝은 장기, 중기, 단기로 픽스
    data_sets, target_name, req_time = predict_service()
    print(len(data_sets))
    print(target_name)
    print(req_time)
    preData(data_sets,target_name)
