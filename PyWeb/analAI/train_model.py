# #train_model.py
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as  np
import requests
import pickle
import matplotlib.pyplot as plt
import os
import re
import time
from datetime import date
import tensorflow as tf
from analAI.lstm_and_conv import createModel_conv,createModel_lstm,createCallback
NAME_URL = r"https://api.bithumb.com/v1/market/all"
MAIN_URL = r"https://api.bithumb.com/v1/candles/"

SHORT = 30
MIDDLE = 60
LONG = 90
LLONG = 180
ROOT_MODEL = f"/analAI/" #웹서비스로컬 경로
#ROOT_MODEL = f"/" #구글에서 통합본 훈련시

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
def receive_data(target_name="BTC",req_time="days",getcnt=200,last_date_time=None):
    # <--- 끝날짜에서 전방으로 200개씩
    dt=None
    data_sets=[]# 최근데이터를 맨 뒤로 보냄
    date_datas = []# 수신된 날짜 데이터
    all_data_sets=None
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
        res.reverse()#최근이 맨뒤

        if not res:
            break
        dt = datetime.strptime(res[0]["candle_date_time_kst"], "%Y-%m-%dT%H:%M:%S")
        data_sets=[o for o in res if not o["candle_date_time_kst"] in date_datas]+data_sets
        #minutes , days, weeks, months
        if minutetime:
            dt = dt - timedelta(minutes=minutetime-1)
        date_datas.extend([ o["candle_date_time_kst"] for o in res ])
        if last_date_time is not None:
            fdate_time = datetime.strptime(data_sets[0]["candle_date_time_kst"],"%Y-%m-%dT%H:%M:%S")
            if fdate_time<=last_date_time:
                fix = next(i for i, o in enumerate(data_sets) if last_date_time < datetime.strptime(o["candle_date_time_kst"],"%Y-%m-%dT%H:%M:%S"))
                all_data_sets = data_sets[:]
                data_sets = data_sets[fix:]
                break
        else:
            time.sleep(0.05)
            if len(data_sets)>=getcnt:break
    return data_sets,target_name,all_data_sets
    # requests.get()
    #pass #주소로부터 데이터 수신
def createScaler(coinname,pdata_sets=None):
    scalers = []
    paths = f".{ROOT_MODEL}configs/{coinname}_scaler"
    if os.path.exists(paths):
        with open(paths, "rb") as fp:
            scalers = pickle.load(fp)
    else:
        #정규분포 z = (데이터 - 평균) / 표준편차
      if pdata_sets is not None:
          if not os.path.exists(f".{ROOT_MODEL}configs"):
            os.mkdir(f".{ROOT_MODEL}configs")
          for i in range(pdata_sets.shape[1]):
              #scalers.append({"max":pdata_sets[:,i].max(),"min": pdata_sets[:,i].min()})
              scalers.append({"mean": pdata_sets[:, i].mean(), "std": pdata_sets[:, i].std()})
          with open(paths,"wb") as fp:
              pickle.dump(scalers,fp)
      else : print("최초에는 데이터셋을 입력해야 합니다.")
    return scalers
def preData(data_sets,coinname):
    pdata_sets = np.array([[d['opening_price'],d['high_price'],d['low_price'],d['candle_acc_trade_price'],d['trade_price']]\
            for d in data_sets])
    raw_sets = pdata_sets.copy()
    #min-max 스케일 X_scaled = X_std * (max - min) + min
    # 정규분포 z = (데이터 - 평균) / 표준편차
    scalers=createScaler(coinname,pdata_sets)
    for i in range(pdata_sets.shape[1]):
       #pdata_sets[:,i] = (pdata_sets[:,i]-scalers[i]["min"])/(scalers[i]["max"]-scalers[i]["min"])
       pdata_sets[:, i] = (pdata_sets[:, i] - scalers[i]["mean"]) / scalers[i]["std"]
    return pdata_sets,raw_sets
def split_xyData(pre_datasets,raw_sets=None,step="middle"):
    time_step=0
    if step=="short":time_step=SHORT
    elif step=="long":time_step=LONG
    elif step=="llong":time_step=LLONG
    else : time_step=MIDDLE
    x_data = []
    y_data = []
    y_raw = []
    for t in range(len(pre_datasets)-time_step):
        x_data.append(pre_datasets[t:time_step+t])
        y_data.append(pre_datasets[time_step+t])#다중선형회귀
        if raw_sets is not None:
          y_raw.append(raw_sets[time_step+t])
    return np.array(x_data),np.array(y_data),np.array(y_raw)
def recovery_info(pred_data,coinname):
    scalers = createScaler(coinname)
    for dic in range(len(scalers)):
        #X_scaled = X_std * (max - min) + min
        #pred_data[:,dic] = pred_data[:,dic]*(scalers[dic]["max"]-scalers[dic]["min"])+scalers[dic]["min"]
        pred_data[:, dic] = (pred_data[:, dic] - scalers[dic]["mean"]) / scalers[dic]["std"]
    return pred_data
def train_model(self,smodel,x_data,y_data,cbs,paths,batsize=None,epoch=None,train_type="lstm",user=None,upgrade_sw=None):
    if not batsize:
        batsize = len(x_data) // 20
    if not epoch:
        epoch = 200
    fhist = smodel.fit(x_data, y_data, epochs=epoch, callbacks=cbs, batch_size=batsize,
                       validation_data=(x_data, y_data))
    print("훈련이 완료되었습니다.")
    plt.subplot(1, 2, 1)
    plt.plot(fhist.history["loss"], label="train_MSE")
    plt.plot(fhist.history["val_loss"], label="valid_MSE")
    plt.legend()
    plt.title("MSE")
    plt.subplot(1, 2, 2)
    plt.plot(fhist.history["MAE"], label="train_MAE")
    plt.plot(fhist.history["val_MAE"], label="valid_MAE")
    plt.legend()
    plt.title("MAE")
    plt.suptitle(f"model( {train_type} ) step( {self.timestepstr} ) time( {self.req_time})")
    plt.savefig(paths + "/tmp1.png")
    #plt.show()
    plt.clf()
    y_pred = smodel.predict(x_data if upgrade_sw is None else upgrade_sw["all_x_data"])

    plt.scatter(y_data if upgrade_sw is None else upgrade_sw["all_y_data"], y_pred, s=1)
    plt.plot(y_data if upgrade_sw is None else upgrade_sw["all_y_data"], y_data if upgrade_sw is None else upgrade_sw["all_y_data"])
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.title(f"model( {train_type} ) step( {self.timestepstr} ) time( {self.req_time})")
    plt.savefig(paths + "/tmp2.png")
    #plt.show()
    plt.clf()
    if upgrade_sw is not None:
        print("****************** 기존모델과 현재 훈련된 모델 비교 테스트 *******************")
        print(f"기존모델 *MSE {upgrade_sw['old_MSE']:.6f}  *MAE {upgrade_sw['old_MAE']:.6f} ")
        new_MSE, new_MAE = smodel.evaluate(upgrade_sw["all_x_data"], upgrade_sw["all_y_data"])  # 기존모델 손실도
        print(f"업그레이드 모델 *MSE {new_MSE:.6f} *MAE {new_MAE:.6f} ")
        res_mse = int((upgrade_sw['old_MSE'] - new_MSE)*1000000)/1000000;res_mae = int((upgrade_sw['old_MAE']-new_MAE)*1000000)/1000000
        print(f"기존 모델 비교 MSE {'성능향상( '+str(res_mse)+' )' if res_mse>=0 else '성능저하('+str(res_mse)+')'} MAE {'성능향상( '+str(res_mae)+' )' if res_mae>=0 else '성능저하('+str(res_mae)+')'} ")
        print("*********************** 기존모델 현재가격 예측 테스트 ***********************")
        print(upgrade_sw["old_text"])
    print("*********************** 현재모델 현재가격 예측 테스트 ***********************")
    newres_text,_ = user.pred_service(coinname=self.coinname, train_type=train_type, timestepstr=self.timestepstr,
                      req_time=self.req_time, param_model=smodel,test_train=True)
    print(newres_text)
    print("**************************** 테스트 출력 종료 ****************************")
    if upgrade_sw is None:
        new_MSE, new_MAE = smodel.evaluate(x_data, y_data)  # 기존모델 손실도
        print(f"현재 모델 *MSE {new_MSE:.6f} *MAE {new_MAE:.6f} ")

    #yn = input("현재 모델을 저장하시겠습니까(y/n)? 저장시 기존모델은 백업(bak)으로 기록됩니다.\n")
    yn = "y"
    if yn == "y":
        prebak = [f for f in os.listdir(paths) if re.match(f'.+{self.timestepstr}_{self.name_req_time if self.name_req_time is not None else self.req_time}.+\.bak', f)]
        if len(prebak):
            for prebackup in prebak:
                os.remove(paths + "/" + prebackup)
        premodel = [f for f in os.listdir(paths) if re.match(f'.+{self.timestepstr}_{self.name_req_time if self.name_req_time is not None else self.req_time}.+\.(keras|png)', f)]
        if len(premodel):
            for predata in premodel:
                os.rename(paths + "/" + predata, paths + "/" + predata.split(".")[0] + ".bak")
        dtstr = datetime.now().strftime("D%Y-%m-%dT%H-%M-%S")
        smodel.save(paths + "/{}_{}_{}_{}.keras".format(self.coinname, self.timestepstr,
                                                        self.name_req_time if self.name_req_time is not None else self.req_time,
                                                        dtstr))
        if os.path.exists(paths + "/{}_{}_{}_{}_plot.png".format(self.coinname, self.timestepstr,
                                                                 self.name_req_time if self.name_req_time is not None else self.req_time,
                                                                 dtstr)):
            os.remove(paths + "/{}_{}_{}_{}_plot.png".format(self.coinname, self.timestepstr,
                                                             self.name_req_time if self.name_req_time is not None else self.req_time,
                                                             dtstr))
        if os.path.exists(paths + "/{}_{}_{}_{}_scatt.png".format(self.coinname, self.timestepstr,
                                                                  self.name_req_time if self.name_req_time is not None else self.req_time,
                                                                  dtstr)):
            os.remove(paths + "/{}_{}_{}_{}_scatt.png".format(self.coinname, self.timestepstr,
                                                              self.name_req_time if self.name_req_time is not None else self.req_time,
                                                              dtstr))
        os.rename(paths + "/tmp1.png", paths + "/{}_{}_{}_{}_plot.png".format(self.coinname, self.timestepstr,
                                                                              self.name_req_time if self.name_req_time is not None else self.req_time,
                                                                              dtstr))
        os.rename(paths + "/tmp2.png", paths + "/{}_{}_{}_{}_scatt.png".format(self.coinname, self.timestepstr,
                                                                               self.name_req_time if self.name_req_time is not None else self.req_time,
                                                                               dtstr))
    else:
        if os.path.exists(paths + "/tmp1.png"):
            os.remove(paths + "/tmp1.png")
        if os.path.exists(paths + "/tmp2.png"):
            os.remove(paths + "/tmp2.png")
class ConfingData():
    def __init__(self,coinname="BTC",timestepstr="middle",req_time="days"):
        self.coinname=coinname
        self.timestepstr = timestepstr
        self.name_req_time=None
        if type(req_time) == int:
            self.name_req_time = "mins" + str(req_time)
        else : self.name_req_time=req_time
        self.req_time=req_time
    def init_train(self,smodels=None,cbs=None,epoch=None,batsize=None):
        # passwd = input("최초 훈련을 시작합니다.........................................."
        #                " \n스케일러등 모든 모델은 초기화 됩니다. 저장시 기존 모델은 백업으로 존재합니다. 비밀번호를 입력해주세요 1234\n")
        passwd="1234"
        if passwd != "1234":
            return
        #getnct = input("최초 훈련으로 얻어올 데이터의 수량을 입력하세요\n")

        print(self.coinname, self.timestepstr, self.name_req_time, "=========최초 훈련을 시작합니다.")
        data_sets, target_name, _ = receive_data(target_name=self.coinname, req_time=self.req_time,
                                                 getcnt=int(GETCNT) if GETCNT else None)
        print(target_name, ":수신데이터수량:", len(data_sets))
        preprocessed_sets, _ = preData(data_sets, self.coinname)
        print(target_name, "데이터 전처리가 완료됨")
        x_data, y_data, y_raw = split_xyData(preprocessed_sets, step=self.timestepstr)
        for train_type in smodels:
            print(self.coinname,train_type, "모델========= 훈련 진행중.............")
            paths = ".%s%s/%s" % (ROOT_MODEL,train_type + "save", self.coinname)
            if not os.path.exists(paths):
                os.makedirs(paths)  # 여러개의 디렉토리 생성
            user = UserService()
            train_model(self, smodels[train_type], x_data, y_data, cbs, paths, batsize, epoch,train_type,user=user)

    def upgrade_train(self,train_types=None,cbs=None,epoch=None,batsize=None):
        #passwd = input("모델의 추가 훈련데이터를 수신하여 기존모델을 업그레이드 합니다. 비밀번호를 입력해주세요5678\n")
        passwd="5678"
        if passwd != "5678":
            return
        print(self.coinname, self.timestepstr, self.name_req_time, "=========업그레이드 훈련을 시작합니다.")
        for train_type in train_types:
            paths = ".%s%s/%s" % (ROOT_MODEL,train_type + "save", self.coinname)
            model_list = [f for f in os.listdir(paths) if re.match(f'.+{self.timestepstr}_{self.name_req_time if self.name_req_time is not None else self.req_time}.+\.keras', f)]
            # print(model_list)
            load_model = None
            print(self.coinname, train_type, "모델========= 업그레이드 훈련 진행중.............")
            nd=datetime.now().replace(microsecond=0)
            if len(model_list):  # pass
                load_model = tf.keras.models.load_model(paths + "/" + model_list[0])
                dt_var = model_list[0].split(".")[0].split("_")[3]
                step_var = model_list[0].split(".")[0].split("_")[1]
                last_time = datetime.strptime(dt_var,"D%Y-%m-%dT%H-%M-%S")
                step_value = 0
                if step_var=="short":
                    step_value=SHORT
                elif step_var=="long":
                    step_value = LONG
                elif step_var=="llong":
                    step_value = LONG
                else : step_value=MIDDLE
                if self.name_req_time is not None:
                    gap=60
                    if (nd-last_time).total_seconds()//gap<self.req_time+1:
                        print(f"최종 업그레드 분수가 부족합니다.  {(nd-last_time).total_seconds()//gap+self.req_time}분 후에 다시 업그레이드를 시도하세요")
                    last_time=last_time - timedelta(minutes=((step_value+1)*self.req_time))
                elif self.req_time == "weeks":
                    gap = 60*60*24*7
                    if (nd-last_time).total_seconds()//gap<1:
                        gd = 7-((nd-last_time).total_seconds()//(60*60*24))
                        print(f"최종 업그레드 주의 일수가 부족합니다. {gd+1} 일 후에 다시 업그레이드를 시도하세요")
                    last_time=last_time - timedelta(weeks=step_value+1)
                elif self.req_time == "months":
                    gap = 60 * 60 * 24 * 31
                    if (nd-last_time).total_seconds()//gap<1:
                        gd = 31 - ((nd - last_time).total_seconds() // (60 * 60 * 24))
                        print(f"최종 업그레드 월의 일수가 부족합니다. {gd+1} 일 후에 다시 업그레이드를 시도하세요")
                    last_time=last_time - relativedelta(months=step_value+1)
                else :
                    gap = 60 * 60 * 24
                    if (nd-last_time).total_seconds()//gap<1:
                        gd = (24-(nd - last_time).total_seconds() // (60 * 60))
                        print(f"최종 업그레드일과 동일 날짜 입니다. {gd+1} 시간 후에 다시 업그레이드를 시도하세요")
                    last_time=last_time-timedelta(days=step_value+1)
                    #"days"
                data_sets,target_name,all_data_sets=receive_data(target_name="BTC", req_time=self.req_time, last_date_time=last_time)
                preprocessed_sets, y_raw = preData(data_sets, self.coinname)
                all_preprocessed_sets, all_y_raw = preData(all_data_sets, self.coinname)
                x_data, y_data, y_raw = split_xyData(preprocessed_sets,y_raw, step=self.timestepstr)
                all_x_data, all_y_data, all_y_raw = split_xyData(all_preprocessed_sets, all_y_raw, step=self.timestepstr)
                old_MSE,old_MAE = load_model.evaluate(all_x_data,all_y_data)#기존모델 손실도
                user = UserService()
                old_text,_ = user.pred_service(coinname=self.coinname, train_type=train_type, timestepstr=self.timestepstr,
                                  req_time=self.req_time, param_model=load_model,test_train=True)#기존모델 손실계산
                train_model(self, load_model, x_data, y_data, cbs, paths, batsize, epoch,train_type,user,{"old_MSE":old_MSE,"old_MAE":old_MAE,"all_x_data":all_x_data,"all_y_data":all_y_data,"all_y_raw":all_y_raw,"old_text":old_text})#True 업그레이드 여부
            else:
                print("해당 모델이 아직 존재하지 않습니다.")
                return

class UserService():
    def __init__(self):
        self.ownData=None
    def pred_service(self,coinname="BTC",timestepstr="middle",req_time="days",train_type="lstm",param_model=None,test_train=False):
        load_model = None
        if not test_train:
            name_req_time=None
            if type(req_time) == int:
                name_req_time = "mins" + str(req_time)
            print("예측을 시작합니다.")
            paths = ".%s%s/%s" % (ROOT_MODEL,train_type + "save", coinname)
            model_list = [f for f in os.listdir(paths) if re.match(f'.+{timestepstr}_{name_req_time if name_req_time is not None else req_time}.+\.keras', f)]
            # print(model_list)
            if len(model_list):#pass
                load_model = tf.keras.models.load_model(paths+"/"+model_list[0])
            else :
                print("해당 모델이 아직 존재하지 않습니다.")
                return
        else : load_model=param_model
        pred_timestep=0
        #기존 데이터 5개 오차율 검증
        if timestepstr == "short":
            pred_timestep = SHORT
        elif timestepstr == "long":
            pred_timestep = LONG
        elif timestepstr == "llong":
            pred_timestep = LLONG
        else:
            pred_timestep = MIDDLE
        x_user=x_data=y_data=y_raw=None
        if self.ownData is None:
            data_sets, target_name,_ = receive_data(target_name=coinname, req_time=req_time,
                                                  getcnt=pred_timestep+6)
            preprocessed_sets,raw_datasets = preData(data_sets, coinname)
            x_data, y_data,y_raw = split_xyData(preprocessed_sets,raw_datasets, step=timestepstr)
            #사용자 데이터 예측
            x_user = np.concatenate((x_data[-1][1:],np.array([y_data[-1]])))
            self.ownData = {"x_user":x_user.copy(),"x_data":x_data.copy(),"y_data":y_data.copy(),"y_raw":y_raw.copy()}
        else:
            x_user = self.ownData["x_user"]
            x_data = self.ownData["x_data"]
            y_data = self.ownData["y_data"]
            y_raw = self.ownData["y_raw"]
        ret_text=""
        if load_model:
            tolerance = 0.01 # 2% 오차율
            y_pred = load_model.predict(x_data)#오차함수 및 최적화함수 작동 없음
            y_pred = recovery_info(y_pred, coinname)
            acc_calgap = np.abs(y_pred / y_raw-1)
            acc_calgap[acc_calgap<tolerance]=0
            acc_mean = acc_calgap.mean(axis=0)
            ret_dict = {"pv":round(acc_mean.mean(),2).astype("float")}
            ret_text+=f"현재 모델의 ± 1% 유의수준 정확률 {1-abs(acc_mean.mean()):.2%}\n"
            pred_avgrat = (y_pred / y_raw - 1).mean(axis=0)
            user_pred = load_model.predict(np.array([x_user]))
            # print(y_pred.shape)
            rec_pred = recovery_info(user_pred, coinname)
            #opening_price,high_price,low_price,candle_acc_trade_price,trade_price
            ret_dict["info"]={"coinname":coinname,"timestepstr":timestepstr,"req_time":req_time,"model_type":train_type}
            ret_text+="1. 시작가 비교 --------------------------------------------------------\n"
            ret_text+=f"opening_price pred:{rec_pred[0][0]:.4f} recent err rate:{pred_avgrat[0]:.2%}\n"
            ret_text+=f"실제값 {y_raw[-1][0]}, 예측값 {y_pred[-1][0]}\n"
            ret_dict["open_pri"] = {"cur_pred":round(rec_pred[0][0],4).astype("float"),"errrat":round(pred_avgrat[0],2).astype("float"),
                                    "pre_true":y_raw[-1][0].astype("float"),"pre_pred":y_pred[-1][0].astype("float")}
            ret_text+="2. 최고가 비교 --------------------------------------------------------\n"
            ret_text+=f"high_price pred:{rec_pred[0][1]:.4f} recent err rate:{pred_avgrat[1]:.2%}\n"
            ret_text+=f"실제값 {y_raw[-1][1]}, 예측값 {y_pred[-1][1]}\n"
            ret_dict["high_pri"] = {"cur_pred": round(rec_pred[0][1], 4).astype("float"), "errrat": round(pred_avgrat[1], 2).astype("float"),
                                    "pre_true": y_raw[-1][1].astype("float"), "pre_pred": y_pred[-1][1].astype("float")}
            ret_text+="3. 최저가 비교 --------------------------------------------------------\n"
            ret_text+=f"low_price pred:{rec_pred[0][2]:.4f} recent err rate:{pred_avgrat[2]:.2%}\n"
            ret_text+=f"실제값 {y_raw[-1][2]}, 예측값 {y_pred[-1][2]}\n"
            ret_dict["low_pri"] = {"cur_pred": round(rec_pred[0][2], 4).astype("float"), "errrat": round(pred_avgrat[2], 2).astype("float"),
                                    "pre_true": y_raw[-1][2].astype("float"), "pre_pred": y_pred[-1][2].astype("float")}
            ret_text+="4. 가격 총 거래량 비교 --------------------------------------------------\n"
            ret_text+=f"candle_acc_trade_price pred:{rec_pred[0][3]:.4f} recent err rate:{pred_avgrat[3]:.2%}\n"
            ret_text+=f"실제값 {y_raw[-1][3]}, 예측값 {y_pred[-1][3]}\n"
            ret_dict["tot_pri"] = {"cur_pred": round(rec_pred[0][3], 4).astype("float"), "errrat": round(pred_avgrat[3], 2).astype("float"),
                                   "pre_true": y_raw[-1][3].astype("float"), "pre_pred": y_pred[-1][3].astype("float")}
            ret_text+="5. 현재가 비교 --------------------------------------------------------\n"
            ret_text+=f"trade_price pred:{rec_pred[0][4]:.4f} recent err rate:{pred_avgrat[4]:.2%}\n"
            ret_text+=f"실제값 {y_raw[-1][4]}, 예측값 {y_pred[-1][4]}\n"
            ret_dict["cur_pri"] = {"cur_pred": round(rec_pred[0][4], 4).astype("float"), "errrat": round(pred_avgrat[4], 2).astype("float"),
                                   "pre_true": y_raw[-1][4].astype("float"), "pre_pred": y_pred[-1][4].astype("float")}
        return ret_text,ret_dict;
def web_service(coinname,timestep_str,modeltype,req_time):#coinname 이름 timestep_str="middle",
    # 4. ======== 사용자 예측값 출력
    user = UserService()
    # useage user.pred_service(coinname="BTC",train_type="lstm"|"conv",timestepstr="middle",req_time=60|"days")
    _,ret_dict = user.pred_service(coinname=coinname,train_type=modeltype,timestepstr=timestep_str,req_time=req_time)
    return ret_dict
import ftplib
def sendFtp():
    session = ftplib.FTP()
    # session.connect('127.0.0.1', 21)  # 두 번째 인자는 port number
    # session.login("FTP서버_이름", "FTP서버_비밀번호")  # FTP 서버에 접속
    # with open('./파일경로/화난무민.jpg', mode='rb') as fp:
    #     session.encoding = 'utf-8'
    #     session.storbinary('STOR ' + '/img/CodingMooMin.jpg', fp)  # 파일 업로드
    # session.quit()  # 서버 나가기
    # print('파일전송함')

if "__main__"==__name__:
    # # learnner
    # # createModel_conv,createModel_lstm,createCallback
    # # 1. ==========환경설정
    # COIN_NAME = "BTC"
    # # short middle long llong
    # TIME_STEP_STR = "middle"  # 변경
    # # months, weeks, days ,  분 단위 : 1, 3, 5, 10, 15, 30, 60, 240
    # #REQ_TIME = "days"  # 변경
    # # REQ_TIME = 3  # 변경
    # REQ_TIME = "weeks"  # 변경

    # #2. ========== lstm 모델 최초 훈련()
    # MODEL_TYPE="lstm"
    # lstm_admin = ConfingData(coinname=COIN_NAME,timestepstr=TIME_STEP_STR,req_time=REQ_TIME)
    # lstm_model = createModel_lstm(TIME_STEP_STR)
    # cbs = createCallback(COIN_NAME)
    # lstm_admin.init_train(train_type=MODEL_TYPE,smodel=lstm_model,cbs=cbs,epoch=3,batsize=None)

    #3. ========== conv 모델 최초 훈련()
    # MODEL_TYPE="conv"
    # conv_admin = ConfingData(coinname=COIN_NAME,timestepstr=TIME_STEP_STR,req_time=REQ_TIME)
    # conv_model = createModel_conv(TIME_STEP_STR)
    # cbs = createCallback(COIN_NAME)
    # conv_admin.init_train(train_type=MODEL_TYPE,smodel=conv_model,cbs=cbs,epoch=2,batsize=None)

    # #4. =========== conv 모델 업그레이드
    # #conv 모델 upgrade
    # MODEL_TYPE = "conv"
    # conv_admin = ConfingData(coinname=COIN_NAME, timestepstr=TIME_STEP_STR, req_time=REQ_TIME)
    # #   upgrade
    # conv_admin.upgrade_train(train_type=MODEL_TYPE, epoch=2, batsize=None)

    # # 5. =========== lstm 모델 업그레이드
    # # lstm 모델 upgrade
    # MODEL_TYPE = "lstm"
    # lstm_admin = ConfingData(coinname=COIN_NAME, timestepstr=TIME_STEP_STR, req_time=REQ_TIME)
    # #   upgrade
    # lstm_admin.upgrade_train(train_type=MODEL_TYPE, epoch=5, batsize=None)

    # print("전처리 main 실행")
    # # months, weeks,days, minutes 분 단위 : 10, 30, 60, 240
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
    # 타입스텝은 장기, 중기, 단기로 픽스

    # 4. ======== 사용자 예측값 출력
    #user = UserService()
    #res_text,_ = user.pred_service(coinname="BTC",train_type=["conv"],timestepstr="middle",req_time="days")
    #print(res_text)
    #{'BTC': ['Bitcoin', '비트코인'], 'ETH': ['Ethereum', '이더리움'], 'ETC': ['Ethereum Classic', '이더리움 클래식'], 'XRP': ['XR
    # # 최초 모델 훈련 자동
    # print(names.keys())
    # #names_arr = [list(names.keys())]
    # names_arr=["BTC", "ETH", "XRP"]
    # GETCNT = 400
    # time_steps = ["middle","long"]
    # # req_times = [10, 30, 60, 240,"days","weeks","months"]
    # req_times = [60,"days"]
    # progressval = len(time_steps)*len(req_times)*len(names_arr)*len(["lstm","conv"])
    # progresscnt=0
    # for cname in ["XRP"]:
    #     for time_step in time_steps:
    #         for req in req_times:
    #             progresscnt+=1
    #             print(f"xxxxxxxxxxxxxxxxxxxxxxxxxx 현재 진행율 :{progresscnt/progressval*100:%} {progresscnt}/{progressval}xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    #             # middle 60 (conv lstm)
    #             if req == "months" and time_step == "llong": continue
    #             t_admin = ConfingData(coinname=cname, timestepstr=time_step, req_time=req)
    #             t_models = {"conv":createModel_conv(time_step), "lstm":createModel_lstm(time_step)}
    #             tcbs = createCallback(cname)
    #             t_admin.init_train(smodels=t_models, cbs=tcbs, epoch=20, batsize=None)

     # #업그레이드 모델 훈련 자동황
     #    # print(names.keys())
     #    #time_steps = ["short","middle","long","llong"]
     # names_arr = [list(names.keys())]

    #    GETCNT = 400
    #    time_steps = ["middle"]
    #    # req_times = [10, 30, 60, 240,"days","weeks","months"]
    #    req_times = [60]
    #    for cname in ["BTC"]:
    #        for time_step in time_steps:
    #            for req in req_times:
    #                # middle 60 (conv lstm)
    #                if req == "months" and time_step == "llong": continue
    #                t_admin = ConfingData(coinname=cname, timestepstr=time_step, req_time=req)
    #                tcbs = createCallback(cname)
    #                t_admin.upgrade_train(train_types=["conv","lstm"], epoch=2, batsize=None)
    # 웹 서비스 구현
    res = web_service("BTC", "middle", "conv", 60)  # coinname 이름 timestep_str="middle",
    print(res)
    # ------------- 훈련전용
    names_arr = [list(names.keys())]
    GETCNT = 500
    # time_steps = ["short","middle","long","llong"]
    time_steps = ["long", "llong"]
    # req_times = [60, 240,"days","months"]
    req_times = [60, 240, "days", "months"]
    train_cnt = 0
    total_count = 1 * len(req_times) * len(time_steps)
    for cname in names_arr[0]:
        for time_step in time_steps:
            for req in req_times:
                train_cnt += 1
                print(
                    f"------------------------------------------------------------ {train_cnt} / {total_count} 진행율 {train_cnt / total_count:.1%}")
                # middle 60 (conv lstm)
                if req == "months" and time_step == "llong": continue
                t_admin = ConfingData(coinname=cname, timestepstr=time_step, req_time=req)
                t_models = {"conv": createModel_conv(time_step), "lstm": createModel_lstm(time_step)}
                tcbs = createCallback(cname)
                t_admin.init_train(smodels=t_models, cbs=tcbs, epoch=500, batsize=None)