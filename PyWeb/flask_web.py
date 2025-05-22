from selectors import SelectSelector
from analAI.train_model import web_service
from flask import Flask,render_template,url_for,request,jsonify,send_from_directory
import os
import re

app = Flask(__name__)
GRAPH_ROOT="./analAI"
titles="선형회귀모델"
#static 디렉토리 운영
# html 경로 {{ url_for('static', filename='css/style.css') }}
#jinja 템플릿 {{출력코드}} {% 로직 %}
#데이터 수신방법
# get - request.args.get("속성명","기본값",수신타입)->("data","1",int)
# post - request.form.get("속성명","기본값",수신타입)->("data","1",int)
# get || post -
#       request.values.get("속성명","기본값",수신타입)->("data","1",int)
@app.route("/")
def index():
    print("인덱스")
    return "hellow python flask webpage"
@app.route("/graph/<path:filename>")
def graph_img(filename):#이미지 출력
    return send_from_directory(GRAPH_ROOT,filename)
@app.route("/graphname/<coinname>")
def getGraph(coinname):#이미지 경로 송출
    coinname=coinname.upper()
    convpath=GRAPH_ROOT+"/convsave/"+coinname
    lstmpath=GRAPH_ROOT + "/lstmsave/"+coinname
    convlist = [f for f in os.listdir(convpath) if re.match(
        f'.+\.png', f)]
    lstmlist = [f for f in os.listdir(convpath) if re.match(
        f'.+\.png', f)]
    return jsonify({"status":"success","data":{"convsave":convlist,"lstmsave":lstmlist}})
#path 파라미터
@app.route("/data/<dataname>")
def getData(dataname):
    return f"{dataname} 파라미터 수신"
@app.route("/<pagename>")
def getHtml(pagename):
    print(pagename)
    if not "favicon" in pagename:
        return render_template(r"/front/{}.html".format(pagename),titles=titles)
    return ""

# 모델 분석 출력

@app.route("/analize/<kind>",methods=["post"])
def analizeAi(kind):
    #모델 경로 루트로 변경
    print("ai 호출")
    rq = request.get_json()
    print(kind)
    print(rq["coinname"])
    if rq["req_time"].isdigit():
        rq["req_time"]=int(rq["req_time"])
    #web_service(coinname,timestep_str,modeltype,req_time)
    res_dict = web_service(rq["coinname"], rq["timestepstr"], kind,rq["req_time"] )  # coinname 이름 timestep_str="middle",
    print(res_dict)
    return jsonify({"status":"success","data":res_dict})
if "__main__"==__name__:
    app.run("127.0.0.1",9999,debug=True)
