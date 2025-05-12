from selectors import SelectSelector
import lstm_and_conv as lac
from flask import Flask,render_template,url_for,request,jsonify
app = Flask(__name__)
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
    return "hellow python flask webpage"
@app.route("/hello")
def hello():
    return "hello path route"
#path 파라미터
@app.route("/data/<dataname>")
def getData(dataname):
    return f"{dataname} 파라미터 수신"
@app.route("/<pagename>")
def getHtml(pagename):
    print(pagename)
    return render_template(r"/front/{}.html".format(pagename))
@app.route("/analize/<kind>",methods=["post"])
def analizeAi(kind):
    rq = request.get_json()
    print(kind)
    data=None
    if kind=="lstm":
        data=""
        res = lac.getLstm()
        #lstm 모델 결과
    elif kind=="conv":
        data=""
        res = lac.getConv()
        #conv 모델 결과
    else:
        data="invalid parameter"
    return jsonify({"status":"success","data":data})

if "__main__"==__name__:
    app.run("127.0.0.1",9999,debug=True)
