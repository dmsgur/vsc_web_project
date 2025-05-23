var msgbox;
var dataArr;
var data_name;
var filterArr;
let dispwidth =""
let dispheight =""
$(async ()=>{
    $("#cancel").on("click",()=>{
        $("#main_btn").css("display","block")
        $("#sub_btn").css("display","none")
        $("#analresult").css("display","none")
            .html("")
    })
    $("#runn").on("click",async()=>{
       let send_data=""
       try{
        send_data = JSON.parse($("info").attr("data"))
       }catch(e){}
       if(!send_data || !$("#timestep").val() || !$("#req_time").val()){
        alert("시계열 길이 또는 분석시간이 누락되었습니다.")
        return;
       }
        //서버로 데이터 송신
        let conn = await fetch(`/analize/${send_data.modeltype}`,{method:"post",headers:{"Content-Type":"application/json"},
                                    body:JSON.stringify({coinname:send_data.coinname,timestepstr:$("#timestep").val(),req_time:$("#req_time").val()})})
        //송신결과를 서버로 부터 수신
        let res = await conn.json()
        if(res.status=="success"){
            let red=res.data
            let ret_text= `<p style="padding:0.3rem;background:Aquamarine">분석코인(${red.info.coinname}) 분석모델:(${red.info.model_type}) 
            분석시간(${red.info.req_time} 분석길이(${red.info.timestepstr}))</p>`
            ret_text+=`<p>현재 모델의 ± 1% 유의수준 정확률 ${red.pv*100}%</p>`

            ret_text+=`<p>1. 예측 거래가 비교 --------------------------------------------------------<br>
            예측 시작가 ${red.cur_pri.cur_pred.toLocaleString()} 오차율 ${red.cur_pri.errrat*100}%<br>
            현재 실제값 ${red.cur_pri.pre_true.toLocaleString()}, 예측값 ${red.cur_pri.pre_pred.toLocaleString()}`

            ret_text+=`<p>2. 예측 시작가 비교 --------------------------------------------------------<br>
            예측 시작가 ${red.open_pri.cur_pred.toLocaleString()} 오차율 ${red.open_pri.errrat*100}%<br>
            현재 실제값 ${red.open_pri.pre_true.toLocaleString()}, 예측값 ${red.open_pri.pre_pred.toLocaleString()}`
           
            ret_text+=`<p>3. 예측 최고가 비교 --------------------------------------------------------<br>
            예측 시작가 ${red.high_pri.cur_pred.toLocaleString()} 오차율 ${red.high_pri.errrat*100}%<br>
            현재 실제값 ${red.high_pri.pre_true.toLocaleString()}, 예측값 ${red.high_pri.pre_pred.toLocaleString()}`

            ret_text+=`<p>4. 예측 최저가 비교 --------------------------------------------------------<br>
            예측 시작가 ${red.low_pri.cur_pred.toLocaleString()} 오차율 :${red.low_pri.errrat*100}%<br>
            현재 실제값 ${red.low_pri.pre_true.toLocaleString()}, 예측값 ${red.low_pri.pre_pred.toLocaleString()}`

            ret_text+=`<p>5. 예측 총거래가 비교 --------------------------------------------------------<br>
            예측 총거래가 ${red.tot_pri.cur_pred.toLocaleString()} 오차율 ${red.tot_pri.errrat*100}%<br>
            현재 실제값 ${red.tot_pri.pre_true.toLocaleString()}, 예측값 ${red.tot_pri.pre_pred.toLocaleString()}`
            
            $("#analresult").css("display","block")
            .html(ret_text)
        }else{
            alert("연결이 원활치 않습니다.")
        }

    })
    $("#lstm_anal,#conv_anal").on("click",function(){
        let coinname =$(this).attr("coinname")
        let ele = $(`.unitbox[coinname=${$(this).attr("coinname")}]`)
        let objtxt = (JSON.stringify({coinname,modeltype:this.id.split("_")[0]}))
        console.log(objtxt)
        $("info").attr("data",objtxt)

        $("#main_btn").css("display","none")
        $("#sub_btn").css("display","block")

    })
    $("#closebtn").on("click",function(){
        $("#cover").css("display","none")
        $("#analBtn").css("display","none")
    })
    $("#searchname").on("keypress",(e)=>{
        if(e.keyCode==13){
            $("#searchbtn").trigger("click")
        }
    })
    $("#searchbtn").on("click",()=>{
        if(!$("#searchname").val()){
            alert("영문검색어를 입력하세요")
            return
        }
        let searchunit = $("#searchname").val()
        let entry = Object.entries(dataArr)
        console.log(entry)
        entry=entry.filter((unitarr)=>{
            return unitarr[0].includes(searchunit.toUpperCase())
        })
        filterArr=Object.fromEntries(entry)
        sprayData(filterArr,data_name)

    })
    $("#runorder").on("click",()=>{
        let entry = Object.entries(dataArr)
        switch($("#selorder").val()){
            case "rat_asc":
                entry.sort((arr,arr1)=>arr[1].fluctate_rate_24H-arr1[1].fluctate_rate_24H)
                break;
            case "rat_desc":
                entry.sort((arr,arr1)=>arr1[1].fluctate_rate_24H-arr[1].fluctate_rate_24H)
                break;
            case "pri_asc":
                entry.sort((arr,arr1)=>arr[1].prev_closing_price-arr1[1].prev_closing_price)
                break;
            case "pri_desc":
                entry.sort((arr,arr1)=>arr1[1].prev_closing_price-arr[1].prev_closing_price)
                break;
            case "name_asc":
                entry.sort((arr,arr1)=>arr[0]>arr1[0]?1:-1);console.log(entry);break;
            case "name_desc":
                entry.sort((arr,arr1)=>arr1[0]>arr[0]?1:-1);console.log(entry);break;
            default:
                return;            
        }
        dataArr = Object.fromEntries(entry)
        sprayData(dataArr,data_name)
    })
    msgbox = $("#message");
    console.log("제이쿼리작동")
    //https://api.bithumb.com/public/ticker/ALL_{payment_currency}
    //fetch("address",{option})   js object {key:value}
    //  options - method:"post|get"
    //          - headers:"Content-Type":"application/json"
    //          - body:보낼데이터
    // js 동기화 작동 async 함수 레벨 - await 호출메소드
    const conn_url = `https://api.bithumb.com/public/ticker/ALL_KRW`
    const conn_han = `https://api.bithumb.com/v1/market/all`

    let conn_name = await fetch(conn_han,{method:"get"}).catch((e)=>console.log(e))
    data_name = await conn_name.json()
    console.log(data_name)
    data_name = data_name.filter((obj)=>{return !obj.market.includes("BTC-")})
    let conn = await fetch(conn_url,{method:"get"}).catch((e)=>console.log(e))
    let data = await conn.json()
    
    
    if(data.status=="0000"){
        $("#marker").css("background","green")
        let receiveTime = new Date(parseInt(data.data.date))
        msgbox.text("데이터 수신 완료 수신시간:"+receiveTime.toLocaleString("kr"))
        delete data.data.date
        console.log(data.data)
        dataArr = {...data.data}        
        sprayData(dataArr,data_name)
    }else{
        msgbox.text("데이터 수신 오류 코드: "+data.status)
        $("#marker").css("background","red")
    }
    
})
// { BTC: {…}, ETH: {…}, ETC: {…}, … }
function sprayData(data,data_name){
    $("#contain").html("")
    names = Object.keys(data)
    for (unit of names){
        let chgrat = parseFloat(data[unit].fluctate_rate_24H)
        let color = "red"
        if(chgrat<0){color="green"}
        let alpa = Math.abs(chgrat/50)
        let cobj =data_name.find((obj)=>obj.market=="KRW-"+unit)
        //.tailname .cname
        //50% 기준으로 bar 크기
        let barsize = chgrat*2
        let inHtml = `<div class="unitbox" coinname="${unit}" hanname="${cobj.korean_name}" rat="${chgrat}">
                    <p class="cname" style='background:${color};opacity:${alpa}'></p>
                    <span class="recontent mtitle">${unit}</span>
                    <p class="tailname" style='background:${color};opacity:${alpa}'></p>
                    <p class="mcontain recontent"><span class="mleft">${cobj.korean_name}</span><span style="color:blue" class="mright">(${cobj.english_name})</span></p>
                    <div>
                        <p><span class="field">최고가</span> <span class="price">${parseFloat(data[unit].max_price).toLocaleString("ko-KR")}</span></p>
                        <p><span class="field">현재가</span> <span class="price">${parseFloat(data[unit].closing_price).toLocaleString("ko-KR")}</span></p>
                        <p><span class="field">최저가</span> <span class="price">${parseFloat(data[unit].min_price).toLocaleString("ko-KR")}</span></p>
                    </div>
                    <p style='font-size:0.8rem;text-align:center'><span>시작가 ${parseFloat(data[unit].opening_price).toLocaleString("ko-KR")}</span><span style="color:${chgrat>0?"red":"green"}"> 변동율 ${data[unit].fluctate_rate_24H} %</span></p>
                    <p class="bar" style="border-radius:2px;float:${barsize<0?"right":"left"};width:${Math.abs(barsize)}%;height:0.2rem;background:${color}"></p>
                </div>`
        $(inHtml).appendTo("#contain").on("click",async function(){
            $("#analresult").html("")
            $("#analresult").css("display","none")            
            $("#sub_btn").css("display","none")
            $("#main_btn").css("display","block")
            $("#train_graph").html("")
            let coinname = $(this).attr("coinname")
            $("#cover").css("display","block")
            $("#analBtn").css("display","block")
                 .find("h1").text(coinname+`( ${$(this).attr("hanname")} )
                    ${$(this).attr("rat")} %`).css("color",$(this).attr("rat")>0?"red":"green")
            $("#btncontain #lstm_anal").attr("coinname",coinname)
            $("#btncontain #conv_anal").attr("coinname",coinname)
            //fetch("address",{option})   js object {key:value}
            //  options - method:"post|get"
            //          - headers:"Content-Type":"application/json"
            //          - body:보낼데이터
            let res = await fetch(`/graphname/${coinname}`).catch((e)=>console.log(e))
            let gnames=""
            try{
             gnames = await res.json()
            }catch(e){
                alert("훈련전입니다. 차후에 이용바랍니다.")
                $("#closebtn").trigger("click")
                return;
            }
            //totalcnt = gnames.data.lstmsave.length+gnames.data.convsave.length
            maxRow = 2
            maxcol = 10
            let w =$("#train_graph").width()-5
            let h =$("#train_graph").height()-maxcol*maxcol
            dispwidth = parseInt(w/maxcol)
            dispheight = parseInt(h/maxRow)
             $("#train_graph").html("")
            if(gnames.status=="success"){
                for (im of gnames.data.convsave){
                    $("#train_graph").append(`<div class="imgunit" style="float:left"><img style="width:${dispwidth}px;height:${dispheight}px;margin:3px 2px" src='/graph/convsave/${coinname.toUpperCase()}/${im}' /></div>`)
                }
                for (im of gnames.data.lstmsave){
                    $("#train_graph").append(`<div class="imgunit" style="float:left"><img style="width:${dispwidth}px;height:${dispheight}px;margin:3px 2px" src='/graph/convsave/${coinname.toUpperCase()}/${im}' /></div>`)
                }
//                $("#train_graph").append(`<img src='/graph/lstmsave/${coinname.toUpperCase()}/${gnames.data.lstmsave[0]}' />`)
            }
            //conresize  imgresize
            $(".imgunit").on("click",function(){
                $(this).addClass("conresize").find("img").css({width:"67vw",height:"32vw"})
                .addClass("imgc")
            })
            $(".imgunit img").on("click",function(e){
                if($(this).hasClass("imgc")){
                    e.stopPropagation()
                    $(this).removeClass("imgc")
                    $(this).css({width:dispwidth+"px",height:dispheight+"px"})
                    .parent(".imgunit").removeClass("conresize")
                }
            })



        })
        //$("#contain").append(inHtml)
    }
    
    
    //0: Object { market: "KRW-BTC", korean_name: "비트코인", english_name: "Bitcoin" }
    // console.log(data[0])
    // console.log(data.keys())
}
// 함수 선언
// function 함수이름(){구현로직}
// 이름없는 무명함수 화살표함수 ()=>{}