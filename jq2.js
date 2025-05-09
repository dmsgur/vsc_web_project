var msgbox;
var dataArr;
var data_name;
var filterArr;
$(async ()=>{
    $("#lstm_anal,#conv_anal").on("click",function(){
        let coinname =$(this).attr("coinname")
        let analtime=""
        let ele = $(`.unitbox[coinname=${$(this).attr("coinname")}]`)
        analtime=ele.attr("analtime")
        console.log(coinname,analtime)
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
        let inHtml = `<div class="unitbox" coinname="${unit}" hanname="${cobj.korean_name}" analtime="24h" rat="${chgrat}">
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
        $(inHtml).appendTo("#contain").on("click",function(){
            console.log($(this))
            console.log($(this).attr("coinname"))
            console.log($(this).attr("hanname"))
            console.log($(this).attr("rat"))
            console.log($(this).attr("analtime"))            
            $("#cover").css("display","block")
            $("#analBtn").css("display","block")
                 .find("h1").text($(this).attr("coinname")+`( ${$(this).attr("hanname")} ) 
                    ${$(this).attr("rat")} %`).css("color",$(this).attr("rat")>0?"red":"green")
            $("#btncontain #lstm_anal").attr("coinname",$(this).attr("coinname"))
            $("#btncontain #conv_anal").attr("coinname",$(this).attr("coinname"))
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