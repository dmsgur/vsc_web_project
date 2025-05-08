var msgbox;
$(async ()=>{
    msgbox = $("#message");
    console.log("제이쿼리작동")
    //https://api.bithumb.com/public/ticker/ALL_{payment_currency}
    //fetch("address",{option})   js object {key:value}
    //  options - method:"post|get"
    //          - headers:"Content-Type":"application/json"
    //          - body:보낼데이터
    // js 동기화 작동 async 함수 레벨 - await 호출메소드
    const conn_url = `https://api.bithumb.com/public/ticker/ALL_KRW`
    conn = await fetch(conn_url,{method:"get"}).catch((e)=>console.log(e))
    data = await conn.json()
    if(data.status=="0000"){
        msgbox.text("데이터 수신 완료")
    }else{
        msgbox.text("데이터 수신 오류 코드: "+data.status)
    }
    
})
// 함수 선언
// function 함수이름(){구현로직}
// 이름없는 무명함수 화살표함수 ()=>{}