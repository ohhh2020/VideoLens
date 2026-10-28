"use strict";

/* 服务菜单只接受正常关闭/重启，等待结果时保留当前页面。 */
(function(){
  const menu=document.getElementById("serviceMenu"),notice=document.getElementById("serviceNotice");
  const restart=document.getElementById("serviceRestart"),stop=document.getElementById("serviceStop");
  let timer=null,busy=false,failures=0;
  function message(text,state="waiting"){notice.hidden=false;notice.textContent=text;notice.dataset.state=state}
  function ready(){busy=false;restart.disabled=stop.disabled=false}
  async function check(action,oldPid){
    try{
      const response=await fetch("/internal/health?service_control=1",{cache:"no-store"});
      if(response.ok){
        const health=await response.json();failures=0;
        if(action==="restart"&&health.status==="ok"&&health.pid&&health.pid!==oldPid){
          message("服务已重启，可以继续使用。","ready");ready();return;
        }
      }
    }catch(error){
      failures++;
      if(action==="stop"&&failures>=2){
        message("网页服务已断开；控制台会在剩余任务结束后退出。再次使用时请通过启动脚本启动。","ready");return;
      }
    }
    timer=setTimeout(()=>check(action,oldPid),1500);
  }
  async function control(action){
    if(busy)return;
    busy=true;failures=0;restart.disabled=stop.disabled=true;menu.open=false;
    message(action==="restart"?"正在请求重启服务…":"正在请求关闭服务…");
    try{
      const response=await fetch("/internal/service/control",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({action})});
      const result=await response.json();
      if(!response.ok||!result.accepted)throw new Error(result.detail||"控制台未接受操作");
      message(action==="restart"?"正在重启服务；如有任务，将等待任务结束后重新加载模型…":"正在关闭服务；等待当前请求和任务结束…");
      timer=setTimeout(()=>check(action,result.pid),800);
    }catch(error){message(error.message||"操作失败，请检查服务控制台","error");ready()}
  }
  restart.onclick=()=>control("restart");stop.onclick=()=>control("stop");
  document.addEventListener("click",event=>{if(!menu.contains(event.target))menu.open=false});
  document.addEventListener("keydown",event=>{if(event.key==="Escape"&&menu.open){menu.open=false;menu.querySelector("summary").focus()}});
  window.addEventListener("pagehide",()=>clearTimeout(timer));
})();
