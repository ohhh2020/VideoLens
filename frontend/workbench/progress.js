"use strict";

/* SSE 为主，连接不可用时串行轮询；完成、取消或离开页面时释放连接。 */
(function(scope){
  const terminal=new Set(["done","failed","cancelled"]);
  function watch(taskId,{onUpdate,onError}){
    const url="/internal/tasks/"+encodeURIComponent(taskId);
    let source=null,timer=null,stopped=false,failures=0,fallbackStarted=false;
    const abort=new AbortController();
    function stop(){
      if(stopped)return;
      stopped=true;source?.close();clearTimeout(timer);abort.abort();
    }
    function update(task){
      if(stopped)return;
      if(!task||typeof task.status!=="string")throw new Error("任务进度格式无效");
      if(terminal.has(task.status))stop();
      onUpdate(task);
    }
    async function poll(){
      if(stopped)return;
      try{
        const response=await fetch(url,{signal:abort.signal});
        if(!response.ok){
          const error=new Error("进度读取失败：HTTP "+response.status);
          error.permanent=response.status>=400&&response.status<500;
          throw error;
        }
        update(await response.json());failures=0;
      }catch(error){
        if(stopped)return;
        failures++;
        if(error.permanent||failures>=3){stop();onError(error);return}
      }
      if(!stopped)timer=setTimeout(poll,Math.min(5000,1500*2**failures));
    }
    function fallback(){
      if(stopped||fallbackStarted)return;
      fallbackStarted=true;source?.close();poll();
    }
    if(typeof scope.EventSource==="function"){
      try{
        source=new scope.EventSource(url+"/events");
        source.onmessage=event=>{
          try{update(JSON.parse(event.data))}
          catch(error){fallback()}
        };
        source.onerror=fallback;
      }catch(error){fallback()}
    }else fallback();
    return {stop};
  }
  scope.VideoLensProgress={watch};
})(globalThis);
