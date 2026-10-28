"use strict";

/* 真正消费网络增量；支持跨分片 UTF-8、CRLF、多行 SSE 和最终 decoder flush。 */
async function* readSSE(body){
  const reader=body.getReader(), decoder=new TextDecoder();
  let buffer="";
  const parse=block=>{
    const data=block.split(/\r?\n/).filter(s=>s.startsWith("data:")).map(s=>s.slice(5).trimStart()).join("\n");
    return data?JSON.parse(data):null;
  };
  try{
    while(true){
      const {value,done}=await reader.read();
      buffer+=done?decoder.decode():decoder.decode(value,{stream:true});
      let match;
      while((match=/\r?\n\r?\n/.exec(buffer))){
        const event=parse(buffer.slice(0,match.index));
        buffer=buffer.slice(match.index+match[0].length);
        if(event)yield event;
      }
      if(done){if(buffer.trim()){const event=parse(buffer);if(event)yield event}break}
    }
  }finally{
    await reader.cancel().catch(()=>{});
    reader.releaseLock();
  }
}

/* 只动画已经到达的 delta，不缓存整篇回答。积压时加速，避免长答案额外等待数十秒。 */
function createTypewriter(element,onUpdate=()=>{},renderText=null){
  let received=[], shown=0, timer=null, resolveDrain=null, stopped=false;
  function tick(){
    timer=null;
    if(stopped)return;
    const remaining=received.length-shown;
    if(remaining){
      shown+=Math.min(remaining,Math.max(1,Math.ceil(remaining/30)));
      const text=received.slice(0,shown).join("");
      if(renderText)element.innerHTML=renderText(text);else element.textContent=text;
      onUpdate();
    }
    if(shown<received.length)timer=setTimeout(tick,16);
    else if(resolveDrain){resolveDrain();resolveDrain=null}
  }
  return {
    push(text){
      if(stopped)return;
      received.push(...Array.from(text));
      if(timer===null)tick();
    },
    drain(){return shown===received.length?Promise.resolve():new Promise(r=>{resolveDrain=r})},
    stop(){
      stopped=true;clearTimeout(timer);timer=null;
      if(resolveDrain){resolveDrain();resolveDrain=null}
    },
    get text(){return received.join("")}
  };
}

// 同一个实现供 Node 的协议/时序回归使用，无须引入前端依赖或构建工具。
if(typeof module!=="undefined")module.exports={readSSE,createTypewriter};
