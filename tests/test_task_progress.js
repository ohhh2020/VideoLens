/* 只用 Node 标准库，验证 SSE、串行备用轮询及终态清理。 */
"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const code=fs.readFileSync(path.join(__dirname,"../frontend/workbench/progress.js"),"utf8");
const flush=()=>new Promise(resolve=>setImmediate(resolve));
function harness(withSSE=true){
  const sources=[],requests=[],timers=[],updates=[],errors=[];
  class Source{constructor(url){this.url=url;sources.push(this)}close(){this.closed=true}}
  const context=vm.createContext({AbortController,EventSource:withSSE?Source:undefined,
    fetch:(url,options)=>new Promise((resolve,reject)=>requests.push({url,options,resolve,reject})),
    setTimeout:(fn,delay)=>{const timer={fn,delay};timers.push(timer);return timer},
    clearTimeout:timer=>{if(timer)timer.cancelled=true},
  });
  vm.runInContext(code,context);
  const watcher=context.VideoLensProgress.watch("task-123",{onUpdate:x=>updates.push(x),onError:x=>errors.push(x)});
  return {sources,requests,timers,updates,errors,watcher};
}
async function main(){
  const s=harness();assert.equal(s.sources[0].url,"/internal/tasks/task-123/events");
  assert.equal(s.requests.length,0);
  s.sources[0].onmessage({data:JSON.stringify({status:"running",progress:.5})});
  s.sources[0].onmessage({data:JSON.stringify({status:"done",progress:1})});
  s.sources[0].onerror();
  assert.equal(s.updates.length,2);assert.equal(s.sources[0].closed,true);assert.equal(s.requests.length,0);

  const f=harness();f.sources[0].onerror();f.sources[0].onerror();
  assert.equal(f.requests.length,1,"多次断线通知不能启动多条备用轮询");
  assert.equal(f.timers.length,0,"前一请求未结束前不能发下一请求");
  f.requests[0].resolve({ok:true,json:async()=>({status:"running"})});await flush();
  assert.equal(f.timers.length,1);
  f.timers.shift().fn();assert.equal(f.requests.length,2);
  f.requests[1].resolve({ok:false,status:404});await flush();
  assert.equal(f.errors.length,1);assert.equal(f.timers.length,0);

  const cancel=harness(false);assert.equal(cancel.requests.length,1);
  cancel.watcher.stop();assert.equal(cancel.requests[0].options.signal.aborted,true);
  cancel.requests[0].resolve({ok:true,json:async()=>({status:"done"})});await flush();
  assert.equal(cancel.updates.length,0);assert.equal(cancel.timers.length,0);

  const malformed=harness();malformed.sources[0].onmessage({data:"invalid"});
  assert.equal(malformed.requests.length,1);
  malformed.requests[0].resolve({ok:true,json:async()=>({status:"cancelled"})});await flush();
  assert.equal(malformed.updates[0].status,"cancelled");assert.equal(malformed.timers.length,0);
  console.log("PASS：SSE 无轮询、终态关闭、断线单路串行备用轮询、404 停止和取消释放");
}
main().catch(error=>{console.error(error);process.exitCode=1});
