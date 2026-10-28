/* 验证菜单、动作防重入，以及重启时必须等到新进程就绪。 */
"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),path=require("node:path");
const {JSDOM}=require("jsdom"),root=path.resolve(__dirname,"..");
async function fixture(){
  const dom=new JSDOM(fs.readFileSync(path.join(root,"frontend/workbench/index.html"),"utf8"),{url:"http://127.0.0.1:8000/ui",runScripts:"outside-only"});
  const w=dom.window,timers=[],calls=[];
  let pid=100,offline=false,fail=false;
  w.setTimeout=fn=>{timers.push(fn);return timers.length};w.clearTimeout=()=>{};
  w.fetch=async(url,options={})=>{
    calls.push({url,options});
    if(url.includes("service/control"))return {ok:!fail,json:async()=>fail?{detail:"控制台不可用"}:{accepted:true,pid:100}};
    if(offline)throw new Error("offline");
    return {ok:true,json:async()=>({status:"ok",pid})};
  };
  w.eval(fs.readFileSync(path.join(root,"frontend/workbench/service.js"),"utf8"));
  return {dom,w,timers,calls,setPid:value=>{pid=value},setOffline:value=>{offline=value},setFail:value=>{fail=value}};
}
async function main(){
  const f=await fixture(),d=f.w.document;
  assert.equal(d.querySelector('.nav a[href="/internal/health"]'),null);
  assert.deepEqual([...d.querySelectorAll(".service-dropdown button")].map(x=>x.textContent),["重启","关闭"]);
  await d.getElementById("serviceRestart").onclick();
  assert.equal(d.getElementById("serviceStop").disabled,true);
  await d.getElementById("serviceRestart").onclick();
  assert.equal(f.calls.length,1);
  await f.timers.shift()();
  assert.equal(d.getElementById("serviceRestart").disabled,true,"旧进程仍健康不能误报重启完成");
  f.setPid(200);await f.timers.shift()();
  assert.equal(d.getElementById("serviceRestart").disabled,false);
  assert.ok(d.getElementById("serviceNotice").textContent.includes("已重启"));
  d.getElementById("serviceMenu").open=true;
  d.body.dispatchEvent(new f.w.MouseEvent("click",{bubbles:true}));
  assert.equal(d.getElementById("serviceMenu").open,false);
  await d.getElementById("serviceStop").onclick();f.setOffline(true);
  await f.timers.shift()();await f.timers.shift()();
  assert.ok(d.getElementById("serviceNotice").textContent.includes("已断开"));
  assert.equal(JSON.parse(f.calls.filter(c=>c.url.includes("service/control"))[1].options.body).action,"stop");
  f.dom.window.close();
  const failed=await fixture();failed.setFail(true);
  await failed.w.document.getElementById("serviceRestart").onclick();
  assert.equal(failed.w.document.getElementById("serviceRestart").disabled,false);
  assert.ok(failed.w.document.getElementById("serviceNotice").textContent.includes("控制台不可用"));
  failed.dom.window.close();
  console.log("PASS：右上角菜单、重启等待新进程、关闭反馈、防重复操作与控制台错误提示");
}
main().catch(error=>{console.error(error);process.exitCode=1});
