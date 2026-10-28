/* 记忆库真实 DOM 编辑流程与草稿保护；测试状态仅保存在本进程。 */
"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),path=require("node:path");
const {JSDOM}=require("jsdom");
const root=path.resolve(__dirname,".."),dom=new JSDOM(fs.readFileSync(path.join(root,"frontend/workbench/index.html"),"utf8"),{url:"http://127.0.0.1:8000/ui",runScripts:"outside-only"});
const w=dom.window,d=w.document,calls=[];
let next=1;
const state={enabled:true,auto_learn:true,learning:false,last_error:"",max_text_chars:220,file:"runtime/data/memory/preferences.json",items:[]};
w.fetch=async(url,options={})=>{
  const method=options.method||"GET",body=options.body?JSON.parse(options.body):null;
  calls.push({url,method,body});
  if(method==="POST"&&url.endsWith("/items"))state.items.push({id:"m"+next++,text:body.text,kind:body.kind,origin:"manual",evidence:[],updated_at:new Date().toISOString()});
  if(method==="PUT"&&url.includes("/items/")){const item=state.items.find(i=>url.endsWith(i.id));Object.assign(item,body)}
  if(method==="DELETE")state.items=state.items.filter(i=>!url.endsWith(i.id));
  if(method==="PUT"&&url.endsWith("/options"))Object.assign(state,body);
  return {ok:true,json:async()=>structuredClone(state)};
};
w.eval(fs.readFileSync(path.join(root,"frontend/workbench/memory.js"),"utf8"));
async function main(){
  await w.VideoLensMemory.load();
  assert.ok(d.querySelector("#memoryList").textContent.includes("暂无记忆"));
  d.querySelector("#memoryNew").value='用户希望看到操作步骤。<img src=x onerror="alert(1)">';
  d.querySelector("#memoryNewKind").value="style";
  await d.querySelector("#memoryAdd").onclick();
  assert.equal(state.items.length,1);
  assert.equal(d.querySelector("#memoryList img"),null);
  let input=d.querySelector("#memoryList textarea");
  assert.equal(input.value,state.items[0].text);
  input.focus();input.value="未保存的编辑草稿";
  await w.VideoLensMemory.load();assert.equal(input.value,"未保存的编辑草稿");
  await d.querySelector("#memoryList button").onclick();
  assert.equal(state.items[0].text,"未保存的编辑草稿");
  input.blur();await w.VideoLensMemory.load();
  await d.querySelectorAll("#memoryList button")[1].onclick();
  assert.equal(state.items.length,0);
  d.querySelector("#memoryEnabled").checked=false;
  await d.querySelector("#memoryEnabled").onchange();
  assert.equal(state.enabled,false);
  assert.ok(calls.some(c=>c.method==="PUT"&&c.url.includes("/items/")));
  assert.ok(calls.some(c=>c.method==="DELETE"));
  console.log("PASS：记忆库新增/修改/删除、即时开关、纯文本显示与编辑草稿保护");
  dom.window.close();
}
main().catch(error=>{console.error(error);dom.window.close();process.exitCode=1});
