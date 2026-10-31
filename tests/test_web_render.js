/* NODE_PATH 指向源码外的 jsdom 安装目录；验证实际工作台脚本。 */
"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),path=require("node:path");
const {JSDOM}=require("jsdom");
const ROOT=path.resolve(__dirname,"..");
const source=file=>fs.readFileSync(path.join(ROOT,"frontend",file),"utf8");
const dom=new JSDOM(source("workbench/index.html"),{url:"http://127.0.0.1:8000/ui?conversation=c1",runScripts:"outside-only"});
const w=dom.window,calls=[];
const report={video_id:"v-private",title:"示例",duration_ms:1000,segment_count:2,clip_count:3,keyframe_count:4,vector_count:7,language:"zh",transcript_source:"whisper",warnings:["调试提示"],summary:"## 内容概览\n\n概览**重点**\n\n## 主要脉络\n\n不应在用户模式出现\n\n## 关键结论\n\n- 最终结论"};
const conversation={id:"c1",title:"测试会话",video_id:"v-private"};
const state={conversation,messages:[{id:1,role:"assistant",kind:"report",text:report.summary,payload:report}],operation:null};
class Events{constructor(url){this.url=url;Events.last=this}close(){this.closed=true}}
w.EventSource=Events;
w.fetch=async(url,options={})=>{
  calls.push([url,options]);
  const data=url.includes("settings")?{values:{},effective:{"ui.report_mode":"user","asr.device":"cpu","embedding.text_device":"cuda","embedding.image_device":"cpu"},active_devices:{"faster-whisper-small":"cpu","bge-m3":"cpu","siglip2-base":"cpu"},capabilities:{torch_cuda:true,asr_cuda:true,gpu_name:"fixture",warnings:[]}}:
    url.startsWith("/internal/conversations?")?{items:[conversation]}:
    url==="/internal/conversations/c1"?state:
    url.endsWith("/open-cache")?{opened:true,directory:"D:/cache/video"}:
    url.includes("/videos")?[{video_id:"v-cache",title:"示例视频",source_kind:"url",status:"ready",duration_ms:1000}]:
    url.includes("health")?{status:"ok",auth:"local-web"}:{usage:{},keyframe_bytes:0};
  return {ok:true,json:async()=>data};
};
w.HTMLElement.prototype.scrollIntoView=function(){};
for(const file of ["vendor/marked.umd.js","vendor/purify.min.js","shared/markdown.js","workbench/keys.js","workbench/conversations.js"])w.eval(source(file));

async function main(){
  const box=w.document.createElement("div");
  box.innerHTML=w.VideoLensMarkdown.render("# 标题\n\n**加粗**\n\n- 条目\n\n|列|值|\n|---|---|\n|甲|乙|\n\n```python\nprint('x')\n```\n\n[网址](https://example.com)");
  for(const selector of ["h1","strong","ul li","table td","pre code","a"])assert.ok(box.querySelector(selector),selector);
  box.innerHTML=w.VideoLensMarkdown.render('<script>alert(1)</script><img src=x onerror="alert(1)"><a href="javascript:alert(1)">恶意链接</a>');
  assert.equal(box.querySelector("script,[onerror],a[href^='javascript:']"),null);
  w.eval(source("workbench/app.js")+"\nObject.assign(window,{applyReportMode,loadSettings,showDevices,loadLibrary,toast});");
  await new Promise(resolve=>setTimeout(resolve,50));
  const body=w.document.querySelector("#conversationFlow");
  assert.deepEqual([...body.querySelectorAll("h2")].map(x=>x.textContent),["内容概览","关键结论"]);
  for(const hidden of ["v-private","切片","关键帧","向量","转写段数","转录来源","主要脉络","调试提示"])assert.ok(!body.textContent.includes(hidden),hidden);
  assert.ok(body.querySelector("strong"));
  const count=calls.length,preserved=JSON.stringify(report);
  w.applyReportMode("detailed");
  for(const shown of ["v-private","主要脉络","调试提示","查看转写"])assert.ok(body.textContent.includes(shown),shown);
  w.applyReportMode("user");assert.ok(!body.textContent.includes("主要脉络"));
  assert.equal(JSON.stringify(report),preserved);
  assert.equal(calls.length,count,"报告切换不重新请求分析");
  await w.loadSettings();
  assert.equal(w.document.querySelector("#llm_base").value,"");
  assert.equal(w.document.querySelector("#llm_model").value,"");
  assert.equal(w.document.querySelector("#llm_model").placeholder,"qwen3.5-flash");
  assert.equal(w.document.querySelector("#text_device").value,"cuda");
  assert.equal(w.document.querySelector("#asr_device").value,"cpu");
  assert.equal(w.document.querySelector("#image_device").value,"cpu");
  assert.ok(w.document.querySelector("#pendingDevices").textContent.includes("BGE-M3 已保存 GPU，当前 CPU"));
  w.showDevices({effective:{"embedding.text_device":"cuda"},active_devices:{"bge-m3":"cuda"}});
  assert.equal(w.document.querySelector("#pendingDevices").hidden,true);
  await w.loadLibrary();
  const cache=w.document.querySelector('#libBody [data-act="cache"]');
  assert.equal(cache.textContent,"文件缓存");await cache.onclick();
  assert.ok(calls.some(([url])=>url==="/internal/videos/v-cache/open-cache"));
  assert.ok(!/SRT|VTT/.test(w.document.querySelector("#libBody").textContent));
  assert.equal(w.document.querySelectorAll("#conversationList .history-item").length,1);
  const oldCalls=calls.length;
  w.VideoLensWorkspace.updateSnapshot({...state,operation:{id:"o1",status:"running",phase:"ask",message:"正在回答"}});
  Events.last.onmessage({data:JSON.stringify({type:"delta",operation_id:"o1",text:"**流式"})});
  assert.ok(body.textContent.includes("流式"));
  const answered={...state,operation:{id:"o1",status:"done",message:"已完成"},messages:[...state.messages,{id:2,role:"assistant",kind:"answer",text:"**定稿** [1]",payload:{refs:[{index:1,start_ms:1000,text:"证据"}]}}]};
  Events.last.onmessage({data:JSON.stringify({type:"snapshot",...answered})});
  assert.ok(body.querySelector(".cite"));assert.ok(body.textContent.includes("定稿"));assert.equal(body.querySelector(".streaming"),null);
  assert.ok(!calls.slice(oldCalls).some(([,options])=>options.method==="POST"),"接收进度或定稿不触发客户端自动问答");
  const historyBefore=calls.filter(([url])=>url.startsWith("/internal/conversations?")).length;
  w.VideoLensWorkspace.updateSnapshot({...answered,conversation:{...conversation,title:"旗舰芯片性能分析"}});
  await new Promise(resolve=>setTimeout(resolve,10));
  assert.equal(w.document.querySelector("#conversationTitle").textContent,"旗舰芯片性能分析");
  assert.ok(calls.filter(([url])=>url.startsWith("/internal/conversations?")).length>historyBefore,"标题改变时立即刷新历史，即使操作状态没有变化");
  const beforeRename=calls.length;
  w.prompt=()=>"字".repeat(16);
  await w.document.querySelector('#conversationList [aria-label="重命名会话"]').onclick();
  assert.ok(!calls.slice(beforeRename).some(([,options])=>options.method==="PATCH"));
  assert.ok(w.document.querySelector("#toast").textContent.includes("15"));
  console.log("PASS：统一工作台、历史栏、流式回答、引用、Markdown安全、即时报告模式和现有设置/缓存入口");
  dom.window.close();
}
main().catch(error=>{console.error(error);dom.window.close();process.exitCode=1});
