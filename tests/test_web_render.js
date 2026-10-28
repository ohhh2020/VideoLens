/* 独立前端回归：开发环境安装 jsdom 后 node tests/test_web_render.js。 */
"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),path=require("node:path");
const {JSDOM}=require("jsdom");
const ROOT=path.resolve(__dirname,"..");
const source=file=>fs.readFileSync(path.join(ROOT,"frontend",file),"utf8");
const dom=new JSDOM(source("workbench/index.html"),{url:"http://127.0.0.1:8000/ui",runScripts:"outside-only"});
const w=dom.window;
const calls=[];
w.fetch=async url=>{
  calls.push(url);
  const data=url.includes("settings")?{values:{},effective:{"ui.report_mode":"user","asr.device":"cpu","embedding.text_device":"cuda","embedding.image_device":"cpu"},active_devices:{"faster-whisper-small":"cpu","bge-m3":"cpu","siglip2-base":"cpu"},capabilities:{torch_cuda:true,asr_cuda:true,gpu_name:"fixture",warnings:[]}}:
    url.endsWith("/open-cache")?{opened:true,directory:"D:/cache/video"}:
    url.includes("/videos")?[{video_id:"v-cache",title:"示例视频",source_kind:"url",status:"ready",duration_ms:1000}]:url.includes("health")?{status:"ok",auth:"local-web"}:{usage:{},keyframe_bytes:0};
  return {ok:true,json:async()=>data};
};
w.HTMLElement.prototype.scrollIntoView=function(){};
for(const file of ["vendor/marked.umd.js","vendor/purify.min.js","shared/markdown.js","workbench/progress.js","workbench/keys.js"])w.eval(source(file));

async function main(){
  const chatPage=new JSDOM(source("chat/index.html"));
  assert.equal(chatPage.window.document.querySelector('a[aria-label="返回视频分析"]').getAttribute("href"),"/ui");
  chatPage.window.close();
  const md="# 标题\n\n**加粗**与*斜体*\n\n- 条目\n\n> 引用\n\n|列|值|\n|---|---|\n|甲|乙|\n\n```python\nprint('<script>')\n```\n\n[网址](https://example.com)";
  const box=w.document.createElement("div");box.innerHTML=w.VideoLensMarkdown.render(md);
  for(const selector of ["h1","strong","em","ul li","blockquote","table td","pre code","a"])assert.ok(box.querySelector(selector),selector);
  assert.equal(box.querySelector("a").rel,"noopener noreferrer");
  box.innerHTML=w.VideoLensMarkdown.render('<script>alert(1)</script><img src=x onerror="alert(1)"><a href="javascript:alert(1)">恶意链接</a><div id="report_mode">伪控件</div>');
  assert.equal(box.querySelector("script,[onerror],[onclick],a[href^='javascript:'],#report_mode"),null);
  assert.ok(box.querySelector("#user-content-report_mode"));

  box.innerHTML=w.VideoLensMarkdown.render("正文[3]，`代码[3]`",{refs:[{index:3,start_ms:2000}],formatTime:()=>"00:00:02"});
  assert.equal(box.querySelectorAll(".cite").length,1);assert.equal(box.querySelector("code").textContent,"代码[3]");

  const report={video_id:"v-private",title:"示例",duration_ms:1000,segment_count:2,clip_count:3,keyframe_count:4,vector_count:7,language:"zh",transcript_source:"whisper",warnings:["调试提示"],summary:"## 内容概览\n\n概览**重点**\n\n## 主要脉络\n\n不应在用户模式出现\n\n## 关键结论\n\n- 最终结论"};
  w.eval(source("workbench/app.js")+"\nObject.assign(window,{showResult,applyReportMode,loadSettings,showDevices,loadLibrary});");
  await new Promise(r=>setTimeout(r,20));
  w.showResult({status:"done",report});
  let body=w.document.querySelector("#resultBody");
  assert.deepEqual([...body.querySelectorAll("h2")].map(x=>x.textContent),["内容概览","关键结论"]);
  for(const hidden of ["v-private","切片","关键帧","向量","转写段数","转录来源","分析提示","主要脉络","调试提示"])assert.ok(!body.textContent.includes(hidden),hidden);
  assert.ok(body.querySelector("strong"));assert.ok(w.document.querySelector("#toggleTranscript").hidden);
  w.applyReportMode("detailed");
  assert.ok(body.textContent.includes("v-private"));assert.ok(body.textContent.includes("主要脉络"));assert.ok(body.textContent.includes("调试提示"));
  assert.equal(w.document.querySelector("#toggleTranscript").hidden,false);
  w.applyReportMode("user");assert.ok(!body.textContent.includes("主要脉络"));
  const preserved=JSON.stringify(report),requestCount=calls.length;
  const selector=w.document.querySelector("#report_mode");
  selector.value="detailed";selector.dispatchEvent(new w.Event("change"));
  assert.ok(body.textContent.includes("主要脉络"));
  selector.value="user";selector.dispatchEvent(new w.Event("change"));
  assert.ok(!body.textContent.includes("主要脉络"));
  assert.equal(JSON.stringify(report),preserved);
  assert.equal(calls.length,requestCount,"报告模式预览无需请求和重新分析");
  w.dispatchEvent(new w.StorageEvent("storage",{key:"videolens.reportMode",newValue:"detailed"}));
  assert.ok(body.textContent.includes("主要脉络"));
  w.applyReportMode("user");
  const filtered=w.VideoLensMarkdown.userReport("## 内容概览\n正文\n### 主要脉络\n应隐藏\n## 关键结论\n结论");
  assert.ok(!filtered.includes("应隐藏"));
  const legacy=w.VideoLensMarkdown.userReport("一段旧版纯文本概览");assert.ok(legacy.includes("旧版纯文本"));assert.ok(legacy.includes("## 关键结论"));
  await w.loadSettings();
  assert.equal(w.document.querySelector("#llm_base").value,"","不预填默认地址");
  assert.ok(!w.document.querySelector("#llm_base").placeholder.includes("https://"));
  assert.equal(w.document.querySelector("#llm_model").value,"","示例模型不作为默认值");
  assert.equal(w.document.querySelector("#llm_model").placeholder,"qwen3.5-flash");
  assert.equal(w.document.querySelector("#text_device").value,"cuda");
  assert.equal(w.document.querySelector("#asr_device").value,"cpu");
  assert.equal(w.document.querySelector("#image_device").value,"cpu");
  selector.value="detailed";selector.dispatchEvent(new w.Event("change"));
  await w.loadSettings();
  assert.equal(selector.value,"detailed","返回设置页仍应显示当前报告预览模式");
  selector.value="user";selector.dispatchEvent(new w.Event("change"));
  const banner=w.document.querySelector("#pendingDevices");
  assert.equal(banner.hidden,false);
  assert.ok(banner.textContent.includes("BGE-M3 已保存 GPU，当前 CPU"));
  w.showDevices({effective:{"embedding.text_device":"cuda"},active_devices:{"bge-m3":"cuda"}});
  assert.equal(banner.hidden,true);
  await w.loadLibrary();
  const cache=()=>w.document.querySelector('#libBody [data-act="cache"]');
  assert.ok(cache().closest("td").textContent.includes("示例视频"));
  assert.equal(cache().textContent,"本地缓存");
  assert.equal(w.document.querySelector('#libBody [data-act="sub"]'),null);
  assert.ok(!/SRT|VTT/.test(w.document.querySelector("#libBody").textContent));
  await cache().onclick();
  assert.ok(calls.includes("/internal/videos/v-cache/open-cache"));
  console.log("PASS：Markdown 标题/列表/表格/代码/链接、内容清理、引用、双报告模式和独立设备表单");
  dom.window.close();
}
main().catch(error=>{console.error(error);dom.window.close();process.exitCode=1});
