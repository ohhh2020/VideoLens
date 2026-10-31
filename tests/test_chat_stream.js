/* 统一会话的文件发送与幂等交互；不依赖旧独立聊天页。 */
"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),path=require("node:path");
const {JSDOM}=require("jsdom");
const root=path.resolve(__dirname,".."),read=p=>fs.readFileSync(path.join(root,p),"utf8");
const dom=new JSDOM(read("frontend/workbench/index.html"),{url:"http://127.0.0.1/ui",runScripts:"outside-only"});
const w=dom.window,calls=[],links=[];
const selected={path:'D:\\视频\\演示.mp4',filename:'演示.mp4',size_bytes:7,last_modified:1700000000000,local:true};
const conversation={id:"c1",title:"新会话",video_id:""};
let operation=null;
w.toast=()=>{};w.HTMLMediaElement.prototype.pause=function(){};w.document.querySelector('[data-t="analyze"]').onclick=()=>{};w.HTMLElement.prototype.scrollIntoView=function(){};
w.VideoLensMarkdown={render:text=>text,userReport:text=>text};
w.EventSource=class{close(){}};
w.VideoLensHash={file(){throw new Error('本地视频禁止读取上传哈希')}};
w.fetch=async(url,options={})=>{
  calls.push([url,options]);
  if(url.endsWith("/messages"))operation={id:"o1",status:"running",message:"处理中"};
  if(url.endsWith('/local-video'))links.push(JSON.parse(options.body));
  const data=url==='/internal/local-videos/select'?selected:
    url.endsWith('/local-video')?{id:'u1',...selected}:url.includes("?limit=")?{items:[conversation]}:
    url==="/internal/conversations"?conversation:
    url.endsWith("/messages")?{operation_id:"o1",status:"queued"}:
    {conversation,messages:[],operation};
  return {ok:true,json:async()=>data};
};
w.eval(read("frontend/workbench/conversations.js"));
async function main(){
  await w.VideoLensWorkspace.init();
  assert.equal(calls.filter(([url,o])=>url==="/internal/conversations"&&o.method==="POST").length,0,"首页不创建空会话");
  assert.ok(!calls.some(([url])=>url==="/internal/conversations/c1"),"首页不自动打开历史");
  assert.equal(w.document.querySelector("#historySidebar").hidden,true);
  w.document.querySelector("#messageInput").value="介绍这个视频";
  await w.VideoLensWorkspace.send();
  assert.ok(w.document.querySelector(".draft-hint").textContent.includes("视频附件"));
  assert.equal(calls.filter(([,o])=>o.method==="POST").length,0,"纯文字引导不创建会话");
  await w.document.querySelector("#draftNew").onclick();
  assert.equal(calls.filter(([,o])=>o.method==="POST").length,0,"新对话只重置草稿");
  w.document.querySelector("#historyToggle").click();
  assert.equal(w.document.querySelector("#historySidebar").hidden,false);
  assert.equal(w.document.querySelector("#historyToggle").getAttribute("aria-expanded"),"true");
  w.document.dispatchEvent(new w.KeyboardEvent("keydown",{key:"Escape"}));
  assert.equal(w.document.querySelector("#historySidebar").hidden,true);
  const before=calls.length;
  await w.document.querySelector('#attachVideo').onclick();
  assert.equal(links.length,0,"选择文件不立即建立链接");
  assert.equal(calls.length,before+1,"只请求本机选择器，不建会话");
  assert.ok(w.document.querySelector("#attachmentPreview").textContent.includes("演示.mp4"));
  w.document.querySelector("#messageInput").value="解释作者依据";
  const first=w.VideoLensWorkspace.send();const second=w.VideoLensWorkspace.send();await Promise.all([first,second]);
  assert.equal(links.length,1);
  assert.equal(links[0].path,selected.path);
  assert.ok(!calls.some(([url])=>/\/uploads|\/chunks\//.test(url)),"本地视频不上传二进制");
  assert.equal(calls.filter(([url,o])=>url==="/internal/conversations"&&o.method==="POST").length,1,"首次发送附件只创建一次会话");
  const sends=calls.filter(([url])=>url.endsWith("/messages"));assert.equal(sends.length,1);
  const payload=JSON.parse(sends[0][1].body);
  assert.equal(payload.attachment_id,"u1");assert.equal(payload.text,"解释作者依据");assert.ok(payload.request_id);
  assert.equal(w.document.querySelector("#attachmentPreview").hidden,true);
  assert.equal(w.document.querySelector("#sendMessage").disabled,true);
  assert.ok(!calls.some(([url])=>url.includes("/ask")),"自动问答只由后端触发");
  operation=null;
  await w.document.querySelector("#draftNew").onclick();
  const creationCount=calls.filter(([url,o])=>url==="/internal/conversations"&&o.method==="POST").length;
  assert.equal(creationCount,1,"分析中开启新草稿也不创建空会话");
  w.document.querySelector("#messageInput").value="分析 https://example.com/video.mp4";
  const urlFirst=w.VideoLensWorkspace.send(),urlSecond=w.VideoLensWorkspace.send();await Promise.all([urlFirst,urlSecond]);
  assert.equal(calls.filter(([url,o])=>url==="/internal/conversations"&&o.method==="POST").length,2,"首次发送链接只创建一次会话");
  assert.equal(links.length,1,"云端链接不重复登记本地附件");
  dom.window.close();console.log("PASS：空首页、草稿/纯文字零创建、隐藏历史、附件/链接首次发送、双击去重与服务端编排边界");
}
main().catch(error=>{console.error(error);dom.window.close();process.exitCode=1});
