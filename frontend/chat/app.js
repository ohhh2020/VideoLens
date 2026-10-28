"use strict";
const $=s=>document.querySelector(s);
const player=$("#player"),flow=$("#flow"),q=$("#q"),send=$("#send"),stopBtn=$("#stop");
let currentVideo=null, currentTitle="", ctrl=null, busy=false;
let sessionId=(crypto.randomUUID ? crypto.randomUUID().slice(0,8) : Date.now().toString(36));
const H=()=>({});
const esc=t=>(t||"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const hms=ms=>{const s=Math.max(0,Math.floor(ms/1000)),h=~~(s/3600),m=~~(s%3600/60),x=s%60;
  return [h,m,x].map(v=>String(v).padStart(2,"0")).join(":")+"."+String(ms%1000).padStart(3,"0")};

/* ---------- 视频列表 ---------- */
async function loadVideos(){
  const r=await fetch("/internal/videos?limit=100",{headers:H()});
  const items=await r.json();
  const box=$("#vlist"); box.innerHTML="";
  if(!items.length){box.innerHTML='<div class="empty">还没有视频<br><span style="font-size:12px">先在「分析」页签入库一个</span></div>';return}
  for(const v of items){
    const d=document.createElement("div");
    d.className="vitem"; d.dataset.id=v.video_id;
    const ready=v.status==="ready";
    d.innerHTML=`<div class="t">${esc(v.title||v.display_name||v.video_id)}</div>
      <div class="m"><span class="st ${ready?"ready":"other"}">${esc(v.status)}</span>
      ${hms(v.duration_ms).slice(0,8)}</div>`;
    d.onclick=()=>selectVideo(v);
    box.appendChild(d);
  }
}

/* ---------- 选片 ---------- */
async function selectVideo(v){
  if(busy)return;
  currentVideo=v.video_id; currentTitle=v.title||v.display_name||v.video_id;
  document.querySelectorAll(".vitem").forEach(e=>e.classList.toggle("on",e.dataset.id===v.video_id));
  player.src=`/internal/videos/${v.video_id}/file`;
  $("#nowplaying").textContent=currentTitle;
  $("#sessWrap").style.display="block";
  flow.innerHTML="";
  await loadSessions();
  // 恢复最近一次会话的持久化对话
  const sess=await (await fetch(`/internal/videos/${v.video_id}/chat/sessions`,{headers:H()})).json();
  const last=(sess.sessions||[])[0];
  const sid=last&&last.session_id!=="legacy"?last.session_id:null;
  const r=await fetch(`/internal/videos/${v.video_id}/chat${sid?"?session_id="+sid:""}`,{headers:H()});
  const data=await r.json();
  $("#placeholder")?.remove?.();
  if(!data.turns||!data.turns.length){addPlaceholder()}
  for(const t of (data.turns||[])){
    addUserBubble(t.question,false);
    await addAssistantBubble(Promise.resolve({text:t.answer,refs:t.refs||[],notes:[]}),t.elapsed_ms,false);
  }
  if(last)sessionId=last.session_id==="legacy"?sessionId:last.session_id;
  flow.scrollTop=flow.scrollHeight;
  q.focus();
}

/* ---------- 历史会话列表 ---------- */
async function loadSessions(){
  if(!currentVideo)return;
  const box=$("#sessList"); box.innerHTML="";
  const data=await (await fetch(`/internal/videos/${currentVideo}/chat/sessions`,{headers:H()})).json();
  const list=data.sessions||[];
  if(!list.length){box.innerHTML='<div class="muted" style="font-size:12px;padding:4px 12px">暂无历史会话</div>';return}
  for(const s of list){
    const d=document.createElement("div");
    d.className="sitem";
    const when=new Date(s.last_ts*1000);
    d.innerHTML=`<span class="t">${s.session_id==="legacy"?"早期问答":s.session_id.slice(0,8)}</span>
      <span class="m">${s.turns} 轮 · ${when.getMonth()+1}/${when.getDate()} ${String(when.getHours()).padStart(2,"0")}:${String(when.getMinutes()).padStart(2,"0")}</span>`;
    d.onclick=()=>loadSessionTurns(s.session_id==="legacy"?null:s.session_id);
    box.appendChild(d);
  }
}
async function loadSessionTurns(sid){
  if(busy)return;
  const r=await fetch(`/internal/videos/${currentVideo}/chat${sid?"?session_id="+sid:""}`,{headers:H()});
  const data=await r.json();
  flow.innerHTML="";
  $("#placeholder")?.remove?.();
  if(!data.turns||!data.turns.length){addPlaceholder();return}
  for(const t of data.turns){
    addUserBubble(t.question,false);
    await addAssistantBubble(Promise.resolve({text:t.answer,refs:t.refs||[],notes:[]}),t.elapsed_ms,false);
  }
  if(sid){sessionId=sid}else{sessionId=(crypto.randomUUID?crypto.randomUUID().slice(0,8):Date.now().toString(36))}
  flow.scrollTop=flow.scrollHeight;
}
function addPlaceholder(){
  const e=document.createElement("div"); e.className="empty"; e.id="placeholder";
  e.innerHTML=`对「${esc(currentTitle)}」的内容提问<br><span style="font-size:12px">例：视频里B费做了什么？第二个进球在哪一分钟？</span>`;
  flow.appendChild(e);
}

/* ---------- 气泡 ---------- */
function addUserBubble(text,scroll=true){
  $("#placeholder")?.remove?.();
  const b=document.createElement("div"); b.className="bubble user";
  b.innerHTML=`<div class="av">我</div><div class="body"><div class="msg">${esc(text)}</div></div>`;
  flow.appendChild(b); if(scroll)flow.scrollTop=flow.scrollHeight;
  return b;
}
function createAssistantBubble(scroll=true){
  const b=document.createElement("div"); b.className="bubble asst";
  b.innerHTML=`<div class="av">知</div><div class="body"><div class="msg markdown-body typing"></div>
    <div class="meta">生成中…</div><div class="evcards"></div></div>`;
  flow.appendChild(b); if(scroll)flow.scrollTop=flow.scrollHeight;
  return b;
}
function finishAssistantBubble(b,{text,refs=[],notes=[],elapsed=0,backend="",status=""}){
  const msg=b.querySelector(".msg"),meta=b.querySelector(".meta"),ev=b.querySelector(".evcards");
  msg.classList.remove("typing");
  msg.innerHTML=renderAnswer(text,refs);
  meta.textContent=[elapsed?(elapsed/1000).toFixed(1)+"s":"",backend,refs.length?`${refs.length} 条证据`:"",status].filter(Boolean).join(" · ");
  if(notes.length)meta.title=notes.join("\n");
  ev.innerHTML=refs.map(evcard).join("");
  bindEvcards(ev,refs);
  flow.scrollTop=flow.scrollHeight;
}
async function addAssistantBubble(streamPromise,elapsed,scroll=true){
  const b=createAssistantBubble(scroll);
  finishAssistantBubble(b,{...await streamPromise,elapsed});
  return b;
}
function renderAnswer(text,refs){
  return VideoLensMarkdown.render(text,{refs,formatTime:ms=>hms(ms).slice(0,8)});
}
function evcard(r){
  const kf=(r.keyframe_paths&&r.keyframe_paths[0])?`/internal/videos/${currentVideo}/keyframes/${r.keyframe_ids?.[0]||""}/image`:"";
  return `<div class="evcard" data-n="${r.index}">
    <div class="row">${kf?`<img src="${kf}" loading="lazy" onerror="this.remove()">`:""}
    <div><div class="time" data-start="${r.start_ms}">${r.time_range}</div>
    <div>${esc((r.text||"").slice(0,220))}${(r.text||"").length>220?"…":""}</div>
    <div class="src">${esc(r.clip_id)}</div></div></div></div>`;
}
function bindEvcards(ev,refs){
  ev.querySelectorAll(".time").forEach(el=>el.onclick=()=>{player.currentTime=(+el.dataset.start)/1000;player.play()});
  ev.parentElement.querySelectorAll(".cite").forEach(c=>c.onclick=()=>{
    const card=ev.querySelector(`.evcard[data-n="${c.dataset.n}"]`);
    if(!card)return; ev.classList.add("open");
    card.scrollIntoView({block:"nearest"});
    player.currentTime=(refs.find(r=>r.index===+c.dataset.n)?.start_ms||0)/1000;
  });
}

/* ---------- 发送（SSE 打字机） ---------- */
async function sendQuestion(){
  const text=q.value.trim(); if(!text||busy||!currentVideo)return;
  busy=true; send.disabled=true; stopBtn.style.display="inline-block";
  q.value=""; q.style.height="auto";
  addUserBubble(text);
  const pending=createAssistantBubble(),meta=pending.querySelector(".meta");
  const writer=createTypewriter(pending.querySelector(".msg"),()=>{flow.scrollTop=flow.scrollHeight},text=>VideoLensMarkdown.render(text));
  const t0=performance.now();
  let final=null;
  ctrl=new AbortController();
  ctrl.signal.addEventListener("abort",()=>writer.stop(),{once:true});
  try{
    const resp=await fetch(`/internal/videos/${currentVideo}/ask/stream`,{
      method:"POST",headers:{...H(),"Content-Type":"application/json"},
      body:JSON.stringify({question:text,session_id:sessionId}),signal:ctrl.signal});
    if(!resp.ok||!resp.body)throw new Error("HTTP "+resp.status);
    for await(const evt of readSSE(resp.body)){
      if(evt.type==="status")meta.textContent=evt.data?.message||"处理中…";
      else if(evt.type==="meta")meta.textContent=`已检索 ${evt.data?.candidates??0} 条证据，等待云端输出…`;
      else if(evt.type==="delta"){
        writer.push(String(evt.data||""));
        meta.textContent="正在生成…";
      }else if(evt.type==="final"){final=evt.result;break}
      else if(evt.type==="error")throw new Error(evt.message);
    }
    if(!final)throw new Error("连接提前结束，未收到完整回答");
    await writer.drain();
    if(ctrl.signal.aborted)throw new DOMException("已停止","AbortError");
    finishAssistantBubble(pending,{text:final.answer||"（回答为空）",refs:final.refs||[],notes:final.notes||[],
      elapsed:performance.now()-t0,backend:final.backend||"",status:final.degraded?"降级":""});
    loadSessions().catch(()=>{});
  }catch(err){
    writer.stop();
    const stopped=err.name==="AbortError";
    finishAssistantBubble(pending,{text:writer.text||(stopped?"（已停止）":"出错了："+err.message),
      elapsed:performance.now()-t0,status:stopped?"已停止":"回答未完成："+err.message});
  }finally{
    writer.stop();busy=false;send.disabled=false;
    stopBtn.style.display="none";ctrl=null;flow.scrollTop=flow.scrollHeight;
  }
}

/* ---------- 交互 ---------- */
send.onclick=sendQuestion;
stopBtn.onclick=()=>ctrl?.abort();
q.addEventListener("keydown",e=>{
  if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();sendQuestion()}});
q.addEventListener("input",()=>{q.style.height="auto";q.style.height=Math.min(140,q.scrollHeight)+"px"});
$("#clearChat").onclick=async()=>{
  if(busy)return;
  if(!currentVideo)return;
  await fetch(`/internal/videos/${currentVideo}/chat?session_id=${sessionId}`,{method:"DELETE",headers:H()});
  flow.innerHTML=""; addPlaceholder(); await loadSessions();
};
$("#newSession").onclick=()=>{
  if(busy)return;
  sessionId=(crypto.randomUUID?crypto.randomUUID().slice(0,8):Date.now().toString(36));
  flow.innerHTML=""; addPlaceholder(); toast2("已开始新会话");
};
$("#exportChat").onclick=()=>{
  if(!currentVideo)return;
  window.open(`/internal/videos/${currentVideo}/chat/export?session_id=${sessionId}`);
};
function toast2(msg){const t=document.createElement("div");
  t.style.cssText="position:fixed;bottom:26px;left:50%;transform:translateX(-50%);background:#1d2330;border:1px solid #2e3542;padding:10px 22px;border-radius:10px;font-size:13px;z-index:9";
  t.textContent=msg;document.body.appendChild(t);setTimeout(()=>t.remove(),2200)}
player.addEventListener("timeupdate",()=>{$("#clock").textContent=
  hms(player.currentTime*1000).slice(0,8)+" / "+hms((player.duration||0)*1000).slice(0,8)});
async function init(){
  await loadVideos();
  // 支持 /chat?video_id=xxx 直接定位（/ui「去对话页追问」入口）
  const want=(new URLSearchParams(location.search)).get("video_id");
  if(!want)return;
  const target=[...document.querySelectorAll(".vitem")].find(e=>e.dataset.id===want);
  if(target)target.click();
}
init();
