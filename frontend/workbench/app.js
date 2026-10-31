"use strict";
const $=s=>document.querySelector(s), $$=s=>document.querySelectorAll(s);
const H=()=>({});
const HJ=()=>({...H(),"Content-Type":"application/json"});
const esc=t=>(t||"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const hms=ms=>{const s=Math.max(0,Math.floor((ms||0)/1000)),h=~~(s/3600),m=~~(s%3600/60),x=s%60;
  return [h,m,x].map(v=>String(v).padStart(2,"0")).join(":")};
let toastTimer=null;
function toast(msg){const t=$("#toast");t.textContent=msg;t.classList.add("show");
  clearTimeout(toastTimer);toastTimer=setTimeout(()=>t.classList.remove("show"),2600)}

/* ---------- 页签 ---------- */
$$(".tabs [data-t]").forEach(b=>b.onclick=()=>{
  $$(".tabs [data-t]").forEach(x=>{x.classList.toggle("on",x===b);x.setAttribute("aria-pressed",String(x===b))});
  $$(".tabpane").forEach(p=>p.classList.toggle("on",p.id==="p-"+b.dataset.t));
  if(b.dataset.t==="library")loadLibrary();
  if(b.dataset.t==="status")loadStatus();
  if(b.dataset.t==="settings")loadSettings();
  if(b.dataset.t==="memory")VideoLensMemory.load();
  else window.VideoLensMemory?.pause();
});

/* ---------- 报告显示偏好：只切换展示，不重新分析 ---------- */
let reportMode="user";
function applyReportMode(mode){
  reportMode=mode==="detailed"?"detailed":"user";
  document.body.classList.toggle("user-mode",reportMode==="user");
  window.VideoLensWorkspace?.render();
}
$("#report_mode").onchange=()=>applyReportMode($("#report_mode").value);
window.addEventListener("storage",event=>{
  if(event.key==="videolens.reportMode"&&["user","detailed"].includes(event.newValue)){
    applyReportMode(event.newValue);$("#report_mode").value=event.newValue;
  }
});

/* ---------- 视频库 ---------- */
async function loadLibrary(){
  const items=await (await fetch("/internal/videos?limit=100",{headers:H()})).json();
  const body=$("#libBody"); body.innerHTML="";
  if(!items.length){body.innerHTML='<tr><td colspan="6" class="muted">还没有视频——切到「分析」页签提交一个</td></tr>';return}
  for(const v of items){
    const ready=v.status==="ready";
    const remote=v.source_kind==="url",url=String(v.source_url||""),validUrl=/^https?:\/\//i.test(url);
    const origin=remote?`${validUrl?`<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(url)}</a>`:`<span>${esc(url||'原链接未记录')}</span>`} <button type="button" class="cache-link" data-act="cache" data-id="${v.video_id}" title="打开该视频的文件缓存目录">文件缓存</button>`:`<span class="local-source-path">${esc(v.local_path||'原地址未记录')}</span>`;
    const tr=document.createElement("tr");
    tr.innerHTML=`<td><div class="video-title-cell"><span>${esc(v.conversation_title||v.title||v.display_name||v.video_id)}</span></div><div class="video-source">${origin}</div></td>
      <td><span class="pill ${ready?"ready":v.status==="failed"?"failed":"other"}">${esc(v.status)}</span></td>
      <td>${hms(v.duration_ms)}</td><td data-detail-only>-</td><td data-detail-only>${esc(v.language||"-")}</td>
      <td>${ready?`<a class="tlink" href="#" data-act="report" data-id="${v.video_id}" data-conversation="${esc(v.conversation_id)}">报告</a>
                   <a class="tlink" href="${v.conversation_id?`/ui?conversation=${encodeURIComponent(v.conversation_id)}`:`/chat?video_id=${v.video_id}`}">对话</a>
                   <a class="tlink" href="#" data-act="del" data-id="${v.video_id}">删除</a>`
                 :`<a class="tlink" href="#" data-act="del" data-id="${v.video_id}">删除</a>`}</td>`;
    body.appendChild(tr);
  }
  body.querySelectorAll('[data-act="report"]').forEach(a=>a.onclick=async e=>{
    e.preventDefault();
    try{await VideoLensWorkspace.openVideo(a.dataset.id,a.dataset.conversation)}catch(err){toast(err.message)}
  });
  body.querySelectorAll('[data-act="del"]').forEach(a=>a.onclick=async e=>{
    e.preventDefault(); const id=a.dataset.id;
    if(!confirm("删除视频 "+id+" 及其全部关联历史对话、报告、切片、向量和本地缓存？"))return;
    try{
      const response=await fetch("/internal/videos/"+id,{method:"DELETE",headers:{...H(),'X-VideoLens-Linked-Delete':'1'}}),result=await response.json();
      if(!response.ok)throw new Error(result.detail||"删除失败");
      await VideoLensWorkspace.deleted(result);toast("已删除视频及全部关联对话");
    }catch(error){toast(error.message)}
  });
  body.querySelectorAll('[data-act="cache"]').forEach(button=>button.onclick=async()=>{
    button.disabled=true;
    try{
      const r=await fetch(`/internal/videos/${encodeURIComponent(button.dataset.id)}/open-cache`,{method:"POST",headers:H()});
      const data=await r.json();
      if(!r.ok)throw new Error(data.detail||"打开失败");
      button.title=data.directory;toast("已打开本地缓存文件夹");
    }catch(err){toast(err.message||"无法打开本地缓存")}
    finally{button.disabled=false}
  });
}
$("#refreshLib").onclick=loadLibrary;

/* ---------- 状态 ---------- */
async function loadStatus(){
  const s=await (await fetch("/internal/stats",{headers:H()})).json();
  const h=await (await fetch("/internal/health")).json();
  const cap=(k,v,lim)=>`<div class="item"><b>${v}</b><span>${k}</span>
    <div class="bar"><i style="width:${lim?Math.min(100,v/lim*100):0}%"></i></div></div>`;
  const u=s.usage||{};
  $("#statBox").innerHTML=
    cap("视频 / 200",s.videos,u.videos)+cap("切片 / 5万",s.clips,u.clips)+
    cap("转录片段 / 50万",s.transcript_segments,u.transcript_segments)+
    cap("关键帧 / 10万",s.keyframes,u.keyframes)+
    `<div class="item"><b>${(s.keyframe_bytes/1048576).toFixed(0)} MB</b><span>关键帧占用</span></div>
     <div class="item"><b>${h.status}</b><span>服务 · 本机网页</span></div>`;
  const st=await (await fetch("/internal/settings",{headers:H()})).json();
  $("#backendBox").innerHTML=(st.backends||[]).map(b=>
    `<div class="backend"><span class="${b.enabled?"on":"off"}">${b.enabled?"●":"○"}</span>
     <b>${esc(b.name)}</b> — ${esc(b.detail)}${b.warning?` <div class="warnline">${esc(b.warning)}</div>`:""}</div>`).join("");
}

/* ---------- 设置 ---------- */
async function loadSettings(){
  const st=await (await fetch("/internal/settings",{headers:H()})).json();
  const v=st.values||{};
  VideoLensKeys.load(st);
  window.VideoLensQuality?.load(st);
  $("#llm_mode").value="cloud";
  $("#llm_base").value=v["llm.cloud.base_url"]||"";
  $("#llm_model").value=v["llm.cloud.model"]||"";
  $("#vl_on").checked=true;
  $("#vl_base").value=v["vl.cloud_base_url"]||"";
  $("#vl_model").value=v["vl.cloud_model"]||"";
  const effective=st.effective||{};
  $("#asr_device").value=effective["asr.device"]||"cpu";
  $("#text_device").value=effective["embedding.text_device"]||"cpu";
  $("#image_device").value=effective["embedding.image_device"]||"cpu";
  // 保持与当前预览一致，避免离开设置页再返回时下拉框与报告模式不一致。
  $("#report_mode").value=reportMode;
  showDevices(st);

  $("#setBackends").innerHTML=(st.backends||[]).map(b=>
    `<div class="backend"><span class="${b.enabled?"on":"off"}">${b.enabled?"●":"○"}</span>
     <b>${esc(b.name)}</b> — ${esc(b.detail)}</div>`).join("");
}
$("#saveSet").onclick=async()=>{
  const button=$("#saveSet");button.disabled=true;
  $("#saveSetHint").textContent="正在应用设置；设备变更会等待当前本地推理结束并验证目标设备…";
  try{
  const payload={};
  const put=(k,el)=>{const val=el.type==="password"?el.value:el.value.trim(); if(val!=="")payload[k]=val};
  payload["llm.mode"]=$("#llm_mode").value;
  put("llm.cloud.base_url",$("#llm_base")); put("llm.cloud.model",$("#llm_model"));
  payload["vl.cloud_enabled"]=$("#vl_on").checked;
  put("vl.cloud_base_url",$("#vl_base")); put("vl.cloud_model",$("#vl_model"));
  VideoLensKeys.append(payload);
  window.VideoLensQuality?.append(payload);
  payload["asr.device"]=$("#asr_device").value;
  payload["asr.compute_type"]=payload["asr.device"]==="cuda"?"float16":"int8";
  payload["embedding.text_device"]=$("#text_device").value;
  payload["embedding.image_device"]=$("#image_device").value;
  payload["ui.report_mode"]=$("#report_mode").value;
  payload["ui.theme_mode"]=window.VideoLensTheme?.mode||"system";
  const r=await fetch("/internal/settings",{method:"POST",headers:HJ(),body:JSON.stringify(payload)});
  if(!r.ok)throw new Error((await r.json()).detail||r.status);
  const st=await r.json();
  applyReportMode(st.effective?.["ui.report_mode"]||payload["ui.report_mode"]);
  try{localStorage.setItem("videolens.reportMode",reportMode)}catch(err){/* 禁用存储不影响本页 */}
  showDevices(st);
  VideoLensKeys.load(st);
  window.VideoLensQuality?.load(st);
  $("#setBackends").innerHTML=(st.backends||[]).map(b=>
    `<div class="backend"><span class="${b.enabled?"on":"off"}">${b.enabled?"●":"○"}</span>
     <b>${esc(b.name)}</b> — ${esc(b.detail)}</div>`).join("");
  $("#saveSetHint").textContent=st.note||"设置已应用，无需重启";
  toast(st.note||"设置已应用，无需重启");
  }catch(err){
    $("#saveSetHint").textContent="应用失败："+err.message;
    toast("应用失败："+err.message);
  }finally{button.disabled=false}
};

function showDevices(st){
  const caps=st.capabilities||{};
  $("#deviceHint").textContent=caps.torch_cuda&&caps.asr_cuda?
    "GPU 运行库可用"+(caps.gpu_name?"："+caps.gpu_name:""):
    (caps.warnings||[]).join(" ");
  const d=st.active_devices||{};
  $("#activeDevices").textContent=`本次服务已生效：Whisper ${d["faster-whisper-small"]||"未加载"} · BGE-M3 ${d["bge-m3"]||"未加载"} · SigLIP2 ${d["siglip2-base"]||"未加载"}`;
  const saved=st.effective||{},names=[["asr.device","faster-whisper-small","Whisper"],["embedding.text_device","bge-m3","BGE-M3"],["embedding.image_device","siglip2-base","SigLIP2"]];
  const label=device=>device==="cuda"?"GPU":device==="cpu"?"CPU":device||"未加载";
  const pending=names.filter(([key,model])=>saved[key]&&saved[key]!==d[model]).map(([key,model,name])=>`${name} 已保存 ${label(saved[key])}，当前 ${label(d[model])}`);
  const banner=$("#pendingDevices");
  banner.hidden=pending.length===0;
  banner.textContent=pending.length?`设备设置与实际运行不一致：${pending.join("；")}。请在设置页保存以重新应用；正在执行的推理结束后会安全切换。`:"";
  $("#activeDevices").classList.toggle("warnline",pending.length>0);
}
async function initialize(){
  try{
    const st=await (await fetch("/internal/settings",{headers:H()})).json();
    applyReportMode(st.effective?.["ui.report_mode"]||"user");
    window.VideoLensTheme?.apply(st.effective?.["ui.theme_mode"]||"system");
    showDevices(st);
  }catch(err){toast("报告偏好读取失败，使用用户模式")}
  loadLibrary();loadStatus();
  await VideoLensWorkspace.init();
}
initialize();
if(["#memory","#memoryNew"].includes(location.hash))document.querySelector('[data-t="memory"]').click();
