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
$$(".tabs button").forEach(b=>b.onclick=()=>{
  $$(".tabs button").forEach(x=>x.classList.toggle("on",x===b));
  $$(".tabpane").forEach(p=>p.classList.toggle("on",p.id==="p-"+b.dataset.t));
  if(b.dataset.t==="library")loadLibrary();
  if(b.dataset.t==="status")loadStatus();
  if(b.dataset.t==="settings")loadSettings();
  if(b.dataset.t==="memory")VideoLensMemory.load();
  else window.VideoLensMemory?.pause();
});

/* ---------- 分析 ---------- */
let taskWatcher=null,reportMode="user",lastTask=null;
window.addEventListener("pagehide",()=>taskWatcher?.stop());
function applyReportMode(mode){
  reportMode=mode==="detailed"?"detailed":"user";
  document.body.classList.toggle("user-mode",reportMode==="user");
  if(lastTask)showResult(lastTask);
  if(reportMode==="user"){
    $("#analyzeHint").textContent="";
    if(lastTask?.status==="done")$("#progText").textContent="分析完成";
  }
}
function reportMarkdown(rep){return reportMode==="user"?VideoLensMarkdown.userReport(rep.summary):rep.summary||"（暂无总结）"}
$("#report_mode").onchange=()=>applyReportMode($("#report_mode").value);
window.addEventListener("storage",event=>{
  if(event.key==="videolens.reportMode"&&["user","detailed"].includes(event.newValue)){
    applyReportMode(event.newValue);$("#report_mode").value=event.newValue;
  }
});

$("#go").onclick=async()=>{
  const src=$("#src").value.trim();
  if(!src){toast("请填写视频来源");return}
  $("#go").disabled=true; $("#analyzeHint").textContent="已提交，排队中…";
  $("#progWrap").style.display="block"; $("#prog").style.width="2%";
  $("#progText").textContent="提交中…"; $("#resultCard").style.display="none";
  try{
    const r=await fetch("/internal/videos/analyze",{method:"POST",headers:HJ(),body:JSON.stringify({
      source:src, title:$("#title").value.trim()||undefined,
      language:$("#lang").value||undefined,
      force:$("#force").checked, skip_vision:$("#skipVision").checked,
      skip_summary:$("#skipSummary").checked, skip_vectors:$("#skipVectors").checked})});
    if(!r.ok)throw new Error((await r.json()).detail||("HTTP "+r.status));
    const {task_id}=await r.json();
    $("#analyzeHint").textContent=reportMode==="detailed"?"task_id: "+task_id:"任务已提交";
    taskWatcher?.stop();
    taskWatcher=VideoLensProgress.watch(task_id,{onUpdate:t=>{
      $("#prog").style.width=Math.round((t.progress||0)*100)+"%";
      $("#progText").textContent=reportMode==="detailed"?(t.stage||"")+" — "+(t.message||"")+
        (t.elapsed_ms?" · 已用 "+Math.round(t.elapsed_ms/1000)+"s":""):
        (t.status==="done"?"分析完成":t.status==="failed"?"分析未完成":"正在分析视频…");
      if(["done","failed","cancelled"].includes(t.status)){
        $("#go").disabled=false;
        showResult(t); loadLibrary();
      }
    },onError:error=>{
      $("#go").disabled=false;
      $("#progText").textContent="进度连接中断，任务可能仍在运行。可在视频库查看结果。";
      toast(error.message);
    }});
  }catch(e){
    $("#go").disabled=false;$("#progText").textContent="提交失败："+e.message;toast("提交失败")
  }
};
let lastReport=null;
function showResult(t){
  const rep=t.report||{};
  lastTask=t;lastReport=rep;
  $("#resultCard").style.display="block";
  $("#transcriptBox").style.display="none";
  $("#transcriptBody").textContent="";
  $("#transcriptMeta").textContent="";
  $("#toggleTranscript").textContent="📝 转写结果";
  $("#toggleTranscript").hidden=reportMode==="user";
  const noTranscript=Number(rep.segment_count||0)===0;
  const notices=[...(rep.degraded||[]),...(rep.warnings||[])];
  $("#resultTitle").textContent=reportMode==="user"?(t.status==="done"?"分析完成":"分析未完成"):t.status!=="done"?"❌ 分析未完成":
    noTranscript?"⚠️ 分析结束，未生成转写文字":
    (rep.degraded||[]).length?"⚠️ 分析完成（部分功能降级）":
    (rep.warnings||[]).length?"✅ 分析完成（有提示）":"✅ 分析完成";
  if(t.status==="failed"){$("#resultBody").innerHTML='<span style="color:var(--bad)">'+esc(t.error||"未知错误")+"</span>";return}
  if(rep.video_id)$("#goChat").href="/chat?video_id="+encodeURIComponent(rep.video_id);
  if(reportMode==="user"){
    $("#resultBody").innerHTML='<div class="markdown-body">'+VideoLensMarkdown.render(reportMarkdown(rep))+'</div>';
    return;
  }
  const rows=[["video_id",rep.video_id],["标题",rep.title],["时长",hms(rep.duration_ms)],
    ["切片",rep.clip_count],["关键帧",rep.keyframe_count],["向量",rep.vector_count],
    ["转写段数",rep.segment_count||0],["转录来源",rep.transcript_source||"-"],["语言",rep.language||"-"]];
  $("#resultBody").innerHTML=rows.map(([k,v])=>`<div><b>${k}</b>  ${esc(String(v??"-"))}</div>`).join("")
    +(noTranscript?'<div style="color:var(--bad);margin-top:8px">未生成语音转写文字，请检查下方转写提示；文本、视觉 API 配置不替代本地语音识别模型。</div>':"")
    +(notices.length?'<div style="margin-top:8px;color:var(--warn)"><b>分析提示</b><ul>'+[...new Set(notices)].map(n=>`<li>${esc(String(n))}</li>`).join("")+'</ul></div>':"")
    +(rep.summary?`<div class="markdown-body" style="margin-top:12px;border-top:1px solid var(--border);padding-top:12px">${VideoLensMarkdown.render(rep.summary)}</div>`:"");
  if(rep.video_id)$("#goChat").href="/chat?video_id="+rep.video_id;
}
$("#exportMd").onclick=async e=>{
  e.preventDefault();
  const rep=lastReport;if(!rep||!rep.video_id)return;
  if(reportMode==="user"){
    downloadReport(rep,reportMarkdown(rep));return;
  }
  let transcriptText="";
  try{
    const r=await fetch(`/internal/videos/${rep.video_id}/transcript`,{headers:H()});
    if(r.ok){
      const d=await r.json();
      transcriptText=(d.segments||[]).map(s=>`[${hms(s.start_ms)}] ${s.text}`).join("\n");
    }
  }catch(err){/* 转写拉取失败不阻断导出 */}
  const md=["# "+(rep.title||rep.video_id),
    "",
    `- 时长：${hms(rep.duration_ms)}`,
    `- 语言：${rep.language||"-"} · 转录来源：${rep.transcript_source||"-"}`,
    `- 切片 ${rep.clip_count||0} 个 · 关键帧 ${rep.keyframe_count||0} 个 · 向量 ${rep.vector_count||0} 条`,
    "",
    "## 综合总结","",
    (rep.summary||"（无）").trim(),"",
    "## 语音转写全文","",
    transcriptText||"（无转写内容）"
  ].join("\n");
  downloadReport(rep,md);
};
function downloadReport(rep,md){
  const blob=new Blob([md],{type:"text/markdown;charset=utf-8"});
  const url=URL.createObjectURL(blob);
  const a=document.createElement("a");
  a.href=url;a.download=(rep.title||rep.video_id).replace(/[\\/:*?"<>|]/g,"")+".md";a.click();
  URL.revokeObjectURL(url);toast("已导出 Markdown");
};
$("#toggleTranscript").onclick=async e=>{
  e.preventDefault();
  const rep=lastReport;if(!rep||!rep.video_id||reportMode==="user")return;
  const box=$("#transcriptBox");
  if(box.style.display==="none"){
    if(!$("#transcriptBody").innerText.trim()){
      $("#transcriptBody").textContent="加载中…";
      try{
        const r=await fetch(`/internal/videos/${rep.video_id}/transcript`,{headers:H()});
        if(!r.ok){$("#transcriptBody").textContent="转写结果加载失败（"+r.status+"）"}
        else{
          const d=await r.json();
          $("#transcriptMeta").textContent=`${d.count} 段 · ${d.language||"-"}`;
          $("#transcriptBody").innerHTML=(d.segments||[]).map(s=>
            `<div><span style="color:var(--accent);font-family:monospace">${hms(s.start_ms)}</span> ${esc(s.text)}</div>`
          ).join("")||"（无转写内容）";
        }
      }catch(err){$("#transcriptBody").textContent="加载失败："+err}
    }
    box.style.display="block";$("#toggleTranscript").textContent="📝 收起转写";
  }else{
    box.style.display="none";$("#toggleTranscript").textContent="📝 转写结果";
  }
};

/* ---------- 视频库 ---------- */
async function loadLibrary(){
  const items=await (await fetch("/internal/videos?limit=100",{headers:H()})).json();
  const body=$("#libBody"); body.innerHTML="";
  if(!items.length){body.innerHTML='<tr><td colspan="6" class="muted">还没有视频——切到「分析」页签提交一个</td></tr>';return}
  for(const v of items){
    const ready=v.status==="ready";
    const tr=document.createElement("tr");
    tr.innerHTML=`<td><div class="video-title-cell"><span>${esc(v.title||v.display_name||v.video_id)}</span>
      ${v.source_kind==="url"?`<button type="button" class="cache-link" data-act="cache" data-id="${v.video_id}" title="打开该视频本地缓存所在的文件夹">本地缓存</button>`:""}</div></td>
      <td><span class="pill ${ready?"ready":v.status==="failed"?"failed":"other"}">${esc(v.status)}</span></td>
      <td>${hms(v.duration_ms)}</td><td data-detail-only>-</td><td data-detail-only>${esc(v.language||"-")}</td>
      <td>${ready?`<a class="tlink" href="#" data-act="report" data-id="${v.video_id}">📋 报告</a>
                   <a class="tlink" href="/chat?video_id=${v.video_id}">💬 对话</a>
                   <a class="tlink" href="#" data-act="del" data-id="${v.video_id}">删除</a>`
                 :`<a class="tlink" href="#" data-act="del" data-id="${v.video_id}">删除</a>`}</td>`;
    body.appendChild(tr);
  }
  body.querySelectorAll('[data-act="report"]').forEach(a=>a.onclick=async e=>{
    e.preventDefault(); const id=a.dataset.id;
    try{
      const r=await fetch(`/internal/videos/${id}/report`,{headers:H()});
      if(!r.ok){toast((await r.json().catch(()=>({}))).detail||"报告读取失败");return}
      showResult({status:"done",report:await r.json()});
      // 切到「分析」页签并滚到结果卡
      $$(".tabs button").forEach(x=>x.classList.toggle("on",x.dataset.t==="analyze"));
      $$(".tabpane").forEach(p=>p.classList.toggle("on",p.id==="p-analyze"));
      $("#resultCard").scrollIntoView({behavior:"smooth",block:"start"});
    }catch(err){toast("报告读取失败："+err)}
  });
  body.querySelectorAll('[data-act="del"]').forEach(a=>a.onclick=async e=>{
    e.preventDefault(); const id=a.dataset.id;
    if(!confirm("删除视频 "+id+" 及其切片/关键帧/向量？"))return;
    await fetch("/internal/videos/"+id,{method:"DELETE",headers:H()});
    toast("已删除"); loadLibrary();
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
     <b>${esc(b.name)}</b> — ${esc(b.detail)}${b.warning?` <div class="warnline">⚠ ${esc(b.warning)}</div>`:""}</div>`).join("");
}

/* ---------- 设置 ---------- */
async function loadSettings(){
  const st=await (await fetch("/internal/settings",{headers:H()})).json();
  const v=st.values||{};
  VideoLensKeys.load(st);
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
  payload["asr.device"]=$("#asr_device").value;
  payload["asr.compute_type"]=payload["asr.device"]==="cuda"?"float16":"int8";
  payload["embedding.text_device"]=$("#text_device").value;
  payload["embedding.image_device"]=$("#image_device").value;
  payload["ui.report_mode"]=$("#report_mode").value;
  const r=await fetch("/internal/settings",{method:"POST",headers:HJ(),body:JSON.stringify(payload)});
  if(!r.ok)throw new Error((await r.json()).detail||r.status);
  const st=await r.json();
  applyReportMode(st.effective?.["ui.report_mode"]||payload["ui.report_mode"]);
  try{localStorage.setItem("videolens.reportMode",reportMode)}catch(err){/* 禁用存储不影响本页 */}
  showDevices(st);
  VideoLensKeys.load(st);
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
    showDevices(st);
  }catch(err){toast("报告偏好读取失败，使用用户模式")}
  loadLibrary();loadStatus();
}
initialize();
if(["#memory","#memoryNew"].includes(location.hash))document.querySelector('[data-t="memory"]').click();
