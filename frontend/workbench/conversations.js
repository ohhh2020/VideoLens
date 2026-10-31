"use strict";

/* 网页选择本地原文件、提交与订阅；服务端负责软链接和分析后自动问答。 */
(function(scope){
  let current=null,snapshot=null,source=null,file=null,uploadBusy=false,submitting=false,historyOffset=0,hasMore=false;
  let history=[],requestPending=null,streamText="",streamOperation="",upload=null;
  let selection=0;
  let draftHint="";
  const $=selector=>document.querySelector(selector);
  const escape=value=>String(value||"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const uid=()=>scope.crypto?.randomUUID?.()||Date.now().toString(36)+Math.random().toString(36).slice(2);
  const stamp=ms=>{const s=Math.floor((ms||0)/1000);return [Math.floor(s/3600),Math.floor(s/60)%60,s%60].map(v=>String(v).padStart(2,"0")).join(":")};
  const detailed=()=>document.body.classList.contains("user-mode")===false;
  async function api(url,options={}){
    const response=await fetch(url,{cache:"no-store",...options});
    const data=await response.json();
    if(!response.ok)throw new Error(typeof data.detail==="string"?data.detail:"请求失败（"+response.status+"）");
    return data;
  }
  const post=(url,data)=>api(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});
  function active(){return submitting||uploadBusy||["queued","running"].includes(snapshot?.operation?.status)}
  function buttons(){
    $("#sendMessage").disabled=active();$("#attachVideo").disabled=active();
    $("#messageInput").disabled=uploadBusy;
  }
  function reportText(report){return detailed()?report.summary||"（未生成总结）":VideoLensMarkdown.userReport(report.summary||"（未生成总结）")}
  function integrityText(report){
    const p=report.profile||{},c=p.coverage||{},u=p.usage||{};
    if(!p.analysis_version||p.analysis_version==='legacy')return '\n\n> 分析版本：旧版，复用已保存的报告。';
    if(c.representative)return `\n\n## 分析记录\n\n- 分析版本：${p.analysis_version}\n- 实际分辨率：${p.source_width||'?'} × ${p.source_height||'?'}\n- 固定采样：${p.visual_coverage?.sample_count||0} 帧\n- 单次报告取样：${c.covered_clips||0}/${c.total_clips||0} 切片（代表性节选，未逐条总结全片）\n- 云端请求：${u.requests||0}，输入tokens：${u.prompt_tokens||0}，输出tokens：${u.completion_tokens||0}\n- 报告截断：${p.report_truncated?'是':'否'}\n`;
    return `\n\n## 分析完整性记录\n\n- 分析版本：${p.analysis_version}\n- 实际分辨率：${p.source_width||'?'} × ${p.source_height||'?'}\n- 切片处理覆盖：${c.covered_clips||0}/${c.total_clips||report.clip_count||0}，${Math.round((c.ratio||0)*100)}%（不等于语义召回率或逐帧观看）\n- 云端请求：${u.requests||0}，输入tokens：${u.prompt_tokens||0}，输出tokens：${u.completion_tokens||0}\n- 报告截断：${p.report_truncated?'是，报告尚未完成':'否'}\n\n## 章节与关键事实索引\n\n`+(p.summary_tree||[]).map(node=>`### ${stamp(node.start_ms)} — ${stamp(node.end_ms)} · 层级 ${node.level}\n\n${node.summary}\n\n${(node.facts||[]).map(f=>'- '+f).join('\n')}\n\n证据ID：${(node.evidence_ids||[]).join(', ')}\n切片：${(node.clip_ids||[]).join(', ')}`).join('\n\n');
  }
  function integrityHtml(report){
    const p=report.profile||{},c=p.coverage||{},u=p.usage||{};
    if(!p.analysis_version||p.analysis_version==='legacy')return '<div class="message-meta">分析版本：旧版 · 复用已保存的报告</div>';
    const cost=u.estimated_cost==null?'':` · 估算费用 ${Number(u.estimated_cost).toFixed(4)}（按配置单价）`;
    if(c.representative)return `<details class="analysis-notices"><summary>分析记录 · 单次报告取样 ${c.covered_clips||0}/${c.total_clips||0} 切片</summary><p>版本 ${escape(p.analysis_version)} · 实际分辨率 ${p.source_width||'?'} × ${p.source_height||'?'}</p><p>固定采样 ${p.visual_coverage?.sample_count||0} 帧；整帧识别，不执行局部加密或高清裁图。报告为代表性节选，未逐条总结全片。</p><p>云端请求 ${u.requests||0} · 输入 ${u.prompt_tokens||0} / 输出 ${u.completion_tokens||0} tokens${cost}</p><details class="raw-evidence" data-video="${escape(report.video_id)}" data-run="${escape(p.analysis_run_id)}"><summary>原始语音 / 画面 / OCR证据</summary><div class="raw-evidence-content">展开后加载</div></details></details>`;
    return `<details class="analysis-notices"><summary>分析完整性 · ${c.covered_clips||0}/${c.total_clips||report.clip_count||0} 切片 · ${Math.round((c.ratio||0)*100)}%</summary><p>版本 ${escape(p.analysis_version)} · 实际分辨率 ${p.source_width||'?'} × ${p.source_height||'?'}</p><p>离散采样 ${p.visual_coverage?.sample_count||0} 帧；连续视频窗口 ${p.visual_coverage?.continuous_windows?.length||0}。切片处理覆盖率不等于语义召回率或逐帧观看。</p><p>云端请求 ${u.requests||0} · 输入 ${u.prompt_tokens||0} / 输出 ${u.completion_tokens||0} tokens${cost}</p>${u.warning?'<p class="warnline">云端请求数已达到预警阈值。</p>':''}<details><summary>全部章节与关键事实（${(p.summary_tree||[]).length} 个节点）</summary>${(p.summary_tree||[]).map(node=>`<section><button type="button" data-time="${Number(node.start_ms)||0}">${stamp(node.start_ms)} — ${stamp(node.end_ms)} · 第${node.level}层</button><p>${escape(node.summary)}</p>${(node.facts||[]).map(f=>`<p>${escape(f)}</p>`).join('')}<small>证据：${escape((node.evidence_ids||[]).join('、'))}</small></section>`).join('')}</details><details class="raw-evidence" data-video="${escape(report.video_id)}" data-run="${escape(p.analysis_run_id)}"><summary>原始语音 / 画面 / OCR证据</summary><div class="raw-evidence-content">展开后加载</div></details></details>`;
  }
  function reportHtml(report){
    const stats=detailed()?`<div class="message-meta">视频 ${escape(report.video_id)} · 切片 ${report.clip_count||0} · 关键帧 ${report.keyframe_count||0} · 向量 ${report.vector_count||0}<br>转写段数 ${report.segment_count||0} · 转录来源 ${escape(report.transcript_source||"-")} · 语言 ${escape(report.language||"-")}</div>`:"";
    const notices=detailed()?[...(report.warnings||[]),...(report.degraded||[]),...(report.errors||[])]:[];
    return stats+(detailed()?integrityHtml(report):"")+(report.profile?.report_truncated?'<p class="warnline" role="status">报告尚未完成：输出达到长度上限。完整章节与原始证据已保留，可在详细模式查看。</p>':"")+`<div class="markdown-body">${VideoLensMarkdown.render(reportText(report))}</div>`+
      (notices.length?`<details class="analysis-notices"><summary>分析提示</summary>${notices.map(n=>`<p>${escape(n)}</p>`).join("")}</details>`:"");
  }
  function seek(ms){
    const panel=$("#videoPanel"),player=$("#player");panel.hidden=false;panel.open=true;
    player.currentTime=ms/1000;player.play()?.catch(()=>{});panel.scrollIntoView({block:"nearest",behavior:"smooth"});
  }
  function render(){
    const flow=$("#conversationFlow");if(!flow)return;
    const nearBottom=flow.scrollHeight-flow.scrollTop-flow.clientHeight<100;
    const oldScroll=flow.scrollTop;
    const messages=snapshot?.messages||[];
    flow.innerHTML=messages.length?"":'<div class="conversation-empty"><div class="empty-label">VIDEOLENS WORKSPACE</div><h2>从一段视频，开始理解</h2><p>上传本地视频，或粘贴视频分享链接。<br>附上问题，我会在分析完成后继续回答。</p><span>视频分析 · 时间证据 · 多轮追问</span></div>';
    if(!messages.length&&draftHint){const hint=document.createElement('p');hint.className='draft-hint';hint.setAttribute('role','status');hint.textContent=draftHint;flow.querySelector('.conversation-empty').append(hint)}
    for(const message of messages){
      const item=document.createElement("article");item.className="conversation-message "+message.role;
      item.dataset.messageId=message.id;
      const payload=message.payload||{},refs=payload.refs||[];
      const title=message.role==="user"?"你":message.kind==="report"?"知影 · 视频分析":"知影";
      let content=message.kind==="report"?reportHtml(payload):message.role==="user"?
        `<div class="user-text">${escape(message.text)}</div>`:VideoLensMarkdown.render(message.text,{refs,formatTime:stamp});
      if(payload.attachment)content+=`<div class="message-attachment">${escape(payload.attachment.filename)} · ${(payload.attachment.size_bytes/1048576).toFixed(1)} MB</div>`;
      if(message.kind==="clarification"){
        const choices=[...(payload.attachment_id?["附件"]:[]),...(payload.urls||[])];
        content+=`<ol class="source-choices">${choices.map((value,i)=>`<li><button type="button" data-choice="${i+1}">${escape(value)}</button></li>`).join("")}</ol>`;
      }
      item.innerHTML=`<div class="message-author">${title}</div><div class="message-content markdown-body">${content}</div>`;
      if(message.kind==="report"){
        const actions=document.createElement("div");actions.className="message-actions";
        const download=document.createElement("button");download.type="button";download.textContent="导出报告";download.onclick=()=>exportReport(payload);
        actions.append(download);
        if(detailed()){
          const transcript=document.createElement("button");transcript.type="button";transcript.textContent="查看转写";
          transcript.onclick=async()=>{
            let box=item.querySelector(".transcript-view");
            if(box){box.hidden=!box.hidden;return}
            box=document.createElement("div");box.className="transcript-view";box.textContent="加载中…";item.append(box);
            try{const data=await api(`/internal/videos/${payload.video_id}/transcript`);box.textContent=(data.segments||[]).map(s=>`[${stamp(s.start_ms)}] ${s.text}`).join("\n")||"（无转写内容）"}
            catch(error){box.textContent=error.message}
          };actions.append(transcript);
        }
        item.append(actions);
      }
      if(refs.length){
        const evidence=document.createElement("details");evidence.className="evidence-list";
        evidence.innerHTML=`<summary>${refs.length} 条视频证据</summary>`+refs.map(ref=>`<button type="button" data-time="${ref.start_ms}">[${ref.index}] ${escape(ref.time_range||stamp(ref.start_ms))}</button><p>${escape(ref.text)}</p>`).join("");
        item.append(evidence);
        item.querySelectorAll(".cite").forEach(c=>c.onclick=()=>{const ref=refs.find(r=>r.index===Number(c.dataset.n));if(ref)seek(ref.start_ms)});
      }
      item.querySelectorAll("[data-time]").forEach(button=>button.onclick=()=>seek(Number(button.dataset.time)));
      item.querySelectorAll("[data-choice]").forEach(button=>button.onclick=()=>{$("#messageInput").value="选择第"+button.dataset.choice+"个";send()});
      if(payload.new_conversation){const button=document.createElement("button");button.className="btn ghost";button.textContent="新建会话";button.onclick=newConversation;item.append(button)}
      item.querySelectorAll('.raw-evidence').forEach(box=>box.addEventListener('toggle',async()=>{
        if(!box.open||box.dataset.loaded)return;box.dataset.loaded='loading';
        const content=box.querySelector('.raw-evidence-content');content.textContent='正在加载原始证据…';
        try{const rows=await allEvidence(box.dataset.video,box.dataset.run);content.replaceChildren();for(const row of rows){const part=document.createElement('section'),button=document.createElement('button'),text=document.createElement('pre');button.type='button';button.textContent=`${stamp(row.start_ms)} · ${row.kind} · ${row.evidence_id}`;button.onclick=()=>seek(row.start_ms);text.textContent=row.text;part.append(button,text);content.append(part)}box.dataset.loaded='yes'}catch(error){content.textContent=error.message;delete box.dataset.loaded}
      }));
      flow.append(item);
    }
    if(streamText&&active()){
      const pending=document.createElement("article");pending.className="conversation-message assistant streaming";
      pending.innerHTML='<div class="message-author">知影 · 正在回答</div><div class="message-content markdown-body">'+VideoLensMarkdown.render(streamText)+"</div>";
      flow.append(pending);
    }
    const operation=snapshot?.operation,status=$("#operationStatus");
    status.hidden=!operation;
    if(operation){
      status.replaceChildren();
      const label=document.createElement("span");label.textContent=["failed","interrupted"].includes(operation.status)?operation.error:
        !detailed()&&["queued","running"].includes(operation.status)?({intent:"正在理解视频请求…",analysis:"正在分析视频…",ask:"正在回答视频问题…"}[operation.phase]||"正在处理…"):operation.message||"正在处理…";
      status.append(label);
      if(["queued","running"].includes(operation.status)){
        const progress=document.createElement("progress");progress.max=1;progress.value=operation.progress||0;status.append(progress);
      }
      if(["failed","interrupted"].includes(operation.status)){
        const retry=document.createElement("button");retry.type="button";retry.textContent="重试";
        retry.onclick=async()=>{retry.disabled=true;try{await post(`/internal/conversations/${current.id}/operations/${operation.id}/retry`,{});await refresh()}catch(error){toast(error.message);retry.disabled=false}};
        status.append(retry);
      }
    }
    $("#conversationTitle").textContent=current?.title||"视频对话";
    buttons();flow.scrollTop=nearBottom?flow.scrollHeight:oldScroll;
  }
  async function allEvidence(video,run){
    let rows=[],offset=0;for(;;){const data=await api(`/internal/videos/${video}/evidence?run_id=${encodeURIComponent(run||'')}&offset=${offset}&limit=200`);rows.push(...data.items);offset=rows.length;if(offset>=data.total||!data.items.length)return rows}
  }
  async function exportReport(report){
    let text=reportText(report);
    if(detailed()){
      text+=integrityText(report);
      if(report.profile?.analysis_run_id)try{const rows=await allEvidence(report.video_id,report.profile.analysis_run_id);text+='\n\n## 原始证据全文\n\n'+rows.map(row=>`### ${stamp(row.start_ms)} · ${row.kind} · ${row.evidence_id}\n\n${row.text}`).join('\n\n')}catch(error){toast('原始证据导出失败：'+error.message)}
      try{const data=await api(`/internal/videos/${report.video_id}/transcript`);text+='\n\n## 语音转写全文\n\n'+(data.segments||[]).map(s=>`[${stamp(s.start_ms)}] ${s.text}`).join("\n")}
      catch(error){toast("报告已保留；转写读取失败："+error.message)}
    }
    const blob=new Blob(["# "+(report.title||"视频报告")+"\n\n"+text],{type:"text/markdown;charset=utf-8"});
    const url=URL.createObjectURL(blob),link=document.createElement("a");link.href=url;link.download=(report.title||"视频报告").replace(/[\\/:*?"<>|]/g,"_")+".md";link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  async function loadHistory(reset=true){
    const data=await api(`/internal/conversations?limit=30&offset=${reset?0:historyOffset}`);
    history=reset?data.items:[...history,...data.items];historyOffset=history.length;hasMore=data.items.length===30;
    const list=$("#conversationList");list.replaceChildren();
    for(const conversation of history){
      const row=document.createElement("div");row.className="history-item"+(conversation.id===current?.id?" selected":"");
      const select=document.createElement("button");select.className="history-select";select.textContent=conversation.title||"新会话";select.title=select.textContent;select.onclick=()=>open(conversation.id);
      const rename=document.createElement("button");rename.textContent="改名";rename.setAttribute("aria-label","重命名会话");rename.onclick=async()=>{
        const title=prompt("会话名称（最多15个字）",conversation.title);if(!title?.trim())return;
        if([...title.trim()].length>15){toast("会话标题最多15个字");return}
        try{await api('/internal/conversations/'+conversation.id,{method:"PATCH",headers:{"Content-Type":"application/json"},body:JSON.stringify({title:title.trim()})});await loadHistory();if(current?.id===conversation.id)await refresh()}catch(error){toast(error.message)}
      };
      const remove=document.createElement("button");remove.textContent="删除";remove.setAttribute("aria-label","删除会话");remove.onclick=async()=>{
        if(!confirm(conversation.video_id?"删除此会话将同时删除对应视频、该视频的全部历史对话、报告、向量和本地缓存。确定继续？":"删除这条未绑定视频的会话及附件缓存？"))return;
        remove.disabled=true;
        try{const result=await api('/internal/conversations/'+conversation.id,{method:"DELETE",headers:{'X-VideoLens-Linked-Delete':'1'}});await deleted(result);toast(result.video_id?"已删除视频及全部关联对话":"已删除会话")}catch(error){toast(error.message)}finally{remove.disabled=false}
      };
      row.append(select,rename,remove);list.append(row);
    }
    $("#moreConversations").hidden=!hasMore;
  }
  async function deleted(result,broadcast=true){
    const ids=result.conversation_ids||[];
    if(current&&(ids.includes(current.id)||(result.video_id&&current.video_id===result.video_id))){
      selection++;source?.close();source=null;current=snapshot=null;
      streamText="";streamOperation="";requestPending=null;upload=null;file=null;draftHint="";
      $("#messageInput").value="";$("#videoPanel").hidden=true;
      $("#player").pause?.();$("#player").removeAttribute("src");
      historyState();preview();render();
    }
    await loadHistory();
    if(typeof scope.loadLibrary==='function')await scope.loadLibrary();
    if(broadcast)try{localStorage.setItem('videolens.deleted',JSON.stringify({video_id:result.video_id||'',conversation_ids:ids,time:Date.now()}))}catch(error){}
  }
  function historyState(){const url=new URL(location.href);url.searchParams.delete("video_id");current?url.searchParams.set("conversation",current.id):url.searchParams.delete("conversation");scope.history.replaceState(null,"",url)}
  function updateSnapshot(value){
    if(value.conversation.id!==current?.id)return;
    const prior=snapshot?.operation?.status,priorTitle=current?.title;
    current=value.conversation;snapshot=value;
    if(!["queued","running"].includes(value.operation?.status)){streamText="";streamOperation=""}
    const player=$("#player"),path=current.video_id?`/internal/videos/${current.video_id}/file`:"";
    if(player.getAttribute("src")!==path){path?player.setAttribute("src",path):player.removeAttribute("src");$("#playerHint").textContent=""}
    $("#videoPanel").hidden=!current.video_id;
    render();
    if(["queued","running"].includes(value.operation?.status))connectEvents();
    else{source?.close();source=null}
    if(prior!==value.operation?.status||priorTitle!==current.title)loadHistory().catch(error=>toast(error.message));
  }
  async function refresh(){if(!current)return;const id=current.id;const value=await api('/internal/conversations/'+id);if(current?.id===id)updateSnapshot(value)}
  async function open(cid){
    if(uploadBusy){toast("请等待上传结束");return}
    const serial=++selection;source?.close();source=null;streamText="";streamOperation="";requestPending=null;upload=null;file=null;preview();
    const data=await api('/internal/conversations/'+cid);if(serial!==selection)return;
    current=data.conversation;snapshot=null;historyState();updateSnapshot(data);
    setHistory(false);
    await loadHistory();
  }
  function connectEvents(){
    if(source||!current||typeof EventSource==="undefined")return;
    const cid=current.id,serial=selection;
    {
      source=new EventSource(`/internal/conversations/${cid}/events`);
      source.onmessage=event=>{
        if(current?.id!==cid||serial!==selection)return;
        const value=JSON.parse(event.data);
        if(value.type==="snapshot")updateSnapshot(value);
        if(value.type==="progress"&&snapshot?.operation){Object.assign(snapshot.operation,{phase:value.phase,progress:value.progress,message:value.message,status:"running"});render()}
        if(value.type==="delta"){
          if(streamOperation!==value.operation_id){streamOperation=value.operation_id;streamText=""}
          streamText+=value.text;render();
        }
      };
    }
  }
  function setHistory(show){
    $("#historySidebar").hidden=!show;$("#historyBackdrop").hidden=!show;
    $("#historyToggle").setAttribute("aria-expanded",String(show));
    if(show)$("#historyClose").focus();
  }
  async function newConversation(){
    if(uploadBusy||submitting){toast("请等待当前提交完成");return}
    selection++;source?.close();source=null;current=snapshot=null;
    streamText="";streamOperation="";requestPending=null;upload=null;file=null;draftHint="";
    $("#messageInput").value="";$("#videoPanel").hidden=true;
    $("#player").pause?.();$("#player").removeAttribute("src");
    historyState();preview();render();setHistory(false);
    document.querySelector('[data-t="analyze"]').click();$("#messageInput").focus();
  }
  async function openVideo(videoId,conversationId=""){
    document.querySelector('[data-t="analyze"]').click();
    if(conversationId){await open(conversationId);return}
    const data=await api('/internal/conversations?limit=100');
    const existing=data.items.find(c=>c.video_id===videoId);
    const conversation=existing||await post('/internal/conversations',{video_id:videoId});
    await open(conversation.id);
  }
  function preview(){
    const box=$("#attachmentPreview");box.hidden=!file;box.replaceChildren();
    if(file){const text=document.createElement("span");text.textContent=`${file.name} · ${(file.size/1048576).toFixed(1)} MB · ${file.path||""}`;
      const remove=document.createElement("button");remove.type="button";remove.textContent="删除";remove.setAttribute("aria-label","移除视频附件");remove.onclick=()=>{if(!uploadBusy){file=null;upload=null;preview()}};box.append(text,remove)}
  }
  function localDescriptor(data){return {...data,name:data.filename,size:data.size_bytes,lastModified:data.last_modified,local:true}}
  async function chooseLocalVideo(){
    if(active()){toast("当前会话正在处理，请稍后选择视频");return}
    if(current?.video_id){toast("当前会话已绑定视频，请新建会话选择");return}
    let chosen=null;uploadBusy=true;buttons();
    try{chosen=await post('/internal/local-videos/select',{})}catch(error){toast(error.message)}
    finally{uploadBusy=false;buttons()}
    if(chosen&&!chosen.cancelled)await selectFile(localDescriptor(chosen));
  }
  async function selectFile(value){
    if(!value)return;
    if(active()){toast("当前会话正在处理，请稍后添加视频");return}
    if(current?.video_id){toast("当前会话已绑定视频，请新建会话选择");return}
    if(value.size>8*1073741824){toast("视频不能超过8 GiB");return}
    try{
      if(!value.local){
        const path=value.path||prompt("请输入此视频的完整本地路径，以使用原文件建立链接","");
        if(!path)return;
        const selected=await post('/internal/local-videos/inspect',{path});
        if(selected.filename!==value.name||selected.size_bytes!==value.size)throw new Error("该路径与所选视频不一致，请重新选择");
        value=localDescriptor(selected);
      }
      file=value;upload=null;requestPending=null;preview();
    }catch(error){toast(error.message)}
  }
  async function attachLocalFile(cid,video){
    const status=$("#operationStatus");status.hidden=false;
    status.textContent='正在校验原文件并创建本地链接…';
    return post(`/internal/conversations/${cid}/local-video`,{path:video.path,size_bytes:video.size,last_modified:video.lastModified});
  }
  async function send(){
    const text=$("#messageInput").value.trim();if(active()||(!text&&!file))return;
    if(!current&&!file&&!/https?:\/\/[^\s<>]+/i.test(text)){
      draftHint="请先添加一个视频附件，或粘贴含 HTTP / HTTPS 链接的视频分享文本。也可以在同一条消息中附上问题。";render();return;
    }
    draftHint="";submitting=true;buttons();
    try{
      if(!current){
        const pendingFile=file;
        const data=await post('/internal/conversations',{});
        await open(data.id);file=pendingFile;preview();
      }
      const cid=current.id;uploadBusy=true;buttons();
      if(file&&!upload)upload=await attachLocalFile(cid,file);
      const options={title:$("#title").value.trim(),language:$("#lang").value||null,force:$("#force").checked,
        skip_vision:$("#skipVision").checked,skip_summary:$("#skipSummary").checked,skip_vectors:$("#skipVectors").checked};
      const key=JSON.stringify([text,upload?.id||"",options]);
      if(!requestPending||requestPending.key!==key)requestPending={key,id:uid()};
      await post(`/internal/conversations/${cid}/messages`,{text,attachment_id:upload?.id||"",request_id:requestPending.id,options});
      $("#messageInput").value="";file=null;upload=null;requestPending=null;preview();streamText="";
      await refresh();await loadHistory();
    }catch(error){toast(error.message);$("#operationStatus").hidden=false;$("#operationStatus").textContent=error.message}
    finally{uploadBusy=false;submitting=false;buttons()}
  }
  async function init(){
    $("#newConversation").onclick=()=>newConversation().catch(error=>toast(error.message));
    $("#draftNew").onclick=()=>newConversation().catch(error=>toast(error.message));
    $("#historyToggle").onclick=()=>{document.querySelector('[data-t="analyze"]').click();const show=$("#historySidebar").hidden;setHistory(show);if(show)loadHistory().catch(error=>toast(error.message))};
    $("#historyClose").onclick=()=>{setHistory(false);$("#historyToggle").focus()};
    $("#historyBackdrop").onclick=$("#historyClose").onclick;
    document.addEventListener("keydown",event=>{if(event.key==="Escape"&&!$("#historySidebar").hidden)$("#historyClose").click()});
    $("#moreConversations").onclick=()=>loadHistory(false).catch(error=>toast(error.message));
    $("#attachVideo").onclick=()=>chooseLocalVideo();
    $("#sendMessage").onclick=()=>send();
    $("#composer").onsubmit=event=>{event.preventDefault();send()};
    $("#messageInput").onkeydown=event=>{if(event.key==="Enter"&&!event.shiftKey&&!event.isComposing){event.preventDefault();send()}};
    const main=$("#conversationMain");main.ondragover=event=>{event.preventDefault();main.classList.add("drag-over")};
    main.ondragleave=event=>{if(!main.contains(event.relatedTarget))main.classList.remove("drag-over")};
    main.ondrop=event=>{event.preventDefault();main.classList.remove("drag-over");if(event.dataTransfer.files.length!==1){toast("每次只能添加一个视频");return}selectFile(event.dataTransfer.files[0])};
    $("#player").onerror=()=>{$("#playerHint").textContent="视频文件不可用或浏览器不支持此编码，分析报告与历史仍可查看。"};
    scope.addEventListener("pagehide",()=>source?.close());
    scope.addEventListener('storage',event=>{if(event.key==='videolens.deleted'&&event.newValue)try{deleted(JSON.parse(event.newValue),false).catch(error=>toast(error.message))}catch(error){}});
    await loadHistory();
    const params=new URLSearchParams(location.search);
    try{
      if(params.get("conversation"))await open(params.get("conversation"));
      else if(params.get("video_id"))await openVideo(params.get("video_id"));
      else render();
    }catch(error){toast(error.message);await newConversation()}
    setHistory(false);
  }
  scope.VideoLensWorkspace={init,render,openVideo,open,send,selectFile,updateSnapshot,reportHtml,deleted};
})(globalThis);
