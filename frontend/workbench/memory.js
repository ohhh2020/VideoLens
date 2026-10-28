"use strict";

/* 只渲染记忆文段；向量由后端同步保存，界面不展示原始数组。 */
(function(scope){
  const $=id=>document.getElementById(id);
  let timer=null,loading=false,optionsBusy=false;
  const kinds=[["interest","视频兴趣"],["question","问题关注点"],["style","回答风格"]];
  function notice(text){$("memoryStatus").textContent=text}
  async function request(path,method="GET",data){
    const response=await fetch("/internal/memory"+path,{method,headers:{"Content-Type":"application/json"},body:data===undefined?undefined:JSON.stringify(data)});
    const result=await response.json();
    if(!response.ok)throw new Error(typeof result.detail==="string"?result.detail:"记忆操作失败");
    return result;
  }
  function pause(){clearTimeout(timer)}
  function renderItem(item,limit){
    const card=document.createElement("div");card.className="card memory-item";card.dataset.memoryId=item.id;
    const meta=document.createElement("p");meta.className="muted";
    meta.textContent=(item.origin==="manual"?"手动记忆":"历史归纳")+" · "+new Date(item.updated_at).toLocaleString("zh-CN")+(item.evidence.length?" · "+item.evidence.length+" 条历史依据":"");
    const input=document.createElement("textarea");input.value=item.text;input.rows=2;input.maxLength=limit;input.setAttribute("aria-label","记忆文段");
    const select=document.createElement("select");select.setAttribute("aria-label","记忆类型");
    for(const [value,label] of kinds){const option=document.createElement("option");option.value=value;option.textContent=label;select.append(option)}
    select.value=item.kind;
    const actions=document.createElement("div");actions.className="memory-actions";
    const save=document.createElement("button");save.className="btn ghost";save.textContent="保存修改";
    const remove=document.createElement("button");remove.className="btn ghost";remove.textContent="删除记忆";
    save.onclick=async()=>{save.disabled=true;try{await request("/items/"+encodeURIComponent(item.id),"PUT",{text:input.value,kind:select.value});save.blur();notice("记忆已修改并重新向量化");await load()}catch(e){notice(e.message)}finally{save.disabled=false}};
    remove.onclick=async()=>{remove.disabled=true;try{await request("/items/"+encodeURIComponent(item.id),"DELETE");remove.blur();await load()}catch(e){notice(e.message);remove.disabled=false}};
    actions.append(save,remove);card.append(meta,input,select,actions);
    if(item.evidence.length){
      const details=document.createElement("details"),summary=document.createElement("summary"),body=document.createElement("p");
      summary.textContent="查看历史依据";body.className="muted";body.textContent=item.evidence.map(e=>e.label+"（"+e.id+"）").join("；");
      details.append(summary,body);card.append(details);
    }
    return card;
  }
  async function load(){
    pause();if(loading)return;loading=true;
    try{
      const state=await request("");
      if(!optionsBusy){$("memoryEnabled").checked=state.enabled;$("memoryAuto").checked=state.auto_learn}
      $("memoryNew").maxLength=state.max_text_chars;
      $("memoryRefresh").disabled=state.learning||!state.auto_learn;
      notice(`共 ${state.items.length} 条记忆。`+(state.learning?"正在归纳历史…":state.last_error||"记忆可即时用于后续生成。"));
      $("memoryFile").textContent="本地文字与向量映射："+state.file;
      const list=$("memoryList");
      if(!list.contains(document.activeElement)){
        list.replaceChildren(...state.items.map(item=>renderItem(item,state.max_text_chars)));
        if(!state.items.length){const empty=document.createElement("p");empty.className="muted";empty.textContent="暂无记忆。历史依据不足时不会强行归纳，也可以直接手动添加。";list.append(empty)}
      }
      if(state.learning&&$("p-memory").classList.contains("on"))timer=setTimeout(load,2000);
    }catch(e){notice(e.message)}finally{loading=false}
  }
  async function options(){
    optionsBusy=true;$("memoryEnabled").disabled=$("memoryAuto").disabled=true;
    try{await request("/options","PUT",{enabled:$("memoryEnabled").checked,auto_learn:$("memoryAuto").checked});await load()}
    catch(e){notice(e.message)}finally{optionsBusy=false;$("memoryEnabled").disabled=$("memoryAuto").disabled=false}
  }
  $("memoryEnabled").onchange=options;$("memoryAuto").onchange=options;
  $("memoryRefresh").onclick=async()=>{try{await request("/refresh","POST",{});await load()}catch(e){notice(e.message)}};
  $("memoryReload").onclick=load;
  $("memoryAdd").onclick=async()=>{
    const button=$("memoryAdd");button.disabled=true;
    try{await request("/items","POST",{text:$("memoryNew").value,kind:$("memoryNewKind").value});$("memoryNew").value="";await load()}
    catch(e){notice(e.message)}finally{button.disabled=false}
  };
  window.addEventListener("pagehide",pause);
  scope.VideoLensMemory={load,pause};
})(globalThis);
