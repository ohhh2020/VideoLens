"use strict";

/* 所有文字记忆集中展示；选择、展开与编辑草稿不因状态刷新丢失。 */
(function(scope){
  const $=id=>document.getElementById(id);
  let timer=null,loading=null,optionsBusy=false,mutating=false,editing=null,limit=220,items=[];
  const selected=new Set();
  function notice(text){$("memoryStatus").textContent=text}
  async function request(path,method="GET",data){
    const response=await fetch("/internal/memory"+path,{method,headers:{"Content-Type":"application/json"},body:data===undefined?undefined:JSON.stringify(data)});
    const result=await response.json();
    if(!response.ok)throw new Error(typeof result.detail==="string"?result.detail:"记忆操作失败，请刷新后重试");
    return result;
  }
  function pause(){clearTimeout(timer)}
  function selection(){
    $("memorySelectionCount").textContent=`已选 ${selected.size} 条`;
    $("memorySelectAll").checked=items.length>0&&selected.size===items.length;
    $("memorySelectAll").indeterminate=selected.size>0&&selected.size<items.length;
    $("memorySelectAll").disabled=mutating||!items.length;
    $("memoryClearSelection").disabled=mutating||!selected.size;
    $("memoryDeleteSelected").disabled=mutating||!selected.size;
    $("memoryCreate").disabled=mutating;
    document.querySelectorAll("#memoryList input,#memoryList button").forEach(control=>{control.disabled=mutating});
    document.querySelectorAll("#memoryList input[type=checkbox]").forEach(control=>{control.checked=selected.has(control.value)});
  }
  function showEditor(item=null){
    if(mutating)return;
    editing=item?{...item}:null;
    $("memoryEditorTitle").textContent=item?"修改记忆":"创建记忆";
    $("memoryAdd").textContent=item?"保存修改":"创建记忆";
    $("memoryNew").value=item?.text||"";
    $("memoryNew").maxLength=limit;
    $("memoryEditorStatus").textContent="";
    const dialog=$("memoryEditor");
    if(!dialog.open){if(dialog.showModal)dialog.showModal();else dialog.setAttribute("open","")}
    $("memoryNew").focus();
  }
  function closeEditor(){
    if(mutating)return;
    const dialog=$("memoryEditor");
    if(dialog.close)dialog.close();else dialog.removeAttribute("open");
    editing=null;
  }
  function editorBusy(value){
    mutating=value;
    for(const id of ["memoryNew","memoryAdd","memoryEditorCancel","memoryEditorClose"])$(id).disabled=value;
    selection();
  }
  async function save(event){
    event?.preventDefault();if(mutating)return;
    const text=$("memoryNew").value.trim();
    if(!text||[...text].length>limit){$("memoryEditorStatus").textContent=`请填写1–${limit}个字符的记忆。`;return}
    const current=editing,payload={text};
    // 保留旧记录的内部属性，界面不再按类别分组或要求用户分类。
    if(current)payload.kind=current.kind;
    editorBusy(true);$("memoryEditorStatus").textContent="正在保存文字与向量…";
    try{
      await request(current?"/items/"+encodeURIComponent(current.id):"/items",current?"PUT":"POST",payload);
      editorBusy(false);closeEditor();$("memorySpace").open=true;
      await load();notice(current?"记忆已修改并重新向量化。":"记忆已创建，可用于后续总结和对话。");
    }catch(error){$("memoryEditorStatus").textContent=error.message}
    finally{editorBusy(false)}
  }
  async function remove(ids){
    if(mutating||!ids.length)return;
    if(!scope.confirm(`确定删除这 ${ids.length} 条记忆？文字和对应向量会一起删除，此操作无法撤销。`))return;
    mutating=true;selection();
    try{
      const result=await request("/items/delete-many","POST",{ids});
      for(const id of ids)selected.delete(id);
      await load();notice(`已删除 ${result.deleted_count} 条记忆。`);
    }catch(error){notice(error.message)}finally{mutating=false;selection()}
  }
  function renderItem(item){
    const row=document.createElement("article");row.className="memory-item";row.dataset.memoryId=item.id;
    const checkbox=document.createElement("input");checkbox.type="checkbox";checkbox.value=item.id;
    checkbox.setAttribute("aria-label","选择记忆："+item.text);checkbox.checked=selected.has(item.id);
    checkbox.onchange=()=>{checkbox.checked?selected.add(item.id):selected.delete(item.id);selection()};
    const body=document.createElement("div");body.className="memory-item-body";
    const text=document.createElement("p");text.className="memory-text";text.textContent=item.text;
    const meta=document.createElement("p");meta.className="memory-meta";
    meta.textContent=(item.origin==="manual"?"手动记录":"历史归纳")+" · "+new Date(item.updated_at).toLocaleString("zh-CN");
    body.append(text,meta);
    if(item.evidence?.length){
      const details=document.createElement("details"),summary=document.createElement("summary"),sources=document.createElement("p");
      details.className="memory-evidence";summary.textContent=`${item.evidence.length} 条历史依据`;
      sources.textContent=item.evidence.map(e=>e.label+"（"+e.id+"）").join("；");
      details.append(summary,sources);body.append(details);
    }
    const actions=document.createElement("div");actions.className="memory-row-actions";
    const edit=document.createElement("button");edit.type="button";edit.className="btn ghost";edit.textContent="修改";
    edit.onclick=()=>showEditor(item);
    const erase=document.createElement("button");erase.type="button";erase.className="btn danger";erase.textContent="删除";
    erase.onclick=()=>remove([item.id]);actions.append(edit,erase);row.append(checkbox,body,actions);
    return row;
  }
  function load(){
    pause();
    // 保存后若有旧刷新尚未返回，等旧请求结束再取新状态，不能吞掉本次刷新。
    if(loading)return loading.then(load);
    loading=readState().finally(()=>{loading=null});
    return loading;
  }
  async function readState(){
    try{
      const state=await request("");items=state.items;limit=state.max_text_chars;
      if(!optionsBusy){$("memoryEnabled").checked=state.enabled;$("memoryAuto").checked=state.auto_learn}
      $("memoryNew").maxLength=limit;$("memoryRefresh").disabled=state.learning||!state.auto_learn;
      notice(`共 ${items.length} 条记忆。`+(state.learning?"正在归纳历史…":state.last_error||"记忆可即时用于后续生成。"));
      $("memoryFile").textContent="本地文字与向量映射："+state.file;
      $("memoryCount").textContent=items.length+" 条";
      const valid=new Set(items.map(item=>item.id));for(const id of selected)if(!valid.has(id))selected.delete(id);
      const list=$("memoryList"),openEvidence=new Set([...list.querySelectorAll('.memory-item')].filter(row=>row.querySelector('details')?.open).map(row=>row.dataset.memoryId));
      list.replaceChildren(...items.map(item=>renderItem(item)));
      list.querySelectorAll('.memory-item').forEach(row=>{const details=row.querySelector('details');if(details&&openEvidence.has(row.dataset.memoryId))details.open=true});
      if(!items.length){const empty=document.createElement("p");empty.className="memory-empty muted";empty.textContent="暂无记忆。点击“创建记忆”，记录希望知影记住的内容。";list.append(empty)}
      selection();
      if(location.hash==="#memoryNew"){
        showEditor();history.replaceState(null,"",location.pathname+location.search+"#memory");
      }
      if(state.learning&&$("p-memory").classList.contains("on"))timer=setTimeout(load,2000);
    }catch(error){notice(error.message)}
  }
  async function options(){
    optionsBusy=true;$("memoryEnabled").disabled=$("memoryAuto").disabled=true;
    try{await request("/options","PUT",{enabled:$("memoryEnabled").checked,auto_learn:$("memoryAuto").checked});await load()}
    catch(error){notice(error.message)}finally{optionsBusy=false;$("memoryEnabled").disabled=$("memoryAuto").disabled=false}
  }
  $("memoryEnabled").onchange=options;$("memoryAuto").onchange=options;
  $("memoryRefresh").onclick=async()=>{try{await request("/refresh","POST",{});await load()}catch(error){notice(error.message)}};
  $("memoryReload").onclick=load;$("memoryCreate").onclick=()=>showEditor();
  $("memoryEditorForm").onsubmit=save;
  $("memoryEditorClose").onclick=$("memoryEditorCancel").onclick=closeEditor;
  $("memoryEditor").addEventListener("cancel",event=>{if(mutating)event.preventDefault();else editing=null});
  $("memorySelectAll").onchange=()=>{selected.clear();if($("memorySelectAll").checked)items.forEach(item=>selected.add(item.id));selection()};
  $("memoryClearSelection").onclick=()=>{selected.clear();selection()};
  $("memoryDeleteSelected").onclick=()=>remove([...selected]);
  window.addEventListener("pagehide",pause);
  scope.VideoLensMemory={load,pause};
})(globalThis);
