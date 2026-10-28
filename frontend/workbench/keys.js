"use strict";

/* 隐藏时使用独立的固定圆点遮罩，不按真实密钥长度绘制密码字符。 */
(function(){
  const fields=[];
  function create(prefix,key,name){
    const input=document.getElementById(`${prefix}_key`),control=document.getElementById(`${prefix}_key_control`);
    const mask=document.getElementById(`${prefix}_key_mask`),toggle=document.getElementById(`${prefix}_key_toggle`);
    const clear=document.getElementById(`${prefix}_clear_key`);
    let configured=false,dirty=false,visible=false,generation=0;
    function render(){
      control.classList.toggle("key-hidden",!visible);
      input.type=visible?"text":"password";
      mask.hidden=visible||!(input.value||(!dirty&&configured));
      input.placeholder=input.value||(!dirty&&configured)?"":"请输入完整 API Key";
      toggle.textContent=visible?"隐藏":"显示";
      toggle.setAttribute("aria-pressed",String(visible));
      toggle.setAttribute("aria-label",`${visible?"隐藏":"显示"}${name} API Key`);
    }
    function hide(){
      generation++;visible=false;toggle.disabled=false;
      // 已保存密钥仅在显示期间留在输入框；新输入的修改保留至保存。
      if(!dirty)input.value="";
      render();
    }
    input.addEventListener("input",()=>{dirty=true;render()});
    toggle.onclick=async()=>{
      if(visible){hide();return}
      const token=++generation;
      toggle.disabled=true;
      try{
        if(!dirty&&configured){
          const response=await fetch("/internal/settings/reveal-key",{method:"POST",cache:"no-store",headers:{"Content-Type":"application/json"},body:JSON.stringify({key})});
          if(!response.ok)throw new Error("无法读取 API Key，请稍后重试");
          const result=await response.json();
          if(token!==generation)return;
          // 请求期间输入了新密钥时，不能用服务器返回值覆盖用户输入。
          if(!dirty)input.value=result.value||"";
        }
        if(token===generation){visible=true;render()}
      }catch(error){
        if(token===generation)document.getElementById("saveSetHint").textContent="无法显示 API Key，请检查服务后重试。";
      }finally{if(token===generation)toggle.disabled=false}
    };
    return {
      load(settings){
        configured=settings.secret_present?.[key]??Boolean(settings.values?.[key]);
        dirty=false;input.value="";clear.checked=false;hide();
      },
      append(payload){
        if(dirty&&input.value.trim())payload[key]=input.value;
        else if(clear.checked)payload[key]="";
      },
      hide,
    };
  }
  fields.push(create("llm","llm.cloud.api_key","大模型"),create("vl","vl.cloud_api_key","视觉理解"));
  window.VideoLensKeys={
    load:settings=>fields.forEach(field=>field.load(settings)),
    append:payload=>fields.forEach(field=>field.append(payload)),
    hide:()=>fields.forEach(field=>field.hide()),
  };
  document.querySelectorAll(".tabs button").forEach(button=>button.addEventListener("click",()=>window.VideoLensKeys.hide()));
  document.addEventListener("visibilitychange",()=>{if(document.hidden)window.VideoLensKeys.hide()});
  window.addEventListener("pagehide",()=>window.VideoLensKeys.hide());
})();
