"use strict";
/* 主题只改变展示；即时应用、持久化，并跟随系统/其他标签页变化。 */
(function(scope){
  const valid=value=>['system','light','dark'].includes(value);
  const media=scope.matchMedia?.('(prefers-color-scheme: dark)');
  let mode='system',saving=Promise.resolve();
  try{const saved=localStorage.getItem('videolens.theme');if(valid(saved))mode=saved}catch(error){}
  function apply(value){
    mode=valid(value)?value:'system';
    const resolved=mode==='system'?(media?.matches?'dark':'light'):mode;
    document.documentElement.dataset.theme=resolved;
    document.documentElement.style.colorScheme=resolved;
    scope.VideoLensFluent?.applyTheme(resolved);
    for(const id of ['themeQuick','theme_mode']){const select=document.getElementById(id);if(select)select.value=mode}
    try{localStorage.setItem('videolens.theme',mode)}catch(error){}
  }
  function save(value){
    apply(value);
    const selected=mode;
    // 顺序保存，避免快速切换时旧请求覆盖最后一次选择。
    saving=saving.catch(()=>{}).then(async()=>{
      const response=await fetch('/internal/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({'ui.theme_mode':selected})});
      if(!response.ok)throw new Error('主题已在本页生效，但保存失败，请重试');
    }).catch(error=>scope.toast?.(error.message));
    return saving;
  }
  media?.addEventListener?.('change',()=>{if(mode==='system')apply(mode)});
  scope.addEventListener('storage',event=>{if(event.key==='videolens.theme')apply(event.newValue)});
  scope.VideoLensTheme={apply,save,get mode(){return mode}};
  apply(mode);
  document.getElementById('themeQuick').onchange=event=>save(event.target.value);
  document.getElementById('theme_mode').onchange=event=>save(event.target.value);
})(globalThis);
