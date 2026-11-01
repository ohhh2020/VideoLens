/* 统一记忆空间、弹窗编辑与原子批量删除；测试数据仅存在本进程。 */
"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),path=require("node:path");
const {JSDOM}=require("jsdom");
const root=path.resolve(__dirname,".."),dom=new JSDOM(fs.readFileSync(path.join(root,"frontend/workbench/index.html"),"utf8"),{url:"http://127.0.0.1:8000/ui",runScripts:"outside-only"});
const w=dom.window,d=w.document,calls=[];
w.HTMLDialogElement.prototype.showModal=function(){this.setAttribute('open','')};
w.HTMLDialogElement.prototype.close=function(){this.removeAttribute('open')};
let next=1,confirmed=true,deleteFails=false,saveFails=false;
w.confirm=()=>confirmed;
const state={enabled:true,auto_learn:true,learning:false,last_error:"",max_text_chars:220,file:"runtime/data/memory/preferences.json",items:[]};
w.fetch=async(url,options={})=>{
  const method=options.method||"GET",body=options.body?JSON.parse(options.body):null;
  calls.push({url,method,body});let data=state;
  if(saveFails&&method==='PUT'&&url.includes('/items/'))return {ok:false,json:async()=>({detail:'保存失败，保留草稿'})};
  if(method==="POST"&&url.endsWith("/items")){const item={id:"m"+next++,text:body.text,kind:body.kind||'interest',origin:"manual",evidence:[],updated_at:new Date().toISOString()};state.items.push(item);data=item}
  if(method==="PUT"&&url.includes("/items/")){const item=state.items.find(i=>url.endsWith(i.id));Object.assign(item,body);data=item}
  if(url.endsWith('/delete-many')){
    if(deleteFails)return {ok:false,json:async()=>({detail:'删除失败，原记忆已保留'})};
    state.items=state.items.filter(item=>!body.ids.includes(item.id));data={deleted:true,deleted_count:body.ids.length,ids:body.ids};
  }
  if(method==="PUT"&&url.endsWith("/options"))Object.assign(state,body);
  return {ok:true,json:async()=>structuredClone(data)};
};
w.eval(fs.readFileSync(path.join(root,"frontend/workbench/memory.js"),"utf8"));
const save=()=>d.querySelector('#memoryEditorForm').onsubmit({preventDefault(){}});
const select=(index,value=true)=>{const c=d.querySelectorAll('#memoryList input[type=checkbox]')[index];c.checked=value;c.onchange()};
async function main(){
  await w.VideoLensMemory.load();
  assert.equal(d.querySelector('#memorySpace').open,false,'初次进入空间折叠');
  assert.equal(d.querySelectorAll('#memoryNewKind').length,0,'不要求分类');
  assert.ok(d.querySelector('#memoryList').textContent.includes('暂无记忆'));
  d.querySelector('#memoryCreate').onclick();
  assert.equal(d.querySelector('#memoryEditor').open,true);
  d.querySelector('#memoryNew').value='用户希望看到步骤。<img src=x onerror="alert(1)">';await save();
  assert.equal(state.items.length,1);assert.equal(d.querySelector('#memorySpace').open,true);
  assert.equal(d.querySelector('#memoryEditor').open,false);assert.equal(d.querySelector('#memoryList img'),null);
  assert.ok(d.querySelector('.memory-text').textContent.includes('<img'));
  // 既有内部属性保留，界面统一展示文字；模态外刷新不丢草稿。
  state.items[0].kind='style';await w.VideoLensMemory.load();
  d.querySelector('.memory-row-actions button').onclick();
  const input=d.querySelector('#memoryNew');input.value='未保存的编辑草稿';input.blur();
  await w.VideoLensMemory.load();assert.equal(input.value,'未保存的编辑草稿');
  saveFails=true;await save();assert.equal(d.querySelector('#memoryEditor').open,true);
  assert.ok(d.querySelector('#memoryEditorStatus').textContent.includes('保存失败'));
  saveFails=false;await save();assert.equal(state.items[0].text,'未保存的编辑草稿');assert.equal(state.items[0].kind,'style');
  for(const text of ['用户关注CPU硬件数据。','用户希望说明软件配置。']){d.querySelector('#memoryCreate').onclick();input.value=text;await save()}
  assert.equal(state.items.length,3);
  select(0);select(1);assert.equal(d.querySelector('#memorySelectAll').indeterminate,true);
  d.querySelector('#memorySpace').open=false;await w.VideoLensMemory.load();
  assert.equal(d.querySelector('#memorySpace').open,false,'刷新保留空间折叠状态');
  assert.equal(d.querySelectorAll('#memoryList input:checked').length,2,'刷新保留勾选');
  confirmed=false;const before=calls.length;await d.querySelector('#memoryDeleteSelected').onclick();
  assert.equal(calls.length,before,'取消确认不发送删除');confirmed=true;deleteFails=true;
  await d.querySelector('#memoryDeleteSelected').onclick();assert.equal(state.items.length,3);
  assert.equal(d.querySelectorAll('#memoryList input:checked').length,2,'失败保留选择');deleteFails=false;
  await d.querySelector('#memoryDeleteSelected').onclick();assert.equal(state.items.length,1);
  assert.equal(d.querySelector('#memoryDeleteSelected').disabled,true);
  const batches=calls.filter(c=>c.url.endsWith('/delete-many'));assert.equal(batches.at(-1).body.ids.length,2);
  assert.ok(!calls.some(c=>c.method==='DELETE'),'只发送一个原子批量接口请求');
  d.querySelector('#memorySelectAll').checked=true;d.querySelector('#memorySelectAll').onchange();
  d.querySelector('#memoryClearSelection').onclick();assert.equal(d.querySelectorAll('#memoryList input:checked').length,0);
  d.querySelector('#memoryEnabled').checked=false;await d.querySelector('#memoryEnabled').onchange();assert.equal(state.enabled,false);
  d.querySelector('.memory-row-actions .danger').onclick();await new Promise(resolve=>setTimeout(resolve,0));
  assert.equal(state.items.length,0,'单条删除走同一原子接口');
  console.log('PASS：单一折叠空间、无分类、新建/编辑弹窗、刷新草稿与勾选保护、多选/全选/取消、原子删除与失败恢复');
  dom.window.close();
}
main().catch(error=>{console.error(error);dom.window.close();process.exitCode=1});
