/* 实际网页脚本：双向删除刷新、同视频多会话、跨标签同步与失败不冒报成功。 */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {JSDOM}=require('jsdom');
const read=file=>fs.readFileSync(path.join(__dirname,'..','frontend',file),'utf8');
const dom=new JSDOM(read('workbench/index.html'),{url:'http://127.0.0.1:8000/ui?conversation=c2',runScripts:'outside-only'}),w=dom.window;
let conversations=[{id:'c1',video_id:'v1',title:'第一会话'},{id:'c2',video_id:'v1',title:'第二会话'},{id:'c3',video_id:'v2',title:'其他视频'}];
let videos=[{video_id:'v1',title:'第一视频',status:'ready'},{video_id:'v2',title:'其他视频',status:'ready'}],busy=false;
const prompts=[];w.confirm=text=>{prompts.push(text);return true};
w.HTMLElement.prototype.scrollIntoView=function(){};w.HTMLMediaElement.prototype.pause=function(){};
w.EventSource=class{close(){}};
function erase(video){const ids=conversations.filter(c=>c.video_id===video).map(c=>c.id);conversations=conversations.filter(c=>c.video_id!==video);videos=videos.filter(v=>v.video_id!==video);return {deleted:true,video_id:video,conversation_ids:ids}}
w.fetch=async(url,options={})=>{
  let data={},ok=true;
  if(options.method==='DELETE'){
    if(busy){ok=false;data={detail:'该视频正在问答，请等待完成'}}
    else {const id=url.split('/').pop(),video=url.startsWith('/internal/conversations/')?conversations.find(c=>c.id===id).video_id:id;data=erase(video)}
  }else if(url.startsWith('/internal/conversations?'))data={items:conversations};
  else if(url.startsWith('/internal/conversations/')){const c=conversations.find(c=>c.id===url.split('/').pop());data={conversation:c,messages:[{id:1,role:'assistant',kind:'report',text:'',payload:{video_id:c.video_id,summary:'## 内容概览\n当前报告'}}],operation:null}}
  else if(url.startsWith('/internal/videos?'))data=videos;
  else if(url==='/internal/settings')data={values:{},effective:{'ui.report_mode':'user'},backends:[]};
  else if(url.includes('/health'))data={status:'ok'};
  else data={usage:{},keyframe_bytes:0};
  return {ok,json:async()=>data};
};
for(const file of ['vendor/marked.umd.js','vendor/purify.min.js','shared/markdown.js','workbench/keys.js','workbench/conversations.js'])w.eval(read(file));
w.eval(read('workbench/app.js').replace('\ninitialize();','\n')+'\nObject.assign(window,{initialize,loadLibrary,toast});');
(async()=>{
  await w.initialize();
  w.confirm=()=>false;
  await w.document.querySelector('#conversationList [aria-label="删除会话"]').onclick();
  assert.equal(conversations.length,3,'取消确认不会执行删除');
  w.confirm=text=>{prompts.push(text);return true};
  await w.document.querySelector('#conversationList [aria-label="删除会话"]').onclick();
  assert.ok(prompts[0].includes('全部历史对话'));
  assert.deepEqual(conversations.map(c=>c.id),['c3']);
  assert.equal(w.document.querySelector('#conversationFlow').textContent.includes('当前报告'),false,'删除同视频另一会话也清空当前页面');
  assert.equal(new URL(w.location.href).searchParams.has('conversation'),false);
  assert.equal(w.document.querySelector('#player').getAttribute('src'),null);
  assert.equal(w.document.querySelectorAll('#conversationList .history-item').length,1);
  assert.equal(w.document.querySelectorAll('#libBody [data-act="del"]').length,1);
  await w.VideoLensWorkspace.open('c3');await w.loadLibrary();
  busy=true;await w.document.querySelector('#libBody [data-act="del"]').onclick({preventDefault(){}});
  assert.ok(w.document.querySelector('#toast').textContent.includes('正在问答'));
  assert.equal(videos.length,1,'失败时不能删除列表或假报成功');
  busy=false;await w.document.querySelector('#libBody [data-act="del"]').onclick({preventDefault(){}});
  assert.equal(conversations.length,0);assert.equal(videos.length,0);
  assert.equal(w.document.querySelectorAll('#conversationList .history-item').length,0);
  assert.ok(w.document.querySelector('#libBody').textContent.includes('还没有视频'));
  assert.equal(w.document.querySelector('#videoPanel').hidden,true);
  conversations.push({id:'c4',video_id:'v4',title:'跨标签'});videos.push({video_id:'v4',title:'跨标签',status:'ready'});
  await w.VideoLensWorkspace.open('c4');
  const event=erase('v4');
  w.dispatchEvent(new w.StorageEvent('storage',{key:'videolens.deleted',newValue:JSON.stringify(event)}));
  await new Promise(resolve=>setTimeout(resolve,10));
  assert.equal(new URL(w.location.href).searchParams.has('conversation'),false);
  assert.equal(w.document.querySelectorAll('#conversationList .history-item').length,0);
  dom.window.close();console.log('PASS：双向删除刷新、同视频多会话、失败保留与跨标签同步');
})().catch(error=>{console.error(error);dom.window.close();process.exitCode=1});
