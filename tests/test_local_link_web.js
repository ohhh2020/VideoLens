/* 本机选择只预览，发送才登记链接；不上传、分块或读取完整视频。 */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {JSDOM}=require('jsdom');
const read=file=>fs.readFileSync(path.join(__dirname,'..','frontend',file),'utf8');
const dom=new JSDOM(read('workbench/index.html'),{url:'http://127.0.0.1:8000/ui',runScripts:'outside-only'}),w=dom.window;
const selected={path:'D:\\视频\\原片.mp4',filename:'原片.mp4',size_bytes:123456,last_modified:1700000000000,local:true};
const calls=[];let cancelled=true,created=0,state=null;
w.HTMLElement.prototype.scrollIntoView=function(){};w.HTMLMediaElement.prototype.pause=function(){};
w.EventSource=class{close(){}};w.VideoLensHash={file(){throw new Error('禁止对本地视频做上传哈希')}};
w.prompt=()=>selected.path;
w.fetch=async(url,options={})=>{
  calls.push([url,options]);let data={};
  if(url==='/internal/local-videos/select')data=cancelled?{cancelled:true}:selected;
  else if(url==='/internal/local-videos/inspect')data=selected;
  else if(url==='/internal/conversations'&&options.method==='POST'){
    created++;state={conversation:{id:'c-new',title:'新会话',video_id:''},messages:[],operation:null};data=state.conversation;
  }else if(url==='/internal/conversations/c-new/local-video'){
    const payload=JSON.parse(options.body);assert.equal(payload.path,selected.path);assert.equal(payload.size_bytes,selected.size_bytes);data={id:'u-linked',...selected};
  }else if(url==='/internal/conversations/c-new/messages'){
    assert.equal(JSON.parse(options.body).attachment_id,'u-linked');state.operation={id:'o',status:'running',phase:'analysis',message:'开始分析'};data={operation_id:'o',status:'running'};
  }else if(url==='/internal/conversations/c-new')data=state;
  else if(url.startsWith('/internal/conversations?'))data={items:state?[state.conversation]:[]};
  else if(url.startsWith('/internal/videos?'))data=[
    {video_id:'v-local',title:'原片.mp4',conversation_title:'本地对话标题',conversation_id:'c-local',source_kind:'local',local_path:selected.path,status:'ready'},
    {video_id:'v-cloud',title:'旧标题',conversation_title:'云端对话标题',conversation_id:'c-cloud',source_kind:'url',source_url:'https://example.org/video?a=1&b=2',status:'ready'}];
  else if(url==='/internal/settings')data={values:{},effective:{'ui.report_mode':'user'},backends:[]};
  else if(url==='/internal/health')data={status:'ok'};
  else data={usage:{},keyframe_bytes:0};
  return {ok:true,json:async()=>data};
};
for(const file of ['vendor/marked.umd.js','vendor/purify.min.js','shared/markdown.js','workbench/keys.js','workbench/conversations.js'])w.eval(read(file));
w.eval(read('workbench/app.js').replace('\ninitialize();','\n')+'\nObject.assign(window,{initialize,loadLibrary,toast});');
(async()=>{
  await w.initialize();
  await w.document.querySelector('#attachVideo').onclick();
  assert.equal(created,0);assert.equal(w.document.querySelector('#attachmentPreview').hidden,true);
  cancelled=false;await w.document.querySelector('#attachVideo').onclick();
  assert.equal(created,0,'本机选择时不新建会话');
  assert.ok(w.document.querySelector('#attachmentPreview').textContent.includes(selected.path));
  assert.ok(!calls.some(([url])=>url.includes('/local-video')&&!url.includes('/local-videos/')));
  await w.VideoLensWorkspace.selectFile({name:selected.filename,size:selected.size_bytes});
  assert.ok(calls.some(([url])=>url==='/internal/local-videos/inspect'),'拖放通过用户明确填写的原路径登记');
  await w.VideoLensWorkspace.send();
  assert.equal(created,1);
  assert.equal(calls.filter(([url])=>url==='/internal/conversations/c-new/local-video').length,1);
  assert.ok(!calls.some(([url])=>/\/uploads|\/chunks|\/complete/.test(url)));
  const rows=w.document.querySelectorAll('#libBody tr');
  assert.ok(rows[0].textContent.includes('本地对话标题'));
  assert.ok(rows[0].textContent.includes(selected.path));
  assert.equal(rows[0].querySelector('[data-act="cache"]'),null);
  assert.equal(rows[1].querySelector('.video-source a').href,'https://example.org/video?a=1&b=2');
  assert.equal(rows[1].querySelector('[data-act="cache"]').textContent,'文件缓存');
  assert.ok(rows[1].querySelector('a[href="/ui?conversation=c-cloud"]'));
  dom.window.close();console.log('PASS：本机选择/取消/拖放路径、发送才建链接、零上传和视频库标题与来源');
})().catch(error=>{console.error(error);dom.window.close();process.exitCode=1});
