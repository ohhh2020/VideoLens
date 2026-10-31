/* 真实主题模块：系统切换、持久化、跨标签同步、快速切换顺序保存。 */
"use strict";
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {JSDOM}=require('jsdom');
const root=path.resolve(__dirname,'..');
const dom=new JSDOM(fs.readFileSync(path.join(root,'frontend/workbench/index.html'),'utf8'),{url:'http://127.0.0.1/ui',runScripts:'outside-only'});
const w=dom.window,calls=[],fluent=[];
let systemHandler;
const media={matches:false,addEventListener:(name,cb)=>systemHandler=cb};
w.matchMedia=()=>media;
w.VideoLensFluent={applyTheme:value=>fluent.push(value)};
w.fetch=async(url,options)=>{calls.push(JSON.parse(options.body));return {ok:true}};
w.eval(fs.readFileSync(path.join(root,'frontend/workbench/theme.js'),'utf8'));
async function main(){
  assert.equal(w.document.documentElement.dataset.theme,'light');
  await w.VideoLensTheme.save('dark');
  assert.equal(w.document.documentElement.dataset.theme,'dark');
  assert.equal(w.document.querySelector('#themeQuick').value,'dark');
  assert.equal(w.document.querySelector('#theme_mode').value,'dark');
  assert.equal(w.localStorage.getItem('videolens.theme'),'dark');
  assert.deepEqual(calls[0],{'ui.theme_mode':'dark'},'主题保存不携带密钥/模型设备');
  await w.VideoLensTheme.save('system');
  media.matches=true;systemHandler();
  assert.equal(w.document.documentElement.dataset.theme,'dark');
  await w.VideoLensTheme.save('light');
  systemHandler();assert.equal(w.document.documentElement.dataset.theme,'light');
  w.dispatchEvent(new w.StorageEvent('storage',{key:'videolens.theme',newValue:'dark'}));
  assert.equal(w.document.documentElement.dataset.theme,'dark');
  const a=w.VideoLensTheme.save('light'),b=w.VideoLensTheme.save('dark');await Promise.all([a,b]);
  assert.deepEqual(calls.slice(-2),[{'ui.theme_mode':'light'},{'ui.theme_mode':'dark'}]);
  assert.equal(fluent.at(-1),'dark');
  dom.window.close();console.log('PASS：三种主题、即时Fluent同步、持久化、系统跟随、跨标签与快速切换顺序');
}
main().catch(error=>{console.error(error);dom.window.close();process.exitCode=1});
