/* 质量设置的单位换算、预算边界、完整报告安全展示和增量哈希。 */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {JSDOM}=require('jsdom');
const root=path.resolve(__dirname,'..'),read=name=>fs.readFileSync(path.join(root,name),'utf8');
const dom=new JSDOM(read('frontend/workbench/index.html'),{url:'http://127.0.0.1/ui',runScripts:'outside-only'}),w=dom.window;
w.eval(read('frontend/workbench/quality.js'));
w.VideoLensQuality.load({quality:{integrity:{scan_fps:1,cloud_total:8,gpu_batch_max:16},source_height:1080,max_file_bytes:8*1073741824,max_duration_ms:7200000,regular_side:1280}});
assert.equal(w.document.querySelector('[data-quality="limits.max_file_bytes"]').value,'8');
const payload={};w.VideoLensQuality.append(payload);
assert.equal(payload['limits.max_file_bytes'],8589934592);
assert.equal(payload['limits.max_duration_ms'],7200000);
assert.equal(w.document.querySelector('[data-quality="integrity.refine_fps"]'),null);
assert.equal(w.document.querySelector('[data-quality="integrity.summary_batch_target"]'),null);
assert.equal(w.document.querySelector('[data-quality="integrity.ocr_frames"]'),null);
w.document.querySelector('[data-quality="limits.max_file_bytes"]').value='9';
assert.throws(()=>w.VideoLensQuality.append({}),/超出范围/);
w.VideoLensMarkdown={render:value=>value,userReport:value=>'概览与结论'};
w.eval(read('frontend/workbench/conversations.js'));
const report={video_id:'v',summary:'完整报告',profile:{analysis_version:'integrity-v2',source_width:1920,source_height:1080,
  coverage:{ratio:1,covered_clips:33,total_clips:33},usage:{requests:10},summary_tree:[{start_ms:1000,end_ms:2000,level:0,summary:'<script>bad</script>',facts:['数字123'],evidence_ids:['e1']}]}};
const detailed=w.VideoLensWorkspace.reportHtml(report);
assert.ok(detailed.includes('33/33'));assert.ok(detailed.includes('1920 × 1080'));assert.ok(detailed.includes('原始语音 / 画面 / OCR证据'));
assert.ok(!detailed.includes('<script>bad</script>'));
const fastReport={...report,profile:{...report.profile,analysis_version:'fast-v3',coverage:{representative:true,covered_clips:32,total_clips:36},summary_tree:[]}};
const fastHtml=w.VideoLensWorkspace.reportHtml(fastReport);
assert.ok(fastHtml.includes('单次报告取样 32/36'));
assert.ok(fastHtml.includes('未逐条总结全片'));
assert.ok(!fastHtml.includes('全部章节与关键事实'));
w.document.body.classList.add('user-mode');
const simple=w.VideoLensWorkspace.reportHtml(report);assert.ok(!simple.includes('33/33'));assert.ok(simple.includes('概览与结论'));
report.profile.report_truncated=true;assert.ok(w.VideoLensWorkspace.reportHtml(report).includes('报告尚未完成'));
require('../frontend/workbench/hash.js');
for(const size of [0,1,55,56,63,64,65,1000000]){
  const bytes=crypto.randomBytes(size),hash=new VideoLensHash.Sha256();
  for(let i=0;i<size;i+=113)hash.update(bytes.subarray(i,i+113));
  assert.equal(hash.hex(),crypto.createHash('sha256').update(bytes).digest('hex'));
}
dom.window.close();console.log('PASS：8GiB换算、质量设置范围、详细/用户报告、截断可见与增量SHA256边界');
