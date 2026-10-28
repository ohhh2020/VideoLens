/* 用 Node 运行：node tests/test_chat_stream.js；不需要 npm 安装依赖。 */
"use strict";
const assert=require("node:assert/strict");
const {readSSE,createTypewriter}=require("../frontend/chat/stream.js");

async function main(){
  const bytes=new TextEncoder().encode('data: {"type":"delta","data":"中文😀"}\r\n\r\n: heartbeat\n\ndata: {"type":"final","result":{"answer":"结束"}}');
  const body=new ReadableStream({start(c){for(let i=0;i<bytes.length;i+=2)c.enqueue(bytes.slice(i,i+2));c.close()}});
  const events=[];
  for await(const e of readSSE(body))events.push(e);
  assert.equal(events.length,2);
  assert.equal(events[0].data,"中文😀");
  assert.equal(events[1].result.answer,"结束");

  const element={textContent:""};
  const writer=createTypewriter(element);
  writer.push("逐字显示😀");
  assert.equal(element.textContent,"逐"); // final 尚未到达，第一字已显示
  assert.notEqual(element.textContent,writer.text);
  await writer.drain();
  assert.equal(element.textContent,"逐字显示😀");
  writer.stop();

  const rich={innerHTML:""};
  const rendered=createTypewriter(rich,()=>{},text=>`<strong>${text}</strong>`);
  rendered.push("格式");
  assert.equal(rich.innerHTML,"<strong>格</strong>");
  await rendered.drain();
  assert.equal(rich.innerHTML,"<strong>格式</strong>");
  rendered.stop();

  const cancelled=createTypewriter(element);
  cancelled.push("取消时仍有尚未显示的内容".repeat(20));
  const draining=cancelled.drain();
  cancelled.stop();await draining;
  console.log("PASS：分片中文/表情、CRLF、心跳、EOF flush、逐字增量与取消收尾");
}
main().catch(e=>{console.error(e);process.exitCode=1});
