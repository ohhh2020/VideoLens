"use strict";

/* 两个网页共用本地 Markdown 库；模型正文先渲染、再清理 HTML。 */
(function(scope){
  const canonical=text=>String(text).replace(/[*`]/g,"").replace(/^[\s\d一二三四五六七八九十、.．)）]+/,"").trim();
  function userReport(markdown){
    const source=String(markdown||"").replace(/^\s*\*\*(内容概览|关键结论|主要脉络)\*\*\s*[:：]?\s*$/gm,"## $1");
    const sections=new Map([["内容概览",[]],["关键结论",[]]]);
    let selected=null,depth=0;
    const preamble=[];
    let anyHeading=false;
    for(const token of scope.marked.lexer(source)){
      if(token.type==="heading"){
        anyHeading=true;
        if(canonical(token.text)==="主要脉络"){selected=null;continue}
        if(selected&&token.depth<=depth)selected=null;
        const label=canonical(token.text);
        if(sections.has(label)){selected=label;depth=token.depth;continue}
      }
      if(selected)sections.get(selected).push(token.raw);
      else if(!anyHeading)preamble.push(token.raw);
    }
    if(!sections.get("内容概览").join("").trim())sections.set("内容概览",preamble);
    return [...sections].map(([title,parts])=>`## ${title}\n\n${parts.join("").trim()||"该报告暂未提供此部分内容。"}`).join("\n\n");
  }

  function render(markdown,{refs=[],formatTime=ms=>String(Math.floor(ms/1000))+"s"}={}){
    const html=scope.marked.parse(String(markdown||""),{gfm:true,breaks:true});
    const clean=scope.DOMPurify.sanitize(html,{USE_PROFILES:{html:true},SANITIZE_NAMED_PROPS:true,FORBID_TAGS:["style","form","input"],FORBID_ATTR:["style"]});
    const box=scope.document.createElement("div");box.innerHTML=clean;
    box.querySelectorAll("a").forEach(a=>{a.target="_blank";a.rel="noopener noreferrer"});
    const byIndex=new Map(refs.map(ref=>[Number(ref.index),ref]));
    if(byIndex.size){
      const walker=scope.document.createTreeWalker(box,scope.NodeFilter.SHOW_TEXT),nodes=[];
      while(walker.nextNode()){
        const node=walker.currentNode;
        if(!node.parentElement.closest("code,pre,a")&&/\[\d{1,6}\]/.test(node.textContent))nodes.push(node);
      }
      for(const node of nodes){
        const fragment=scope.document.createDocumentFragment();let from=0;
        for(const match of node.textContent.matchAll(/\[(\d{1,6})\]/g)){
          const ref=byIndex.get(Number(match[1]));if(!ref)continue;
          fragment.append(node.textContent.slice(from,match.index));
          const cite=scope.document.createElement("button");cite.type="button";cite.className="cite";cite.dataset.n=String(ref.index);
          cite.textContent=`[${ref.index}] ${formatTime(ref.start_ms)}`;
          fragment.append(cite);from=match.index+match[0].length;
        }
        fragment.append(node.textContent.slice(from));node.replaceWith(fragment);
      }
    }
    return box.innerHTML;
  }
  scope.VideoLensMarkdown={render,userReport};
})(globalThis);
