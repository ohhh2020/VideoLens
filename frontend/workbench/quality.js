(function(scope){
  'use strict';
  const groups={
    '下载与磁盘':[['media.download.format_max_height','下载短边上限 / px',144,2160],['limits.max_file_bytes','文件上限 / GiB',0.001,8,1073741824],['limits.max_duration_ms','时长上限 / 分钟',1,120,60000],['integrity.disk_reserve_bytes','磁盘安全余量 / GiB',0,1024,1073741824]],
    '固定采样与整帧识别':[['vl.max_image_side','整帧识别长边 / px',224,1920],['integrity.max_sample_gap_s','固定采样间隔 / 秒',1,60],['integrity.evidence_long_side','证据帧长边 / px',640,3840],['integrity.ocr_tokens','整帧识别输出tokens',128,131072]],
    '单次报告预算':[['integrity.report_input_tokens','报告总预算tokens',8192,131072],['integrity.report_output_tokens','报告输出tokens',128,131072],['integrity.max_continuations','最多续写次数',0,2]],
    '问答与检索':[['integrity.qa_local_k','局部检索条数',1,20],['integrity.qa_global_k','比较检索条数',12,20],['integrity.qa_extra_rounds','最多补充检索轮数',0,2],['integrity.qa_local_input_tokens','局部问答总预算',4096,131072],['integrity.qa_local_output_tokens','局部问答输出',128,131072],['integrity.qa_global_input_tokens','全局问答总预算',8192,131072],['integrity.qa_global_output_tokens','全局问答输出',128,131072]],
    '并发与费用预警':[['integrity.cloud_total','云端总在途上限',1,32],['integrity.cloud_text','文本在途上限',1,32],['integrity.cloud_vision','视觉在途上限',1,32],['integrity.cloud_retries','临时错误重试次数',0,3],['integrity.gpu_batch_max','BGE/SigLIP最大批次',1,16],['integrity.asr_batch_max','Whisper最大批次',1,8],['integrity.warning_requests','请求数预警',1,10000],['integrity.input_price_per_million','每百万输入tokens价格',0,10000],['integrity.output_price_per_million','每百万输出tokens价格',0,10000]]
  };
  const definitions=Object.values(groups).flat();
  function load(settings){
    const root=document.querySelector('#qualityFields');if(!root)return;
    root.replaceChildren();const values=settings.quality||{};
    for(const [name,fields] of Object.entries(groups)){
      const section=document.createElement('fieldset'),title=document.createElement('legend');title.textContent=name;section.append(title);section.className='grid';
      for(const [key,label,min,max,scale=1] of fields){
        const box=document.createElement('div'),text=document.createElement('label'),input=document.createElement('input');
        text.textContent=label;input.type='number';input.min=min;input.max=max;input.step=(key.includes('threshold')||key.includes('_fps')||key.includes('price')||scale!==1)?'any':'1';input.dataset.quality=key;
        const value=key.startsWith('integrity.')?values.integrity?.[key.slice(10)]:key==='media.download.format_max_height'?values.source_height:key==='limits.max_file_bytes'?values.max_file_bytes:key==='vl.max_image_side'?values.regular_side:values.max_duration_ms;
        input.value=value==null?'':value/scale;input.id='quality_'+key.replaceAll('.','_');text.htmlFor=input.id;box.append(text,input);section.append(box);
      }root.append(section);
    }
  }
  function append(payload){
    for(const [key,label,min,max,scale=1] of definitions){const field=document.querySelector(`[data-quality="${key}"]`);if(!field||field.value==='')continue;const number=Number(field.value);if(!Number.isFinite(number)||number<min||number>max)throw new Error(label+'超出范围');payload[key]=scale===1?number:Math.round(number*scale)}
  }
  scope.VideoLensQuality={load,append};
})(globalThis);
