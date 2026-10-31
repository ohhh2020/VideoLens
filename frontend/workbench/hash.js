/* 本项目实现的增量 SHA-256；单次只保留一个分块，支持 >4GiB 文件。 */
(function(scope){
  'use strict';
  const K=new Uint32Array([0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2]);
  const r=(x,n)=>(x>>>n)|(x<<(32-n));
  class Sha256{
    constructor(){this.h=new Uint32Array([0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]);this.buffer=new Uint8Array(64);this.used=0;this.bytes=0;this.w=new Uint32Array(64)}
    block(bytes,offset){
      const w=this.w;
      for(let i=0;i<16;i++){const p=offset+i*4;w[i]=(bytes[p]<<24)|(bytes[p+1]<<16)|(bytes[p+2]<<8)|bytes[p+3]}
      for(let i=16;i<64;i++){const a=w[i-15],b=w[i-2];w[i]=(w[i-16]+(r(a,7)^r(a,18)^(a>>>3))+w[i-7]+(r(b,17)^r(b,19)^(b>>>10)))>>>0}
      let [a,b,c,d,e,f,g,h]=this.h;
      for(let i=0;i<64;i++){const t1=(h+(r(e,6)^r(e,11)^r(e,25))+((e&f)^(~e&g))+K[i]+w[i])>>>0,t2=((r(a,2)^r(a,13)^r(a,22))+((a&b)^(a&c)^(b&c)))>>>0;h=g;g=f;f=e;e=(d+t1)>>>0;d=c;c=b;b=a;a=(t1+t2)>>>0}
      [a,b,c,d,e,f,g,h].forEach((x,i)=>{this.h[i]=(this.h[i]+x)>>>0});
    }
    update(input){
      const bytes=input instanceof Uint8Array?input:new Uint8Array(input);this.bytes+=bytes.length;
      let offset=0;
      if(this.used){while(offset<bytes.length&&this.used<64)this.buffer[this.used++]=bytes[offset++];if(this.used===64){this.block(this.buffer,0);this.used=0}}
      while(offset+64<=bytes.length){this.block(bytes,offset);offset+=64}
      while(offset<bytes.length)this.buffer[this.used++]=bytes[offset++];
      return this;
    }
    hex(){
      const tail=new Uint8Array(this.used<56?64:128);tail.set(this.buffer.subarray(0,this.used));tail[this.used]=128;
      const view=new DataView(tail.buffer),bits=this.bytes*8;view.setUint32(tail.length-8,Math.floor(bits/4294967296));view.setUint32(tail.length-4,bits>>>0);
      for(let n=0;n<tail.length;n+=64)this.block(tail,n);
      return Array.from(this.h,x=>x.toString(16).padStart(8,'0')).join('');
    }
  }
  async function file(video,progress=()=>{}){
    if(typeof document!=='undefined'&&typeof Worker!=='undefined')return new Promise((resolve,reject)=>{
      const worker=new Worker('/assets/workbench/hash-worker.js?v=integrity-v2');
      worker.onmessage=e=>{if(e.data.hash){resolve(e.data.hash);worker.terminate()}else if(e.data.error){reject(new Error(e.data.error));worker.terminate()}else progress(e.data.progress)};
      worker.onerror=()=>{worker.terminate();reject(new Error('文件校验线程失败，请重试'))};worker.postMessage(video);
    });
    const hash=new Sha256(),step=8*1024*1024;
    for(let offset=0;offset<video.size;offset+=step){hash.update(await video.slice(offset,offset+step).arrayBuffer());progress(Math.min(1,(offset+step)/video.size));await new Promise(resolve=>setTimeout(resolve,0))}
    return hash.hex();
  }
  scope.VideoLensHash={Sha256,file};
})(globalThis);
