importScripts('/assets/workbench/hash.js?v=integrity-v2');
onmessage=async event=>{try{postMessage({hash:await VideoLensHash.file(event.data,progress=>postMessage({progress}))})}catch(error){postMessage({error:error.message})}};
