// 실행 중인 대화와 Goal을 유지하며 GPU API만 교체한다.
const endpoints=await(await fetch('http://127.0.0.1:9229/json/list')).json();
const ws=new WebSocket(endpoints[0].webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
const expression=`(async()=>{
 const {server}=await import('file:///home/namhj/EventPose/src/apps/research-workbench/server.mjs');
 const {gpuStats}=await import('file:///home/namhj/EventPose/src/apps/research-workbench/gpu-stats.mjs');
 const fs=await import('node:fs');const crypto=await import('node:crypto');
 const key=fs.readFileSync('/home/namhj/EventPose/.experiment_storage/research_workbench/access-key','utf8').trim();
 const original=server.listeners('request')[0];
 server.removeListener('request',original);
 server.on('request',async(req,res)=>{
  if(new URL(req.url,'http://workbench').pathname!=='/api/gpus')return original(req,res);
  const authorized=(req.headers.cookie||'').split(';').some(c=>{const [name,...v]=c.trim().split('=');const value=v.join('=');return name==='workbench'&&Buffer.byteLength(value)===Buffer.byteLength(key)&&crypto.timingSafeEqual(Buffer.from(value),Buffer.from(key));});
  if(!authorized)return original(req,res);
  try{const gpus=await gpuStats();res.writeHead(200,{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store'});res.end(JSON.stringify({gpus}));}
  catch{res.writeHead(503,{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store'});res.end(JSON.stringify({error:'GPU 상태 조회 실패'}));}
 });
 setTimeout(async()=>{(await import('node:inspector')).close();},500);
 return 'GPU API updated without restarting conversations';
})()`;
const reply=new Promise((resolve,reject)=>{ws.onmessage=e=>{const data=JSON.parse(e.data);if(data.id===1){if(data.result?.exceptionDetails)reject(data.result.exceptionDetails);else resolve(data.result?.result);}};});
ws.send(JSON.stringify({id:1,method:'Runtime.evaluate',params:{expression:'process.getBuiltinModule("vm").runInThisContext('+JSON.stringify(expression)+', {importModuleDynamically:process.getBuiltinModule("vm").constants.USE_MAIN_CONTEXT_DEFAULT_LOADER})',awaitPromise:true,returnByValue:true}}));
console.log(await reply);ws.close();
