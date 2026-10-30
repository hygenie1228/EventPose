// Update a user-authorized objective through the existing owning RPC client.
import fs from 'node:fs';
const [configPath, serverPid, outputPath] = process.argv.slice(2);
const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
const cmd = fs.readFileSync(`/proc/${serverPid}/cmdline`, 'utf8');
if (!cmd.includes('research-workbench/server.mjs')) throw Error('Unexpected server');
process.kill(Number(serverPid), 'SIGUSR1');
await new Promise(r => setTimeout(r, 500));
const targets = await (await fetch('http://127.0.0.1:9229/json/list')).json();
const ws = new WebSocket(targets[0].webSocketDebuggerUrl);
await new Promise((resolve, reject) => {ws.onopen=resolve;ws.onerror=reject;});
let sequence=0; const pending=new Map();
ws.onmessage = event => {const value=JSON.parse(event.data);if(value.id && pending.has(value.id)){
 const {resolve,reject}=pending.get(value.id);pending.delete(value.id);value.error?reject(Error(JSON.stringify(value.error))):resolve(value.result);
}};
const rpc=(method,params={})=>new Promise((resolve,reject)=>{const id=++sequence;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
try {
 const handler=await rpc('Runtime.evaluate',{expression:"process._getActiveHandles().find(h=>h.constructor.name==='Server' && h.listeners('request').length).listeners('request')[0]"});
 const props=await rpc('Runtime.getProperties',{objectId:handler.result.objectId});
 const scopes=props.internalProperties.find(p=>p.name==='[[Scopes]]').value.objectId;
 const scopeList=await rpc('Runtime.getProperties',{objectId:scopes,ownProperties:true});
 let map;
 for(const scope of scopeList.result.filter(p=>/^\d+$/.test(p.name))){
  const values=await rpc('Runtime.getProperties',{objectId:scope.value.objectId,ownProperties:true});
  map=values.result.find(p=>p.name==='goalClients')?.value.objectId;if(map)break;
 }
 if(!map)throw Error('Owning goal client map not found');
 const result=await rpc('Runtime.callFunctionOn',{objectId:map,awaitPromise:true,returnByValue:true,
  functionDeclaration:"async function(c){const client=this.get(c.session_id);if(!client)throw Error('Session client missing');return await client.request('thread/goal/set',{threadId:c.thread_id,objective:c.objective,status:'active',origin:'user'});}",
  arguments:[{value:config}]});
 if(result.exceptionDetails)throw Error(JSON.stringify(result.exceptionDetails));
 fs.writeFileSync(outputPath,JSON.stringify(result.result.value,null,2)+'\n');
 console.log(JSON.stringify({status:result.result.value.goal.status,objective_updated:true}));
} finally {
 await rpc('Runtime.evaluate',{expression:"setTimeout(()=>process.getBuiltinModule('inspector').close(),100)"}).catch(()=>{});
 ws.close();
}
