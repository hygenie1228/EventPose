import {spawn} from 'node:child_process';
import readline from 'node:readline';

// Native Codex app-server goal state; no simulated loop or prompt-only goal mode.
export class CodexGoalClient {
 constructor({root,onEvent,onError}){this.root=root;this.onEvent=onEvent;this.onError=onError;this.pending=new Map();this.next=0;this.turn=null;this.closed=false;}
 async connect(){
  this.proc=spawn(process.env.CODEX_BIN||'codex',['app-server','--stdio'],{cwd:this.root,env:{...process.env,CODEX_THREAD_ID:''},stdio:['pipe','pipe','pipe']});this.stderr='';
  this.proc.stderr.on('data',b=>this.stderr=(this.stderr+b).slice(-4000));
  readline.createInterface({input:this.proc.stdout}).on('line',line=>{let m;try{m=JSON.parse(line);}catch{return;}if(m.id!==undefined&&this.pending.has(m.id)){const p=this.pending.get(m.id);this.pending.delete(m.id);clearTimeout(p.timer);if(m.error)p.reject(new Error(m.error.message));else p.resolve(m.result);return;}
   if(m.id!==undefined){this.proc.stdin.write(JSON.stringify({id:m.id,error:{code:-32601,message:'Interactive approval is unavailable; use configured never-approval workspace policy'}})+'\n');return;}
   if(m.method==='turn/started')this.turn=m.params.turn.id;
   if(m.method==='turn/completed')this.turn=null;
   if(m.method)this.onEvent(m.method,m.params||{});
  });
  this.proc.on('error',e=>this.fail(e));this.proc.on('close',()=>this.fail(new Error('Codex app-server disconnected: '+this.stderr)));
  await this.request('initialize',{clientInfo:{name:'eventpose_workbench',title:'EventPose Workbench',version:'1.0.0'},capabilities:{experimentalApi:true}});
  this.proc.stdin.write(JSON.stringify({method:'initialized',params:{}})+'\n');
 }
 fail(e){for(const p of this.pending.values()){clearTimeout(p.timer);p.reject(e);}this.pending.clear();if(!this.closed)this.onError(e);this.closed=true;}
 request(method,params){return new Promise((resolve,reject)=>{const id=++this.next;const timer=setTimeout(()=>{this.pending.delete(id);reject(new Error(method+' timed out'));},60000);this.pending.set(id,{resolve,reject,timer});this.proc.stdin.write(JSON.stringify({id,method,params})+'\n',e=>{if(e){clearTimeout(timer);this.pending.delete(id);reject(e);}});});}
 async load(threadId,instructions,writeRoots){const params={cwd:this.root,approvalPolicy:'never',sandbox:'workspace-write',developerInstructions:instructions,config:{'sandbox_workspace_write.writable_roots':writeRoots,'sandbox_workspace_write.network_access':true}};const result=await this.request(threadId?'thread/resume':'thread/start',threadId?{...params,threadId}:params);this.thread=result.thread.id;return this.thread;}
 async goal(objective,status='active'){return (await this.request('thread/goal/set',{threadId:this.thread,...(objective?{objective}:{}),status})).goal;}
 async getGoal(){return (await this.request('thread/goal/get',{threadId:this.thread})).goal;}
 async start(text){return this.request('turn/start',{threadId:this.thread,input:[{type:'text',text}],approvalPolicy:'never',cwd:this.root});}
 async steer(text){if(!this.turn)throw new Error('No active turn');return this.request('turn/steer',{threadId:this.thread,expectedTurnId:this.turn,input:[{type:'text',text}]});}
 async pause(){await this.goal(null,'paused');if(this.turn)await this.request('turn/interrupt',{threadId:this.thread,turnId:this.turn});}
 async clear(){if(this.turn)await this.pause();await this.request('thread/goal/clear',{threadId:this.thread});}
 close(){this.closed=true;this.proc?.kill('SIGTERM');}
}
