import http from 'node:http';
import path from 'node:path';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import crypto from 'node:crypto';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const HERE=path.dirname(fileURLToPath(import.meta.url));
export const ROOT=path.resolve(process.env.WORKBENCH_ROOT||path.join(HERE,'../..'));
const STATE=path.resolve(process.env.WORKBENCH_STATE||path.join(ROOT,'.experiment_storage/research_workbench'));
fs.mkdirSync(STATE,{recursive:true,mode:0o700});
const keyfile=path.join(STATE,'access-key');
if(!fs.existsSync(keyfile)) fs.writeFileSync(keyfile,crypto.randomBytes(24).toString('base64url'),{mode:0o600});
const KEY=fs.readFileSync(keyfile,'utf8').trim();
const sessions=new Map(), streams=new Map(), jobs=new Map();
const dbfile=path.join(STATE,'sessions.json');
try {for(const s of JSON.parse(fs.readFileSync(dbfile,'utf8'))) {s.running=false;sessions.set(s.id,s);}} catch(e){if(e.code!=='ENOENT') throw e;}
const persist=()=>{const tmp=dbfile+'.tmp';fs.writeFileSync(tmp,JSON.stringify([...sessions.values()]));fs.renameSync(tmp,dbfile);};
const blocked=new Set(['.git','.env','.venv-humanpose','node_modules','.codex','access-key','auth.json']);
const readOnlyRoots=['dataset','orig_dataset','etri_processed','data'];
const allowed=[fs.realpathSync(ROOT)];
for(const d of ['dataset','orig_dataset','etri_processed','data','exp','.experiment_storage']) {try{allowed.push(fs.realpathSync(path.join(ROOT,d)));}catch{}}
const within=(p,r)=>p===r||p.startsWith(r+path.sep);
function error(status,message){throw Object.assign(new Error(message),{status});}
export async function resolveFile(relative='',write=false){
 if(typeof relative!=='string'||relative.includes('\0')||path.isAbsolute(relative)||relative.split(/[\\/]/).some(x=>x==='..'||blocked.has(x))||relative.startsWith('.experiment_storage/research_workbench'))error(403,'Path is not available');
 const logical=path.resolve(ROOT,relative);if(!within(logical,ROOT))error(403,'Outside workspace');
 let real;try{real=await fsp.realpath(logical);}catch(e){if(e.code==='ENOENT')error(404,'File not found');throw e;}
 if(!allowed.some(r=>within(real,r))||within(real,STATE)||real.split(path.sep).some(x=>blocked.has(x)))error(403,'Path is not available');
 if(write&&(readOnlyRoots.includes(relative.split('/')[0])||!within(real,fs.realpathSync(ROOT))))error(403,'External data and dataset files are read-only');
 return real;
}
const mime=(p)=>({'.html':'text/html; charset=utf-8','.js':'application/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.svg':'image/svg+xml','.png':'image/png','.jpg':'image/jpeg','.jpeg':'image/jpeg','.gif':'image/gif','.webp':'image/webp','.mp4':'video/mp4','.mov':'video/quicktime','.webm':'video/webm','.mp3':'audio/mpeg','.wav':'audio/wav','.pdf':'application/pdf','.ico':'image/x-icon'}[path.extname(p).toLowerCase()]||'application/octet-stream');
function json(res,status,value){res.writeHead(status,{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store'});res.end(JSON.stringify(value));}
async function body(req){let data='';for await(const chunk of req){data+=chunk;if(Buffer.byteLength(data)>2*1024*1024)error(413,'Request too large');}try{return JSON.parse(data||'{}');}catch{error(400,'Invalid JSON');}}
const sameKey=(value)=>{const a=Buffer.from(value||''),b=Buffer.from(KEY);return a.length===b.length&&crypto.timingSafeEqual(a,b);};
const authenticated=req=>{const cookies=req.headers.cookie||'';return cookies.split(';').some(c=>{const [name,...v]=c.trim().split('=');return name==='workbench'&&sameKey(v.join('='));});};
function emit(s,event){event={...event,time:new Date().toISOString(),seq:s.events.length};s.events.push(event);persist();for(const res of streams.get(s.id)||[])res.write(`id: ${event.seq}\ndata: ${JSON.stringify(event)}\n\n`);}
function publicSession(s){return {id:s.id,title:s.title,running:s.running,created:s.created,codexThread:s.codexThread||null};}
function getSession(id){const s=sessions.get(id);if(!s)error(404,'Chat not found');return s;}
function stop(s){const p=jobs.get(s.id);if(p){try{process.kill(-p.pid,'SIGTERM');}catch{p.kill('SIGTERM');}}}
function startTurn(s,prompt,context){
 if(s.running)error(409,'This chat is already running. Stop it or open another chat.');
 if(jobs.size>=4)error(429,'Four chats are already running');
 if(typeof prompt!=='string'||!prompt.trim()||prompt.length>100000)error(400,'Enter a message (max 100,000 characters)');
 const first=!s.codexThread;
 const instructions=first?`You are the user's research and coding assistant in ${ROOT}. Read _agents/instruct.md before project work. Keep dataset/, orig_dataset/, etri_processed/, and data/ read-only. Do not interrupt existing GPU training or evaluation jobs unless explicitly requested. For long training/evaluation, inspect progress only every 30 minutes. Respond in the user's language. Execute requested work in this workspace.\n\n`:'';
 const message=instructions+(context?`Selected workspace file: ${context}\n\n`:'')+prompt;
 const args=['exec','--json','--color','never','--sandbox','workspace-write','-c','approval_policy="never"','-C',ROOT];
 const writable=[];try{writable.push(fs.realpathSync(path.join(ROOT,'exp')));}catch{}
 if(writable.length)args.push('-c',`sandbox_workspace_write.writable_roots=${JSON.stringify(writable)}`);
 args.push('-c','sandbox_workspace_write.network_access=true');
 if(s.codexThread) args.push('resume',s.codexThread);
 args.push('-');
 // Reuse this user's existing CLI login/config. Never send credentials to the browser.
 const env={...process.env};delete env.CODEX_THREAD_ID;delete env.CODEX_INTERNAL_ORIGINATOR_OVERRIDE;
 const p=spawn(process.env.CODEX_BIN||'codex',args,{cwd:ROOT,env,detached:true,stdio:['pipe','pipe','pipe']});
 s.running=true;if(s.title==='New chat')s.title=prompt.slice(0,44);jobs.set(s.id,p);
 emit(s,{type:'user',text:prompt,context:context||null});emit(s,{type:'status',text:'Codex is working'});
 let buffer='',stderr='';
 const parse=line=>{try{const event=JSON.parse(line);if(event.type==='thread.started')s.codexThread=event.thread_id;emit(s,{type:'codex',event});}catch{if(line.trim())emit(s,{type:'log',text:line.slice(0,12000)});}};
 p.stdout.on('data',chunk=>{buffer+=chunk.toString();let end;while((end=buffer.indexOf('\n'))>=0){parse(buffer.slice(0,end));buffer=buffer.slice(end+1);}});
 p.stderr.on('data',chunk=>{stderr=(stderr+chunk.toString()).slice(-12000);});
 let finished=false;const finish=(code,reason)=>{if(finished)return;finished=true;if(buffer.trim())parse(buffer);s.running=false;jobs.delete(s.id);emit(s,{type:'done',code,text:reason|| (code===0?'Completed':`Codex exited (${code}). ${stderr.slice(-2500)}`)});};
 p.on('error',e=>finish(-1,e.message));p.on('close',(code,signal)=>finish(code,signal?'Stopped':undefined));
 p.stdin.on('error',()=>{});p.stdin.end(message);persist();
}
async function sendFile(req,res,file,{attachment=false,untrusted=false}={}){
 const st=await fsp.stat(file);if(!st.isFile())error(400,'Not a file');
 let start=0,end=st.size-1,status=200;
 const headers={'Content-Type':mime(file),'Accept-Ranges':'bytes','Cache-Control':'private, no-cache','X-Content-Type-Options':'nosniff'};
 if(attachment)headers['Content-Disposition']=`attachment; filename*=UTF-8''${encodeURIComponent(path.basename(file))}`;
 if(untrusted)headers['Content-Security-Policy']="sandbox; default-src 'none'";
 if(req.headers.range){const m=/^bytes=(\d*)-(\d*)$/.exec(req.headers.range);if(!m||(!m[1]&&!m[2])){res.writeHead(416,{'Content-Range':`bytes */${st.size}`});return res.end();}
  if(!m[1])start=Math.max(0,st.size-Number(m[2]));else start=Number(m[1]);if(m[2]&&m[1])end=Math.min(end,Number(m[2]));
  if(start>end||start>=st.size){res.writeHead(416,{'Content-Range':`bytes */${st.size}`});return res.end();}status=206;headers['Content-Range']=`bytes ${start}-${end}/${st.size}`;
 }
 headers['Content-Length']=Math.max(0,end-start+1);res.writeHead(status,headers);if(req.method==='HEAD'||st.size===0)return res.end();
 const stream=fs.createReadStream(file,{start,end});stream.on('error',()=>res.destroy());res.on('close',()=>stream.destroy());stream.pipe(res);
}
export const server=http.createServer(async(req,res)=>{
 try{
 const u=new URL(req.url,'http://workbench');
 res.setHeader('X-Content-Type-Options','nosniff');res.setHeader('Referrer-Policy','no-referrer');
 if(!['GET','HEAD'].includes(req.method)&&req.headers.origin&&req.headers.origin!==`http://${req.headers.host}`&&req.headers.origin!==`https://${req.headers.host}`)error(403,'Cross-origin request rejected');
 if(u.pathname==='/api/login'&&req.method==='POST'){const b=await body(req);if(!sameKey(b.key))error(401,'Invalid access key');res.setHeader('Set-Cookie',`workbench=${KEY}; HttpOnly; SameSite=Strict; Path=/; Max-Age=2592000`);return json(res,200,{ok:true});}
 if(u.pathname==='/api/logout'&&req.method==='POST'){res.setHeader('Set-Cookie','workbench=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0');return json(res,200,{ok:true});}
 if(u.pathname.startsWith('/api/')){
 if(!authenticated(req))error(401,'Sign in to your workspace');
 if(u.pathname==='/api/info')return json(res,200,{root:ROOT,name:path.basename(ROOT),codex:'Current CLI login',port:server.address().port});
 if(u.pathname==='/api/tree'){
  const relative=u.searchParams.get('path')||'',file=await resolveFile(relative);const entries=await fsp.readdir(file,{withFileTypes:true});
  const result=[];for(const e of entries){if(blocked.has(e.name))continue;const p=relative?relative+'/'+e.name:e.name;if(p.startsWith('.experiment_storage/research_workbench'))continue;
   let directory=e.isDirectory();if(e.isSymbolicLink()){try{directory=(await fsp.stat(path.join(file,e.name))).isDirectory();}catch{continue;}}
   result.push({name:e.name,path:p,directory,symlink:e.isSymbolicLink()});}
  result.sort((a,b)=>Number(b.directory)-Number(a.directory)||a.name.localeCompare(b.name));return json(res,200,{entries:result});
 }
 if(u.pathname==='/api/file'){
  const relative=u.searchParams.get('path')||'',file=await resolveFile(relative,req.method==='PUT'),stat=await fsp.stat(file);
  if(!stat.isFile())error(400,'Not a file');if(stat.size>2*1024*1024)error(413,'Use preview or download for files over 2 MB');
  const data=await fsp.readFile(file);if(data.includes(0))error(415,'Binary file — use preview or download');
  const etag=crypto.createHash('sha256').update(data).digest('hex');
  if(req.method==='PUT'){const b=await body(req);if(b.etag!==etag)error(409,'File changed on disk. Reload before saving.');if(typeof b.content!=='string')error(400,'Invalid content');await fsp.writeFile(file,b.content);return json(res,200,{etag:crypto.createHash('sha256').update(b.content).digest('hex')});}
  return json(res,200,{content:data.toString('utf8'),etag,size:stat.size,modified:stat.mtime,readonly:readOnlyRoots.includes(relative.split('/')[0])||!within(file,fs.realpathSync(ROOT))});
 }
 if(u.pathname==='/api/media')return await sendFile(req,res,await resolveFile(u.searchParams.get('path')||''),{attachment:u.searchParams.has('download'),untrusted:true});
 if(u.pathname==='/api/chats'){
  if(req.method==='POST'){const s={id:crypto.randomUUID(),title:'New chat',created:new Date().toISOString(),running:false,events:[]};sessions.set(s.id,s);persist();return json(res,201,publicSession(s));}
  return json(res,200,[...sessions.values()].map(publicSession));
 }
 const match=/^\/api\/chats\/([\w-]+)(?:\/(events|message|stop))?$/.exec(u.pathname);
 if(match){const s=getSession(match[1]);
  if(match[2]==='events'){
   res.writeHead(200,{'Content-Type':'text/event-stream','Cache-Control':'no-cache','Connection':'keep-alive','X-Accel-Buffering':'no'});
   const after=Number(req.headers['last-event-id']??u.searchParams.get('after')??-1);
   for(const e of s.events)if(e.seq>after)res.write(`id: ${e.seq}\ndata: ${JSON.stringify(e)}\n\n`);
   if(!streams.has(s.id))streams.set(s.id,new Set());streams.get(s.id).add(res);
   const timer=setInterval(()=>res.write(': heartbeat\n\n'),20000);req.on('close',()=>{clearInterval(timer);streams.get(s.id)?.delete(res);});return;
  }
  if(match[2]==='message'&&req.method==='POST'){const b=await body(req);if(b.context)await resolveFile(b.context);startTurn(s,b.prompt,b.context);return json(res,202,{ok:true});}
  if(match[2]==='stop'&&req.method==='POST'){stop(s);return json(res,200,{ok:true});}
  if(req.method==='DELETE'){if(s.running)error(409,'Stop this chat first');sessions.delete(s.id);persist();return json(res,200,{ok:true});}
  if(req.method==='PATCH'){const b=await body(req);s.title=String(b.title||'New chat').slice(0,100);persist();return json(res,200,publicSession(s));}
  return json(res,200,{...publicSession(s),events:s.events});
 }
 error(404,'Endpoint not found');
 }
 const file=u.pathname==='/'?'index.html':u.pathname.slice(1);if(!['index.html','app.js','app.css'].includes(file))error(404,'Not found');
 return await sendFile(req,res,path.join(HERE,'public',file));
 }catch(e){if(res.headersSent)res.end();else json(res,e.status||500,{error:e.status?e.message:'Server error'});if(!e.status)console.error(e);}
});
if(process.env.WORKBENCH_NO_LISTEN!=='1'){
 const port=Number(process.env.PORT||18080),host=process.env.HOST||'0.0.0.0';
 if(port<10000)throw new Error('Choose a port >= 10000');
 server.listen(port,host,()=>console.log(`Research Workbench listening on ${host}:${port}; workspace=${ROOT}; access key stored at ${keyfile}`));
}
