import fs from 'node:fs/promises';
import {execFile} from 'node:child_process';

const command=(args)=>new Promise((resolve,reject)=>execFile('nvidia-smi',args,{timeout:4000},(error,out)=>error?reject(error):resolve(out)));
const query=['--query-gpu=index,uuid,utilization.gpu,memory.total,memory.used','--format=csv,noheader,nounits'];
function parse(raw){
 return raw.trim().split('\n').map(line=>{
  const [index,uuid,util,total,used]=line.split(',').map(v=>v.trim());
  if(!/^\d+$/.test(index)||!uuid?.startsWith('GPU-')||!Number.isFinite(Number(total))||!Number.isFinite(Number(used)))throw new Error('GPU 응답 오류');
  return {index:Number(index),uuid,utilization:Number(util)||0,totalMiB:Number(total),usedMiB:Number(used),users:[]};
 });
}
let cache={time:0,value:[]},pending;
export async function gpuStats(){
 if(Date.now()-cache.time<5000)return cache.value;
 if(pending)return pending;
 pending=(async()=>{
  let values;
  try{values=parse(await command(query));}
  catch{
   const root='/proc/driver/nvidia/gpus';
   const devices=await fs.readdir(root);
   values=await Promise.all(devices.map(async bus=>{
    const info=await fs.readFile(`${root}/${bus}/information`,'utf8');
    const index=Number(info.match(/Device Minor:\s*(\d+)/)?.[1]);
    const uuid=info.match(/GPU UUID:\s*(\S+)/)?.[1];
    try{return parse(await command(['-i',bus,...query]))[0];}
    catch{return {index,uuid,error:'GPU 드라이버 조회 실패',users:[]};}
   }));
  }
  // Query processes per device too: a broken device must not hide healthy users.
  await Promise.all(values.filter(g=>!g.error).map(async g=>{
   try{
    const raw=await command(['-i',g.uuid,'--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits']);
    const totals=new Map();
    for(const line of raw.trim().split('\n')){
     const [pid,memory]=line.split(',').map(v=>v.trim());if(!/^\d+$/.test(pid))continue;
     let name='?';try{name=(await fs.readFile(`/proc/${pid}/status`,'utf8')).match(/^Uid:\s*(\d+)/m)?.[1]||'?';}catch{}
     totals.set(name,(totals.get(name)||0)+(Number(memory)||0));
    }
    const passwd=await fs.readFile('/etc/passwd','utf8');const names=new Map(passwd.trim().split('\n').map(l=>{const parts=l.split(':');return [parts[2],parts[0]];}));
    g.users=[...totals].map(([uid,memoryMiB])=>({name:names.get(uid)||uid,memoryMiB}));
   }catch{}
  }));
  values.sort((a,b)=>a.index-b.index);cache={time:Date.now(),value:values};return values;
 })().finally(()=>pending=null);
 return pending;
}
