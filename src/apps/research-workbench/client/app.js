import {EditorView,basicSetup} from 'codemirror';
import {EditorState} from '@codemirror/state';
import {keymap} from '@codemirror/view';
import {python} from '@codemirror/lang-python';
import {javascript} from '@codemirror/lang-javascript';
import {json} from '@codemirror/lang-json';
import {markdown} from '@codemirror/lang-markdown';
import {oneDark} from '@codemirror/theme-one-dark';
import {marked} from 'marked';
import DOMPurify from 'dompurify';

const chatOnly=new URLSearchParams(location.search).has('chatOnly');
const pane=chatOnly?'agents':'chat';
let singlePaneMode=localStorage.getItem('workbench.singlePane')||'chat';
const themeKey='workbench.theme';
const chatStorage='workbench.lastChat.'+(chatOnly?'secondary':'primary');
if(chatOnly)document.body.classList.add('chat-only');
const loadingHTML='<div class="loading-state"><span class="spinner"></span> 불러오는 중…</div>';
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const state={files:new Map(),activeFile:null,chats:[],activeChat:null,events:[],stream:null,editor:null,context:null,busy:false,goal:null,sending:false};
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const basename=p=>p.split('/').pop();
const extension=p=>(p.split('.').pop()||'').toLowerCase();
const mediaURL=p=>'/api/media?path='+encodeURIComponent(p);
const icon=p=>({py:'Py',md:'≡',json:'{}',toml:'⚙',yaml:'⚙',yml:'⚙',mp4:'▷',webm:'▷',png:'▧',jpg:'▧',jpeg:'▧',csv:'▦',js:'JS',mjs:'JS',html:'◇',css:'#',sh:'$'}[extension(p)]||'·');
async function api(url,options={}){const r=await fetch(url,{...options,headers:{'Content-Type':'application/json','X-Workbench-Pane':pane,...options.headers}});let data;try{data=await r.json();}catch{throw new Error('Server connection failed');}if(!r.ok){if(r.status===401)showLogin();throw new Error(data.error||`Request failed (${r.status})`);}return data;}
function toast(text){$('#toast').textContent=text;$('#toast').hidden=false;clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('#toast').hidden=true,4500);}
function showLogin(){$('#login').hidden=false;$('#shell').hidden=true;}
function markdownHTML(text){const template=document.createElement('template');template.innerHTML=marked.parse(text||'',{breaks:true});for(const a of template.content.querySelectorAll('a[href^="file:"]')){try{const url=new URL(a.getAttribute('href')),root=state.root||'/home/namhj/EventPose';if(decodeURIComponent(url.pathname).startsWith(root+'/'))a.setAttribute('href',decodeURIComponent(url.pathname));}catch{}}return DOMPurify.sanitize(template.innerHTML,{ADD_TAGS:['video','audio','source'],ADD_ATTR:['controls','src','type']});}
function resolveLink(href,from){
 if(!href||href.startsWith('#'))return null;let p=href;
 if(/^https?:/i.test(p)){let url;try{url=new URL(p);}catch{return null;}if(url.origin!==location.origin)return null;if(['/api/media','/api/file'].includes(url.pathname))p=url.searchParams.get('path')||'';else p=url.pathname;}
 else if(/^file:/i.test(p)){try{p=new URL(p).pathname;}catch{return null;}}
 else if(/^[a-z][a-z0-9+.-]*:/i.test(p))return null;
 p=p.split('#')[0].split('?')[0];try{p=decodeURIComponent(p);}catch{}
 const root=state.root||'/home/namhj/EventPose';if(p.startsWith(root+'/'))return p.slice(root.length+1).replace(/:\d+(?::\d+)?$/,'');
 const rooted=p.startsWith('/')||/^(?:_agents|src|scripts|tools|exp|dataset|data)\//.test(p)||p==='README.md';
 const parts=rooted?[]:(from||'').split('/').slice(0,-1);for(const part of p.split('/')){if(part==='..')parts.pop();else if(part&&part!=='.')parts.push(part);}return parts.join('/').replace(/:\d+(?::\d+)?$/,'');
}
function wireMarkdown(el,source){el.querySelectorAll('img,video,audio,source').forEach(node=>{const p=resolveLink(node.getAttribute('src'),source);if(p)node.src=mediaURL(p);});el.querySelectorAll('a').forEach(a=>{const p=resolveLink(a.getAttribute('href'),source);if(p){a.href='#';a.onclick=e=>{e.preventDefault();openFile(p);};}else{a.target='_blank';a.rel='noopener noreferrer';}});}
async function loadTree(container,path='',depth=0){
 container.innerHTML=loadingHTML;
 try{const data=await api('/api/tree?path='+encodeURIComponent(path));container.textContent='';for(const entry of data.entries){const wrap=document.createElement('div');wrap.className='tree-node';wrap.dataset.name=entry.name.toLowerCase();const row=document.createElement('button');row.className='tree-row'+(entry.directory?' directory':'');row.style.paddingLeft=(12+depth*14)+'px';row.dataset.path=entry.path;row.setAttribute('role','treeitem');row.title=entry.path;row.oncontextmenu=e=>{e.preventDefault();showMenu(e,[{label:'이름 변경',run:()=>operateFile(entry,'rename')},{label:'삭제',danger:true,run:()=>operateFile(entry,'delete')}]);};
 row.innerHTML=`<span class="chevron">${entry.directory?'›':''}</span><span class="file-icon">${entry.directory?'▱':esc(icon(entry.path))}</span><span class="name">${esc(entry.name)}</span>${entry.symlink?'<span class="symlink">↗</span>':''}`;
 wrap.append(row);container.append(wrap);
 if(entry.directory){row.onclick=async()=>{let children=wrap.querySelector(':scope > .tree-children');if(children){children.hidden=!children.hidden;row.querySelector('.chevron').textContent=children.hidden?'›':'⌄';return;}children=document.createElement('div');children.className='tree-children';wrap.append(children);row.querySelector('.chevron').textContent='⌄';await loadTree(children,entry.path,depth+1);};}
 else row.onclick=()=>openFile(entry.path);
 }}catch(e){container.textContent=e.message;}
 highlightFile();
}
function highlightFile(){$$('.tree-row').forEach(row=>row.classList.toggle('active',row.dataset.path===state.activeFile));}
function tabs(){const el=$('#file-tabs');el.textContent='';for(const [p,file]of state.files){const button=document.createElement('button');button.className='file-tab'+(p===state.activeFile?' active':'');button.innerHTML=`<span>${esc(icon(p))}</span><span>${esc(basename(p))}</span>${file.dirty?'<span class="dirty-dot">●</span>':''}<span class="close" role="button" aria-label="Close file" title="Close file">×</span>`;button.onclick=e=>{if(e.target.closest('.close')){if(file.dirty&&!confirm('저장하지 않은 변경을 닫을까요?'))return;state.files.delete(p);if(p===state.activeFile){state.activeFile=[...state.files.keys()].pop()||null;if(state.activeFile)renderFile();else{state.editor?.destroy();state.editor=null;$('#file-content').innerHTML='<div class="unsupported">왼쪽에서 파일을 선택하세요.</div>';$('#breadcrumb').textContent='EventPose';for(const id of ['#save-file','#download','#preview-toggle','#attach-file','#reload-file'])$(id).hidden=true;}}tabs();return;}state.activeFile=p;renderFile();tabs();};el.append(button);}highlightFile();}
async function openFile(p){if(chatOnly){window.parent.postMessage({type:'open-workspace-file',path:p},location.origin);return;}if(window.matchMedia('(max-width:767px)').matches)document.body.classList.add('mobile-viewer-open');if(state.files.has(p)){state.activeFile=p;renderFile();tabs();return;}$('#file-content').innerHTML=loadingHTML;try{const ext=extension(p),media=['png','jpg','jpeg','gif','webp','svg','mp4','webm','mov','mp3','wav','pdf'].includes(ext);let data={media,content:'',preview:ext==='md',dirty:false};if(!media){try{data={...data,...await api('/api/file?path='+encodeURIComponent(p))};}catch(e){if(/Binary|2 MB/.test(e.message))data.unsupported=e.message;else throw e;}}state.files.set(p,data);state.activeFile=p;renderFile();tabs();localStorage.setItem('workbench.lastFile',p);}catch(e){toast(e.message);}}
function renderFile(){
 const p=state.activeFile,file=state.files.get(p);if(!file)return;state.editor?.destroy();state.editor=null;const el=$('#file-content');el.innerHTML='';$('#breadcrumb').textContent=p.split('/').join('  /  ');$('#reload-file').hidden=false;$('#download').hidden=false;$('#download').href=mediaURL(p)+'&download=1';$('#download').title='Download file';$('#attach-file').hidden=false;$('#save-file').hidden=file.media||file.unsupported||file.readonly;$('#save-file').disabled=!file.dirty;$('#preview-toggle').hidden=extension(p)!=='md';$('#preview-toggle').textContent=file.preview?'〈〉 소스 보기':'▤ 미리보기';$('#file-status').textContent=file.media?`${extension(p).toUpperCase()} · Preview`:file.unsupported||`${file.readonly?'Read-only · ':''}${(file.size/1024).toFixed(1)} KB · UTF-8`;
 if(file.unsupported){el.innerHTML=`<div class="unsupported"><h2>${esc(basename(p))}</h2><p>${esc(file.unsupported)}</p><a href="${mediaURL(p)}&download=1">파일 다운로드 ↓</a></div>`;return;}
 if(file.media){const stage=document.createElement('div');stage.className='media-stage';const ext=extension(p);let node;if(['mp4','webm','mov'].includes(ext)){node=document.createElement('video');node.controls=true;node.preload='metadata';}else if(['mp3','wav'].includes(ext)){node=document.createElement('audio');node.controls=true;}else if(ext==='pdf'){node=document.createElement('iframe');node.className='pdf-preview';}else{node=document.createElement('img');node.alt=basename(p);}node.src=mediaURL(p);stage.append(node);const label=document.createElement('div');label.className='media-label';label.textContent=p;stage.append(label);el.append(stage);return;}
 if(file.preview){const article=document.createElement('article');article.className='document markdown';article.innerHTML=markdownHTML(file.content);wireMarkdown(article,p);el.append(article);return;}
 const language={py:python,js:javascript,mjs:javascript,json,md:markdown}[extension(p)];const extensions=[basicSetup,(document.documentElement.dataset.theme==='dark'?oneDark:[]),EditorState.readOnly.of(!!file.readonly),keymap.of([{key:'Mod-s',run:()=>{saveFile();return true;}}]),EditorView.updateListener.of(update=>{if(update.docChanged){file.content=update.state.doc.toString();file.dirty=true;$('#save-file').disabled=false;tabs();}})];if(language)extensions.push(language());state.editor=new EditorView({doc:file.content,extensions,parent:el});
}
async function saveFile(){const p=state.activeFile,file=state.files.get(p);if(!file?.dirty)return;try{const r=await api('/api/file?path='+encodeURIComponent(p),{method:'PUT',body:JSON.stringify({content:file.content,etag:file.etag})});file.etag=r.etag;file.dirty=false;tabs();$('#save-file').disabled=true;toast('저장했습니다');}catch(e){toast(e.message);}}
function setContext(p){state.context=p;const chip=$('#context-chip');chip.hidden=!p;if(p){chip.innerHTML=`<span>▤ ${esc(p)}</span><button title="Remove context">×</button>`;chip.querySelector('button').onclick=()=>setContext(null);}}
async function refreshChats(){state.chats=await api('/api/chats?pane='+pane);renderChatTabs();}
function renderChatTabs(){const el=$('#chat-tabs');el.textContent='';for(const s of state.chats){const b=document.createElement('button');b.className='chat-tab'+(s.id===state.activeChat?' active':'');b.title=s.title;b.innerHTML=`${s.running?'◌':'◦'} <span>${esc(s.title.length>20?s.title.slice(0,20)+'…':s.title)}</span><span class="chat-close" role="button" aria-label="Close chat" title="채팅 닫기">×</span>`;b.onclick=e=>{if(e.target.closest('.chat-close'))closeChat(s.id).catch(e=>toast(e.message));else selectChat(s.id);};b.oncontextmenu=e=>{e.preventDefault();showMenu(e,[{label:'이름 변경',run:async()=>{const title=prompt('채팅 이름',s.title);if(title){await api('/api/chats/'+s.id,{method:'PATCH',body:JSON.stringify({title})});await refreshChats();}}},{label:'닫기',run:()=>closeChat(s.id)}]);};b.ondblclick=async()=>{const title=prompt('채팅 이름',s.title);if(title){await api('/api/chats/'+s.id,{method:'PATCH',body:JSON.stringify({title})});await refreshChats();}};el.append(b);}el.querySelector('.active')?.scrollIntoView({block:'nearest',inline:'nearest'});}
function setBusy(busy){state.busy=busy;$('#send-chat').hidden=busy&&!state.goal;$('#send-chat').title=busy&&state.goal?'현재 Goal에 추가 메시지 보내기':'메시지 보내기';$('#agent-state')?.classList.toggle('working-animation',busy);$('#stop-chat').hidden=!busy;if($('#agent-state')){$('#agent-state').textContent=busy?'Working':'Ready';$('#agent-state').classList.toggle('busy',busy);}const s=state.chats.find(s=>s.id===state.activeChat);if(s)s.running=busy;renderChatTabs();}
async function selectChat(id){$('#messages').innerHTML=loadingHTML;state.stream?.close();state.activeChat=id;localStorage.setItem(chatStorage,id);let data;try{data=await api('/api/chats/'+id);}catch(e){if(e.message==='Chat not found'){await refreshChats();if(state.chats.length)return selectChat(state.chats[0].id);return newChat();}throw e;}state.events=data.events;$('#prompt').value=localStorage.getItem('workbench.draft.'+pane+'.'+id)||'';renderGoal(data.goal);setBusy(data.running);renderMessages();renderChatTabs();const last=state.events.at(-1)?.seq??-1;const stream=new EventSource(`/api/chats/${id}/events?after=${last}&pane=${pane}`);state.stream=stream;
 stream.onmessage=e=>{if(state.activeChat!==id)return;const event=JSON.parse(e.data);if(state.events.some(v=>v.seq===event.seq))return;state.events.push(event);if(event.type==='goal')renderGoal(event.goal);if(event.type==='done'||(event.type==='codex'&&['turn.completed','turn.failed','error'].includes(event.event.type))){setBusy(false);refreshChats();}else if(event.type==='status')setBusy(true);renderMessages();};stream.onerror=()=>{};}
async function newChat(){const s=await api('/api/chats',{method:'POST',body:JSON.stringify({pane})});await refreshChats();await selectChat(s.id);$('#prompt').focus();}
function renderMessages(){const el=$('#messages');const nearBottom=el.scrollHeight-el.scrollTop-el.clientHeight<120;const blocks=[];let any=false;const itemPositions=new Map();for(const e of state.events){if(e.type==='user'){itemPositions.clear();blocks.push({type:'user',text:e.text,context:e.context});any=true;}else if(e.type==='codex'){
 const c=e.event;if(c.type==='item.completed'||c.type==='item.updated'||c.type==='item.started'){const item=c.item||{},key=item.id||e.seq;const type=['agent_message','assistant_message'].includes(item.type)?'assistant':item.type==='reasoning'?'reasoning':'tool';const block={type,item,text:item.text||item.content||'',status:c.type};if(itemPositions.has(key))blocks[itemPositions.get(key)]=block;else{itemPositions.set(key,blocks.length);blocks.push(block);}any=true;}
 if(c.type==='error'||c.type==='turn.failed')blocks.push({type:'error',text:c.message||c.error?.message||'Codex request failed'});
 }else if(e.type==='done'&&e.code!==0)blocks.push({type:'error',text:e.text});}
 if(!any){el.innerHTML='';if(!chatOnly)showRecentReports(el);return;}
 el.innerHTML='';for(const b of blocks){if(b.type==='user'||b.type==='assistant'){const div=document.createElement('div');div.className='message '+b.type;div.innerHTML=`<div class="message-label">${b.type==='user'?'◉ YOU':'✳ CODEX'}</div><div class="body ${b.type==='assistant'?'markdown':''}"></div>`;const body=div.querySelector('.body');if(b.type==='user'){body.textContent=b.text;if(b.context){const small=document.createElement('small');small.className='muted';small.textContent='\n↳ '+b.context;body.append(small);}}else{body.innerHTML=markdownHTML(String(b.text));wireMarkdown(body,state.context||'');}el.append(div);}else if(b.type==='tool'||b.type==='reasoning'){const item=b.item,details=document.createElement('details');details.className='tool-event';const title=b.type==='reasoning'?'Thinking':item.command||item.type||'Tool';details.innerHTML=`<summary>${b.status==='item.completed'?'✓':'◌'} ${esc(title)}</summary><pre>${esc(String(item.aggregated_output||b.text||JSON.stringify(item,null,2)).slice(-20000))}</pre>`;el.append(details);}else{const div=document.createElement('div');div.className='event-status error';div.textContent=b.text;el.append(div);}}
 if(state.busy){const p=document.createElement('div');p.className='event-status';p.innerHTML='<span class="spinner"></span><span>Codex가 작업 중입니다<span class="typing-dots"><i></i><i></i><i></i></span></span>';el.append(p);}if(nearBottom||state.events.length<5)el.scrollTop=el.scrollHeight;}
async function send(){const input=$('#prompt');let text=input.value.trim();const slash=/^\/goal(?:\s|$)/.test(text);if(slash){text=text.replace(/^\/goal\s*/, '');if(!text){toast('Goal 체크 후 목표를 입력하거나 /goal 목표내용을 입력하세요');return;}}const useGoal=slash||$('#goal-enabled').checked;if(!text||state.sending||(state.busy&&!state.goal))return;if(state.busy&&useGoal){toast('진행 중인 Goal에는 체크를 끄고 추가 메시지를 보내 주세요');return;}state.sending=true;$('#send-chat').disabled=true;try{if(!state.activeChat)await newChat();const sendRequest=()=>api('/api/chats/'+state.activeChat+'/message',{method:'POST',body:JSON.stringify({prompt:text,context:state.context,goal:useGoal})});try{await sendRequest();}catch(e){if(e.message!=='Chat not found')throw e;await newChat();await sendRequest();}input.value='';localStorage.removeItem('workbench.draft.'+pane+'.'+state.activeChat);$('#goal-enabled').checked=false;setBusy(true);refreshChats();}catch(e){toast(e.message);}finally{state.sending=false;$('#send-chat').disabled=false;}}
async function init(){try{const info=await api('/api/info');state.root=info.root;$('#login').hidden=true;$('#shell').hidden=false;$('#project-name').textContent=info.name.toUpperCase();await Promise.all([loadTree($('#tree')),refreshChats()]);const last=localStorage.getItem(chatStorage);const available=state.chats;const requested=chatOnly?new URLSearchParams(location.search).get('agent'):null;if(available.length)await selectChat(available.some(s=>s.id===requested)?requested:available.some(s=>s.id===last)?last:available[0].id);else await newChat();if(!chatOnly)setLayout(localStorage.getItem('workbench.layout')||'2');const file=localStorage.getItem('workbench.lastFile');if(file&&!chatOnly)openFile(file);}catch(e){if(!$('#login').hidden)return;toast(e.message);}}
$('#login-form').onsubmit=async e=>{e.preventDefault();try{await api('/api/login',{method:'POST',body:JSON.stringify({password:$('#access-key').value})});$('#access-key').value='';init();}catch(e){$('#login-error').textContent=e.message;}};
$('#logout').onclick=async()=>{state.stream?.close();await api('/api/logout',{method:'POST',body:'{}'});showLogin();};
$('#new-chat').onclick=()=>{if(!chatOnly&&document.body.classList.contains('one-chat')&&singlePaneMode==='agents'){$('#secondary-chat')?.contentWindow.document.querySelector('#new-chat')?.click();return;}newChat().catch(e=>toast(e.message));};$('#send-chat').onclick=send;$('#stop-chat').onclick=()=>api('/api/chats/'+state.activeChat+'/stop',{method:'POST',body:'{}'}).catch(e=>toast(e.message));$('#prompt').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();send();}};$('#save-file').onclick=saveFile;$('#attach-file').onclick=()=>{if(!chatOnly&&document.body.classList.contains('one-chat')&&singlePaneMode==='agents'){$('#secondary-chat').contentWindow.postMessage({type:'file-context',path:state.activeFile},location.origin);return;}setContext(state.activeFile);$('#prompt').focus();};$('#preview-toggle').onclick=()=>{const file=state.files.get(state.activeFile);file.preview=!file.preview;renderFile();};$('#refresh-tree').onclick=()=>loadTree($('#tree'));$$('[data-open]').forEach(b=>b.onclick=()=>openFile(b.dataset.open));
window.addEventListener('keydown',e=>{if((e.metaKey||e.ctrlKey)&&e.key==='s'){e.preventDefault();saveFile();}if((e.metaKey||e.ctrlKey)&&e.key==='k'){e.preventDefault();$('#tree').focus();}});
window.addEventListener('beforeunload',e=>{if([...state.files.values()].some(f=>f.dirty)){e.preventDefault();e.returnValue='';}});
$$('.resize-handle').forEach(handle=>{handle.onpointerdown=e=>{handle.setPointerCapture(e.pointerId);const start=e.clientX,left=$('#explorer').getBoundingClientRect().width,right=$('#ai-panel').getBoundingClientRect().width;handle.onpointermove=ev=>{const delta=ev.clientX-start;if(handle.dataset.side==='left')document.documentElement.style.setProperty('--left',Math.max(150,Math.min(480,left+delta))+'px');else document.documentElement.style.setProperty('--right',Math.max(340,Math.min(window.innerWidth-left-260,right-delta))+'px');};handle.onpointerup=()=>handle.onpointermove=null;};});
// A fragment access key is exchanged for an HttpOnly cookie and removed from history.
const fragment=new URLSearchParams(location.hash.slice(1)),query=new URLSearchParams(location.search);if(query.has('key')||fragment.has('password')||fragment.has('key')){const password=query.get('key')??fragment.get('password')??fragment.get('key');query.delete('key');const clean=query.toString();history.replaceState(null,'',location.pathname+(clean?'?'+clean:''));api('/api/login',{method:'POST',body:JSON.stringify({password})}).then(init).catch(e=>{showLogin();$('#login-error').textContent=e.message;});}else init();

$('#reload-file').onclick=async()=>{const p=state.activeFile,file=state.files.get(p);if(!p)return;if(file.dirty&&!confirm('변경을 버리고 디스크에서 다시 읽을까요?'))return;state.files.delete(p);await openFile(p);};

async function closeChat(id){
 const s=state.chats.find(s=>s.id===id);if(s?.running){await api('/api/chats/'+id+'/stop',{method:'POST',body:'{}'});for(let n=0;n<30;n++){await new Promise(r=>setTimeout(r,100));if(!(await api('/api/chats/'+id)).running)break;}}
 await api('/api/chats/'+id,{method:'DELETE'});localStorage.setItem('workbench.chatChanged',Date.now());await refreshChats();
 if(id===state.activeChat){state.stream?.close();if(state.chats.length)await selectChat(state.chats[0].id);else await newChat();}
}
function syncAgentDensity(frame){
 try{const root=frame.contentWindow.document.documentElement;frame.contentWindow.document.body.classList.add('embedded-agent');const css=getComputedStyle(document.documentElement);root.style.setProperty('--content-font',window.matchMedia('(max-width:767px)').matches?'14px':css.getPropertyValue('--content-font'));root.style.setProperty('--editor-font',css.getPropertyValue('--editor-font'));root.style.setProperty('--composer-font',getComputedStyle($('#prompt')).fontSize);}catch{}
}
function ensureAgentFrame(parent){
 let frame=$('#secondary-chat');if(!frame){frame=document.createElement('iframe');frame.id='secondary-chat';frame.title='AI Agent';frame.src='/?chatOnly=1&agent='+encodeURIComponent(new URLSearchParams(location.search).get('agent')||'');frame.addEventListener('load',()=>{try{frame.contentWindow.document.body.classList.toggle('embedded-one',document.body.classList.contains('one-chat'));syncAgentDensity(frame);}catch{}});}
 if(frame.parentElement!==parent)parent.append(frame);
 try{frame.contentWindow.document.body.classList.toggle('embedded-one',document.body.classList.contains('one-chat'));syncAgentDensity(frame);}catch{}
 return frame;
}
function selectSinglePane(mode){
 singlePaneMode=mode;localStorage.setItem('workbench.singlePane',mode);const one=document.body.classList.contains('one-chat');
 $('#primary-chat').hidden=one&&mode==='agents';const frame=$('#secondary-chat');if(frame)frame.hidden=one&&mode!=='agents';
 $('#mode-chat')?.classList.toggle('selected',mode==='chat'||!one);$('#mode-agents')?.classList.toggle('selected',mode==='agents'&&one);
}
function setLayout(value){
 if(chatOnly)return;const mobile=window.matchMedia('(max-width: 767px)').matches;if(!mobile)localStorage.setItem('workbench.layout',value);const one=mobile||value==='1';
 document.body.classList.toggle('one-chat',one);document.body.classList.toggle('two-chats',!one);$('#layout-one').classList.toggle('selected',one);$('#layout-two').classList.toggle('selected',!one);
 if(one){$('#bottom-chat-resize')?.remove();const frame=ensureAgentFrame($('#ai-panel'));frame.style.flexBasis='';selectSinglePane(singlePaneMode);}
 else{const frame=ensureAgentFrame($('#viewer'));frame.hidden=false;$('#primary-chat').hidden=false;let handle=$('#bottom-chat-resize');if(!handle){handle=document.createElement('div');handle.id='bottom-chat-resize';handle.title='아래 채팅 높이 조절 (최대 50%)';$('#viewer').insertBefore(handle,frame);handle.onpointerdown=e=>{handle.setPointerCapture(e.pointerId);const top=$('#viewer').getBoundingClientRect().bottom;handle.onpointermove=ev=>{const max=$('#viewer').clientHeight*.5;frame.style.flexBasis=Math.max(180,Math.min(max,top-ev.clientY))+'px';};handle.onpointerup=()=>handle.onpointermove=null;};}selectSinglePane(singlePaneMode);}
}
$('#layout-one').onclick=()=>setLayout('1');$('#layout-two').onclick=()=>setLayout('2');
function setTheme(theme,save=true){document.documentElement.dataset.theme=theme;if(save)localStorage.setItem(themeKey,theme);$('#theme-light').classList.toggle('selected',theme==='light');$('#theme-dark').classList.toggle('selected',theme==='dark');if(state.activeFile)renderFile();}
$('#theme-light').onclick=()=>setTheme('light');$('#theme-dark').onclick=()=>setTheme('dark');window.addEventListener('storage',e=>{if(e.key===themeKey)setTheme(e.newValue||'light',false);if(e.key==='workbench.chatChanged')refreshChats();});setTheme(localStorage.getItem(themeKey)||'light',false);

function showMenu(event,items){
 $('#context-menu')?.remove();const menu=document.createElement('div');menu.id='context-menu';menu.setAttribute('role','menu');
 for(const item of items){const button=document.createElement('button');button.textContent=item.label;if(item.danger)button.className='danger';button.onclick=()=>{menu.remove();Promise.resolve(item.run()).catch(e=>toast(e.message));};menu.append(button);}
 document.body.append(menu);menu.style.left=Math.min(event.clientX,window.innerWidth-185)+'px';menu.style.top=Math.min(event.clientY,window.innerHeight-items.length*42-12)+'px';
}
document.addEventListener('click',e=>{if(!e.target.closest('#context-menu'))$('#context-menu')?.remove();});window.addEventListener('keydown',e=>{if(e.key==='Escape')$('#context-menu')?.remove();});
async function operateFile(entry,action){
 let name;if(action==='rename'){name=prompt('새 이름',entry.name);if(!name||name===entry.name)return;}else if(!confirm(entry.path+'을(를) 삭제할까요? 서버 휴지통에 보관됩니다.'))return;
 const result=await api('/api/file-operation',{method:'POST',body:JSON.stringify({action,path:entry.path,name})});
 for(const [p,file]of [...state.files])if(p===entry.path||p.startsWith(entry.path+'/')){state.files.delete(p);if(action==='rename'){const changed=result.path+p.slice(entry.path.length);state.files.set(changed,file);if(state.activeFile===p)state.activeFile=changed;}else if(state.activeFile===p)state.activeFile=null;}
 if(state.activeFile)renderFile();else{$('#file-content').innerHTML='<div class="unsupported">파일을 선택하세요.</div>';state.editor?.destroy();state.editor=null;}tabs();
 const row=[...document.querySelectorAll('.tree-row')].find(r=>r.dataset.path===entry.path),container=row?.parentElement.parentElement;
 if(container){const parent=entry.path.split('/').slice(0,-1).join('/');const depth=parent?parent.split('/').length:0;await loadTree(container,parent,depth);}else await loadTree($('#tree'));
 toast(action==='rename'?'이름을 변경했습니다':'휴지통으로 이동했습니다');
}
const paneHeading=$('#ai-panel > .panel-heading > span');paneHeading.innerHTML=chatOnly?'<span class="spark">✳</span> AI Agent':'<span class="pane-switcher"><button id="mode-chat" class="selected">Chat</button><button id="mode-agents">AI Agent</button></span>';if(!chatOnly){$('#mode-chat').onclick=()=>selectSinglePane('chat');$('#mode-agents').onclick=()=>selectSinglePane('agents');}

$('#prompt').placeholder=chatOnly?'수정하거나 실행할 작업을 입력하세요…':'궁금한 점을 물어보세요…';

async function showRecentReports(container){
 container.innerHTML='<div class="recent-reports"><div class="recent-heading">최근 보고서</div><div class="loading-state"><span class="spinner"></span></div></div>';
 try{const reports=await api('/api/reports');if(state.events.some(e=>e.type==='user'))return;const list=container.querySelector('.recent-reports');if(!list)return;list.innerHTML='<div class="recent-heading">최근 보고서</div>';for(const report of reports){const button=document.createElement('button');button.className='recent-report';button.innerHTML='<span>▤</span><span>'+esc(report.name)+'</span><span>↗</span>';button.onclick=()=>openFile(report.path);list.append(button);}if(!reports.length){const text=document.createElement('p');text.className='muted';text.textContent='저장된 보고서가 없습니다.';list.append(text);}}
 catch(e){container.textContent=e.message;}
}

function renderGoal(goal){
 state.goal=goal;const panel=$('#goal-status'),icon=$('#goal-info');$('#send-chat').hidden=state.busy&&!goal;panel.hidden=false;icon.dataset.status=goal?.status||'none';icon.title=goal?'목표 상태: '+goal.status:'목표 상태 보기';if(!goal){panel.innerHTML='<p class="muted">현재 설정된 목표가 없습니다. Goal을 체크하고 목표를 전송해 주세요.</p>';return;}
 const labels={active:'진행 중',paused:'일시정지',complete:'완료',completed:'완료',blocked:'확인 필요',budget_limited:'예산 소진'};
 panel.innerHTML='<div class="goal-title"><b>◎ Goal · '+esc(labels[goal.status]||goal.status)+'</b><span>'+Number(goal.tokensUsed||0).toLocaleString()+' tokens</span></div><div class="goal-objective">'+esc(goal.objective)+'</div><div class="goal-actions"><button data-action="'+(goal.status==='active'?'pause':'resume')+'">'+(goal.status==='active'?'일시정지':'재개')+'</button><button data-action="clear">해제</button></div>';
 panel.querySelectorAll('button').forEach(b=>b.onclick=async()=>{try{const result=await api('/api/chats/'+state.activeChat+'/goal',{method:'POST',body:JSON.stringify({action:b.dataset.action})});renderGoal(result.goal);}catch(e){toast(e.message);}});
}

async function refreshGpu(){
 if(chatOnly||document.hidden)return;try{const {gpus}=await api('/api/gpus');const el=$('#gpu-grid');el.innerHTML='';for(const g of gpus){const row=document.createElement('div');row.className='gpustat-row '+(g.usedMiB>100?'gpu-busy':'gpu-free');const users=(g.users||[]).map(u=>u.name+'('+Math.round(u.memoryMiB)+'M)').join(' ');row.title='GPU '+g.index+' · 사용률 '+g.utilization+'% · VRAM '+g.usedMiB+' / '+g.totalMiB+' MB';row.innerHTML='<span class="gpu-number">['+g.index+']</span><span class="gpu-util">'+g.utilization+'%</span><span class="gpu-memory">'+g.usedMiB+' / '+g.totalMiB+' MB</span><span class="gpu-users">'+esc(users)+'</span>';el.append(row);}}catch(e){$('#gpu-grid').title=e.message;}
}
async function refreshLimits(){
 if(chatOnly||document.hidden)return;try{const data=await api('/api/limits');const limits=data.rateLimits||{};const el=$('#cli-limits');el.innerHTML='';for(const [key,fallback]of [['primary','단기'],['secondary','주간']]){const window=limits[key];if(!window)continue;const left=Math.max(0,100-window.usedPercent);const mins=window.windowDurationMins;const label=key==='primary'?'token limit':'token limit '+(mins&&mins%60===0?(mins/60)+'h':fallback);const row=document.createElement('div');row.className='limit-row';row.title=window.resetsAt?'초기화: '+new Date(window.resetsAt*1000).toLocaleString():'';row.innerHTML='<span>'+label+'</span><div class="mini-meter"><i style="width:'+left+'%"></i></div><b>'+left+'%</b>';el.append(row);}if(!el.childNodes.length)el.innerHTML='<span class="muted">계정 한도 정보 없음</span>';}catch(e){$('#cli-limits').innerHTML='<span class="muted">한도 조회 불가</span>';}
}
if(!chatOnly){const wait=setInterval(()=>{if(!$('#shell').hidden){clearInterval(wait);refreshGpu();refreshLimits();setInterval(refreshGpu,5000);setInterval(refreshLimits,60000);}},300);document.addEventListener('visibilitychange',()=>{if(!document.hidden){refreshGpu();refreshLimits();}});}

$('#goal-info').onclick=()=>{const dialog=$('#goal-dialog');if(!dialog.open)dialog.showModal();};
$('#goal-dialog-close').onclick=()=>$('#goal-dialog').close();
$('#goal-dialog').addEventListener('click',event=>{if(event.target===$('#goal-dialog')){const box=event.target.getBoundingClientRect();if(event.clientX<box.left||event.clientX>box.right||event.clientY<box.top||event.clientY>box.bottom)event.target.close();}});

let responsiveTimer;window.addEventListener('resize',()=>{if(chatOnly)return;clearTimeout(responsiveTimer);responsiveTimer=setTimeout(()=>setLayout(localStorage.getItem('workbench.layout')||'2'),150);});

$('#prompt').addEventListener('input',()=>{if(state.activeChat)localStorage.setItem('workbench.draft.'+pane+'.'+state.activeChat,$('#prompt').value);});
window.addEventListener('message',e=>{if(e.origin===location.origin&&e.data?.type==='file-context'){setContext(e.data.path);$('#prompt').focus();}});

window.addEventListener('message',e=>{if(e.origin===location.origin&&e.source===$('#secondary-chat')?.contentWindow&&e.data?.type==='open-workspace-file'&&typeof e.data.path==='string')openFile(e.data.path);});
$('#close-mobile-viewer').onclick=()=>document.body.classList.remove('mobile-viewer-open');
