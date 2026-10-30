import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
const base='http://127.0.0.1:18080';
const key=(await fs.readFile('.experiment_storage/research_workbench/access-key','utf8')).trim();
const report='_agents/report/261027_오래된_채팅_표시_제한.md';
const special='exp/261028/한글 공백 & #/이미지.png';
const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
try{
 for(const width of [1440,390]){
  const context=await browser.newContext({viewport:{width,height:900},permissions:['clipboard-read','clipboard-write']});
  await context.addCookies([{name:'workbench',value:key,url:base}]);
  const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.addInitScript(()=>localStorage.setItem('workbench.lastFile','README.md'));
  const link=new URL(base);link.searchParams.set('file',report);link.searchParams.set('keep','yes');
  await page.goto(link.href);await page.locator('#share-file').waitFor({state:'visible'});
  assert((await page.locator('#breadcrumb').textContent()).includes('오래된_채팅'));
  await page.locator('#share-file').click();
  let copied=await page.evaluate(()=>navigator.clipboard.readText());assert.equal(new URL(copied).searchParams.get('file'),report);assert.equal(new URL(copied).searchParams.get('keep'),'yes');assert(!new URL(copied).searchParams.has('key'));
  // Reuse the embedded Agent file-open bridge, including special URL characters.
  const frame=page.frameLocator('#secondary-chat');await frame.locator('#prompt').waitFor({state:'attached'});
  await frame.locator('body').evaluate((el,path)=>window.parent.postMessage({type:'open-workspace-file',path},location.origin),special);
  await page.waitForFunction(path=>new URL(location.href).searchParams.get('file')===path,special);
  assert(await page.locator('#viewer').isVisible());
  // HTTP fallback: environments without the Clipboard API still copy the full URL.
  await page.evaluate(()=>{Object.defineProperty(navigator,'clipboard',{value:undefined,configurable:true});document.execCommand=command=>{window.copiedFallback=document.activeElement.value;return command==='copy';};});
  await page.locator('#share-file').click();copied=await page.evaluate(()=>window.copiedFallback);assert.equal(new URL(copied).searchParams.get('file'),special);
  await page.reload();await page.locator('#share-file').waitFor({state:'visible'});assert.equal(new URL(page.url()).searchParams.get('file'),special);
  await page.locator('#close-all-files').click();assert(!new URL(page.url()).searchParams.has('file'));assert.equal(await page.locator('#share-file').isVisible(),false);
  assert.deepEqual(errors,[]);console.log(JSON.stringify({width,directLink:true,urlUpdates:true,copy:true,httpFallback:true,reload:true,closeClearsURL:true}));await context.close();
 }
 // The requested file survives an initial login screen.
 const context=await browser.newContext();const page=await context.newPage();
 await page.route('**/api/login',route=>route.fulfill({json:{ok:true},headers:{'Set-Cookie':`workbench=${key}; HttpOnly; SameSite=Strict; Path=/`}}));
 await page.goto(base+'/?file='+encodeURIComponent(report));await page.locator('#login').waitFor({state:'visible'});await page.locator('#access-key').fill('test-login');await page.locator('#login-form button.primary').click();await page.locator('#share-file').waitFor({state:'visible'});assert.equal(new URL(page.url()).searchParams.get('file'),report);console.log('login preserves requested file');await context.close();
}finally{await browser.close();}
