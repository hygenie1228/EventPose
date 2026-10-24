import {chromium} from 'playwright';
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
const root=path.resolve(import.meta.dirname||path.dirname(new URL(import.meta.url).pathname),'../../..');
await import('./prepare-fixtures.mjs');
const base=process.env.WORKBENCH_URL||'http://147.46.89.70:18080';
const key=process.env.WORKBENCH_PASSWORD||(await fs.readFile(path.join(root,'.experiment_storage/research_workbench/access-key'),'utf8')).trim();
const output=path.join(root,'exp/261024/research_workbench');await fs.mkdir(output,{recursive:true});
const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
const page=await browser.newPage({viewport:{width:1600,height:1000}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
const report={};
try{
 await page.goto(base);await page.locator('#access-key').fill(key);await page.locator('#login-form button').click();await page.locator('#shell').waitFor({state:'visible'});await page.locator('.tree-row').first().waitFor();report.login=true;
 const columns=await page.locator('#workspace').evaluate(el=>[...el.children].filter(e=>e.tagName!=='DIV'||!e.classList.contains('resize-handle')).map(e=>({id:e.id,x:e.getBoundingClientRect().x})));assert(columns[0].x<columns[1].x&&columns[1].x<columns[2].x);report.threeColumns=true;
 await page.locator('[data-open="_agents/report/nlf_multihmr_comparison.md"]').click();await page.locator('article table').first().waitFor();assert((await page.locator('article').innerText()).includes('NLF pretrained'));report.markdown=true;
 await page.locator('#attach-file').click();assert((await page.locator('#context-chip').innerText()).includes('nlf_multihmr_comparison.md'));report.fileContext=true;
 await page.screenshot({path:path.join(output,'workbench-report.png'),fullPage:true});
 // Navigate the actual explorer, including nested folders.
 for(const p of ['apps','apps/research-workbench','apps/research-workbench/test','apps/research-workbench/test/fixtures'])await page.locator(`.tree-row[data-path="${p}"]`).click();
 await page.locator('.tree-row[data-path="apps/research-workbench/test/fixtures/editor.md"]').click();await page.locator('#preview-toggle').click();await page.locator('.cm-content').waitFor();await page.locator('.cm-content').click();await page.keyboard.press('Control+End');await page.keyboard.type('\nSaved from browser.');await page.locator('#save-file').click();await page.waitForFunction(()=>document.querySelector('#save-file').disabled);assert((await fs.readFile(path.join(root,'apps/research-workbench/test/fixtures/editor.md'),'utf8')).includes('Saved from browser.'));report.editorSave=true;
 await page.locator('.tree-row[data-path="apps/research-workbench/test/fixtures/preview.png"]').click();await page.locator('.media-stage img').waitFor();await page.waitForFunction(()=>document.querySelector('.media-stage img')?.naturalWidth===320);report.image=true;
 await page.locator('.tree-row[data-path="apps/research-workbench/test/fixtures/preview.webm"]').click();await page.locator('video').waitFor();await page.waitForFunction(()=>document.querySelector('video')?.readyState>=1);await page.locator('video').evaluate(async v=>{v.muted=true;await v.play();});await page.waitForFunction(()=>document.querySelector('video').currentTime>.2);await page.locator('video').evaluate(v=>{v.currentTime=1;v.pause();});report.videoPlayAndSeek=true;
 await page.locator('#new-chat').click();await page.locator('#prompt').fill('Browser integration test. Reply exactly BROWSER_CHAT_OK. Do not use tools or change files.');await page.locator('#send-chat').click();await page.waitForFunction(()=>document.querySelector('#messages').innerText.includes('BROWSER_CHAT_OK'),{},{timeout:120000});await page.waitForFunction(()=>!document.querySelector('#stop-chat').hidden===false,{},{timeout:120000});report.actualCodexChat=true;
 await page.locator('#prompt').fill('Reply with the exact marker from your previous answer followed by _RESUMED. Do not use tools.');await page.locator('#send-chat').click();await page.waitForFunction(()=>document.querySelector('#messages').innerText.includes('BROWSER_CHAT_OK_RESUMED'),{},{timeout:120000});report.resume=true;
 await page.locator('#new-chat').click();await page.locator('#messages .chat-welcome').waitFor();assert((await page.locator('.chat-tab').count())>=3);report.multipleChats=true;
 await page.reload();await page.locator('#shell').waitFor({state:'visible'});assert((await page.locator('.chat-tab').count())>=3);report.reloadPersistence=true;
 assert.deepEqual(errors,[]);report.noBrowserErrors=true;await page.screenshot({path:path.join(output,'workbench-media.png'),fullPage:true});
 console.log(JSON.stringify(report,null,2));await fs.writeFile(path.join(output,'browser-test.json'),JSON.stringify(report,null,2));
}finally{await browser.close();}
