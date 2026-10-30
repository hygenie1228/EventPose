import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
const base='http://127.0.0.1:18080';
const key=(await fs.readFile('.experiment_storage/research_workbench/access-key','utf8')).trim();
const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
try{
 for(const [width,height] of [[1440,900],[390,844]]){
  const page=await browser.newPage({viewport:{width,height}}),errors=[];let requests=0;
  page.on('pageerror',e=>errors.push(e.message));
  await page.context().addCookies([{name:'workbench',value:key,url:base}]);
  page.on('response',r=>{if(r.url().endsWith('/api/gpus')&&r.status()===200)requests++;});
  await page.goto(base);await page.locator('#shell').waitFor({state:'visible'});
  if(width<767)await page.locator('#gpu-open').click();
  await page.waitForFunction(()=>document.querySelector('#gpu-live-status').dataset.state==='live');
  assert.equal(await page.locator('.gpustat-row').count(),7);
  assert.equal(await page.locator('#gpu-live-status').isVisible(),false);
  assert.equal(await page.locator('#gpu-live-status').textContent(),'');
  await page.waitForResponse(r=>r.url().endsWith('/api/gpus')&&r.status()===200,{timeout:20000});
  assert(requests>=2);assert(await page.locator('#gpu-grid').isVisible());
  const bounds=await page.locator('#gpu-grid').boundingBox();assert(bounds.y+bounds.height<=height);
  if(width<767){await page.locator('#gpu-close').click();assert.equal(await page.locator('#gpu-dialog').isVisible(),false);await page.locator('#gpu-open').click();}
  else await page.locator('#gpu-open').click();
  await page.route('**/api/gpus',route=>route.fulfill({status:503,contentType:'application/json',body:'{"error":"test unavailable"}'}));
  await page.locator('#gpu-refresh').click();
  await page.waitForFunction(()=>document.querySelector('#gpu-live-status').dataset.state==='error');
  assert.equal(await page.locator('.gpustat-row').count(),7);
  await page.unroute('**/api/gpus');await page.locator('#gpu-refresh').click();
  await page.waitForFunction(()=>document.querySelector('#gpu-live-status').dataset.state==='live');
  await page.locator('#gpu-close').click();
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({width,height,requests,rows:7,live:true,recovery:true}));
  await page.close();
 }
}finally{await browser.close();}
