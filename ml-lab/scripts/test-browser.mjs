import {fileURLToPath} from 'node:url';
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
process.env.PLAYWRIGHT_BROWSERS_PATH=path.join(root,'.sites-runtime/browsers');
const {chromium}=await import('playwright');
const browser=await chromium.launch({headless:true});
const page=await browser.newPage({viewport:{width:1440,height:1000}});
const errors=[];page.on('pageerror',e=>{errors.push(e.message);console.error('PAGE ERROR',e.message);});
const results=[];await fs.mkdir(path.join(root,'test-results'),{recursive:true});
try{
  await page.goto('http://127.0.0.1:8000/unit-1');
  await page.locator('#unit-page').waitFor({state:'visible'});
  assert.equal(await page.locator('nav a').count(),5);
  await page.screenshot({path:path.join(root,'test-results/desktop.png'),fullPage:false});
  for(let number=1;number<=5;number++){
    if(number>1){await page.locator(`nav a[data-unit="${number}"]`).click();await page.waitForFunction(n=>document.querySelector('#breadcrumb').textContent===`Unit ${n}`,number);}
    await page.getByRole('button',{name:'▶ Run all',exact:true}).click();
    await page.waitForFunction(()=>!document.querySelector('#run-all').disabled,{timeout:300000});
    const mode=await page.locator('#output-mode').textContent();
    const notice=await page.locator('#notice').textContent();
    if(mode!=='Your session · execution complete')console.error('PYTHON FAILURE',await page.locator('.error-output').allTextContents());
    assert.equal(mode,'Your session · execution complete',`Unit ${number} failed: ${notice}`);
    assert.equal(await page.locator('.error-output').count(),0);
    const graphs=await page.locator('.output img').count();assert(graphs>=2,`Unit ${number} missing graphs`);
    const tables=await page.locator('.output table').count();assert(tables>=1);
    results.push({unit:number,graphs,tables,result:'passed'});console.log(JSON.stringify(results.at(-1)));
  }
  const editor=page.locator('.code-editor').first(),original=await editor.inputValue();
  await editor.fill(original+'\nprint("EDITED_CODE_EXECUTED_42")');
  await page.locator('.cell-run').first().click();
  await page.waitForFunction(()=>!document.querySelector('#run-all').disabled,{timeout:60000});
  assert((await page.locator('.output').first().textContent()).includes('EDITED_CODE_EXECUTED_42'));
  await editor.fill('raise ValueError("EXPECTED_TEST_ERROR")');
  await page.locator('.cell-run').first().click();
  await page.waitForFunction(()=>!document.querySelector('#run-all').disabled,{timeout:60000});
  assert((await page.locator('.error-output').textContent()).includes('EXPECTED_TEST_ERROR'));
  await editor.fill(original+'\nprint("RECOVERED_AFTER_ERROR")');
  await page.locator('.cell-run').first().click();
  await page.waitForFunction(()=>!document.querySelector('#run-all').disabled,{timeout:60000});
  assert.equal(await page.locator('.error-output').count(),0);
  assert((await page.locator('.output').first().textContent()).includes('RECOVERED_AFTER_ERROR'));
  const downloadPromise=page.waitForEvent('download');await page.locator('#download').click();const download=await downloadPromise;
  const exported=JSON.parse(await fs.readFile(await download.path(),'utf8'));assert(exported.cells.find(c=>c.cell_type==='code').source.includes('RECOVERED_AFTER_ERROR'));
  assert(!('_fresh' in exported));assert(exported.cells.some(c=>c.outputs?.some(o=>o.data?.['image/png'])));
  await editor.fill('import time\ntime.sleep(20)');await page.locator('.cell-run').first().click();
  await page.locator('#stop').waitFor({state:'visible'});await page.locator('#stop').click();
  assert.equal(await page.locator('#kernel-status').textContent(),'Python stopped');
  assert.equal(await page.locator('#run-all').isEnabled(),true);
  await editor.fill(original);
  await page.locator('nav a[data-unit="1"]').click();await page.waitForFunction(()=>document.querySelector('#breadcrumb').textContent==='Unit 1');
  await page.setViewportSize({width:390,height:844});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Mobile horizontal page overflow');
  await page.screenshot({path:path.join(root,'test-results/mobile.png'),fullPage:false});
  // Deep links must work independently, not only after client-side navigation.
  for(let number=1;number<=5;number++){
    await page.goto(`http://127.0.0.1:8000/unit-${number}`);await page.locator('#unit-page').waitFor({state:'visible'});
    assert.equal(await page.locator('#breadcrumb').textContent(),`Unit ${number}`);
  }
  const pack=await page.request.get('http://127.0.0.1:8000/ml-weekly-notebooks.zip');assert(pack.ok());assert((await pack.body()).length>100000);
  assert.deepEqual(errors,[]);
  results.push({editing:'passed',errorRecovery:'passed',download:'passed',stop:'passed',mobile:'passed',deepLinks:'passed',webmcp:'Supported browser context unavailable; optional registration feature-detected'});
  await fs.writeFile(path.join(root,'browser-verification.json'),JSON.stringify(results,null,2));
  console.log('BROWSER VERIFICATION PASSED');
}finally{await browser.close();}
