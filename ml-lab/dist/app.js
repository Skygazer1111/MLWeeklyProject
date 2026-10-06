const $ = selector => document.querySelector(selector);
const escapeHtml = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
const asText = value => Array.isArray(value) ? value.join('') : value || '';
let units = [], unit, notebook, busy = false, worker, kernelUnit, nextId = 0;
const notebooks = new Map(), pending = new Map();
function inline(text) {
  return escapeHtml(text).replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/`([^`]+)`/g, '<code>$1</code>');
}
function markdown(text) {
  return asText(text).split('\n\n').map(block => {
    if (/^#{1,3} /.test(block)) return block.split('\n').map(line => {
      const match = line.match(/^(#{1,3}) (.*)/);
      return match ? `<h${match[1].length}>${inline(match[2])}</h${match[1].length}>` : `<p>${inline(line)}</p>`;
    }).join('');
    if (/^\d+\. /.test(block)) return `<ol>${block.split('\n').map(l=>`<li>${inline(l.replace(/^\d+\. /,''))}</li>`).join('')}</ol>`;
    return `<p>${inline(block).replaceAll('\n','<br>')}</p>`;
  }).join('');
}
function safeTable(html) {
  const doc = new DOMParser().parseFromString(html, 'text/html');
  const table = doc.querySelector('table');
  if (!table) return null;
  const allowed = new Set(['TABLE','THEAD','TBODY','TFOOT','TR','TH','TD','STRONG','B','EM','I','SPAN','BR']);
  for (const el of [...table.querySelectorAll('*')]) {
    if (!allowed.has(el.tagName)) {el.replaceWith(document.createTextNode(el.textContent)); continue;}
    for (const attr of [...el.attributes]) el.removeAttribute(attr.name);
  }
  for (const attr of [...table.attributes]) table.removeAttribute(attr.name);
  return table.outerHTML;
}
function renderOutputs(index) {
  const cell = notebook.cells[index], output = $(`#output-${index}`);
  if (!output) return;
  output.replaceChildren();
  if (!cell.outputs?.length) {
    output.className = 'output empty';
    output.textContent = cell.execution_count ? 'Cell completed with no displayed output.' : 'Output will appear here after running this cell.';
    return;
  }
  output.className = 'output';
  const label = document.createElement('div'); label.className='output-label';
  label.textContent = `OUT [${cell.execution_count ?? ' '}]`; output.append(label);
  for (const entry of cell.outputs) {
    if (entry.output_type === 'error') {
      const pre = document.createElement('pre'); pre.className='error-output';
      pre.textContent = asText(entry.traceback).replace(/\x1b\[[0-9;]*m/g,'');
      if (Array.isArray(entry.traceback)) pre.textContent = entry.traceback.join('\n').replace(/\x1b\[[0-9;]*m/g,'');
      output.append(pre);
    } else if (entry.data?.['image/png']) {
      const img = document.createElement('img'); img.src = `data:image/png;base64,${asText(entry.data['image/png'])}`;
      img.alt = `Python-generated graph from Unit ${unit.number}, cell ${index}`; output.append(img);
    } else if (entry.data?.['text/html'] && safeTable(asText(entry.data['text/html']))) {
      const div = document.createElement('div'); div.className='table-wrap'; div.innerHTML=safeTable(asText(entry.data['text/html'])); output.append(div);
    } else {
      const pre = document.createElement('pre'); pre.textContent = asText(entry.text || entry.data?.['text/plain']); output.append(pre);
    }
  }
}
function sizeEditor(editor) {
  const lines = editor.value.split('\n').length;
  editor.style.height = `${Math.max(100, lines*20)}px`;
  editor.previousElementSibling.textContent = Array.from({length:lines},(_,i)=>i+1).join('\n');
}
function renderNotebook() {
  $('#cells').replaceChildren();
  let count = 0;
  notebook.cells.forEach((cell,index)=> {
    if (cell.cell_type==='markdown') {
      if (index===0) return;
      if (index===1) {
        const details = document.createElement('details'); details.className='intro-cell';
        details.innerHTML='<summary>Dataset, setup & reproducibility</summary><div class="markdown-cell">'+markdown(cell.source)+'</div>';
        $('#cells').append(details);
      } else {
        const div=document.createElement('div'); div.className='markdown-cell'; div.innerHTML=markdown(cell.source); $('#cells').append(div);
      }
      return;
    }
    if (cell.cell_type!=='code') return;
    count++;
    const article=document.createElement('article'); article.className='code-cell'; article.id=`cell-${index}`;
    article.innerHTML=`<div class="cell-top"><span class="cell-label"><b>IN [<span id="count-${index}">${cell.execution_count??' '}</span>]</b> Python <span class="cell-status" id="status-${index}"></span></span><button class="cell-run" data-cell="${index}" aria-label="Run code cell ${count}">▶ Run cell</button></div><div class="code-area"><pre class="line-numbers" aria-hidden="true"></pre><textarea class="code-editor" spellcheck="false" aria-label="Python code cell ${count}"></textarea></div><div id="output-${index}" class="output"></div>`;
    $('#cells').append(article);
    const editor = article.querySelector('textarea'); editor.value=asText(cell.source); sizeEditor(editor);
    editor.addEventListener('input',()=>{cell.source=editor.value; sizeEditor(editor); $('#output-mode').textContent='Edited code · rerun to update outputs';});
    editor.addEventListener('keydown',event=> {
      if (event.key==='Tab') {event.preventDefault();const a=editor.selectionStart,b=editor.selectionEnd;editor.setRangeText('    ',a,b,'end');editor.dispatchEvent(new Event('input'));}
      if (event.key==='Enter' && event.shiftKey) {event.preventDefault(); if(!busy) run([index]);}
    });
    article.querySelector('.cell-run').addEventListener('click',()=>run([index]));
    renderOutputs(index);
  });
}
async function loadUnit(number, {navigate=false}={}) {
  if (busy) stop('Run stopped because you changed units.');
  const selected=units.find(u=>u.number===number) || units[0];
  if(navigate) history.pushState({},'',`/unit-${selected.number}`);
  $('#unit-page').hidden=true; $('#page-loading').hidden=false;
  try {
    if (!notebooks.has(selected.number)) {
      const response=await fetch(`/notebooks/${selected.filename}`);
      if(!response.ok) throw new Error('The notebook could not be loaded.');
      notebooks.set(selected.number,await response.json());
    }
    unit=selected; notebook=notebooks.get(unit.number);
    if(kernelUnit!==unit.number) kernelUnit=null;
    document.title=`Unit ${unit.number} · ${unit.title} | ML Weekly Lab`;
    $('#breadcrumb').textContent=`Unit ${unit.number}`; $('#unit-kicker').textContent=`UNIT ${String(unit.number).padStart(2,'0')} / 05`;
    $('#unit-title').textContent=unit.title; $('#unit-summary').textContent=unit.summary;
    const short=[['Import & view dataset','Summary & statistics'],['Linear regression','Bayesian logistic & SVM'],['Three clustering methods','Principal component analysis'],['HMM sequential prediction'],['CART classification','Ensemble classification']][unit.number-1];
    $('#practice-list').innerHTML=short.map((s,i)=>`<span class="practice-chip" title="${escapeHtml(unit.practices[i])}"><b>${String(i+1).padStart(2,'0')}</b>${escapeHtml(s)}</span>`).join('');
    $('#unit-nav').innerHTML=units.map(u=>`<a href="/unit-${u.number}" data-unit="${u.number}" class="nav-item ${unit.number===u.number?'active':''}" ${unit.number===u.number?'aria-current="page"':''}><span class="nav-number">${String(u.number).padStart(2,'0')}</span>${escapeHtml(u.title)}</a>`).join('');
    $('#notice').hidden=true; $('#output-mode').textContent=notebook._fresh?'Your session outputs':'Reference run · verified outputs';
    $('#next-unit').href=`/unit-${unit.number===5?1:unit.number+1}`; $('#next-unit').textContent=unit.number===5?'Back to Unit 1':'Next unit';
    renderNotebook(); setStatus(worker?'Python ready':'Python ready to start');
    $('#page-loading').hidden=true; $('#unit-page').hidden=false;
  } catch(error) {$('#page-loading').textContent=`${error.message} Reload the page to retry.`;}
}
function setStatus(text, type='') {$('#kernel-status').textContent=text;$('#kernel-status').className=`kernel-status ${type}`;}
function notice(text,error=false) {$('#notice').hidden=false;$('#notice').textContent=text;$('#notice').className=`notice ${error?'error':''}`;}
function setBusy(value) {
  busy=value;
  $('#run-all').disabled=value;$('#reset').disabled=value;$('#download').disabled=value;$('#stop').hidden=!value;
  document.querySelectorAll('.cell-run').forEach(b=>b.disabled=value);
  document.querySelectorAll('.code-editor').forEach(e=>e.readOnly=value);
}
function terminateWorker() {
  worker?.terminate();worker=null;kernelUnit=null;
  for(const {reject,timer} of pending.values()){clearTimeout(timer);reject(new Error('Execution stopped.'));}pending.clear();
}
function request(action,code) {
  if (!worker) {
    worker=new Worker('/python-worker.js',{type:'module'});
    worker.onmessage=({data})=>{
      if(data.type==='status'){setStatus(data.message,'running');return;}
      const item=pending.get(data.id);if(!item)return;clearTimeout(item.timer);pending.delete(data.id);
      if(data.error)item.reject(new Error(data.error));else item.resolve(data.result);
    };
    worker.onerror=event=>{event.preventDefault();const message=event.message||'Python could not start. Check your internet connection and try again.';for(const p of pending.values()){clearTimeout(p.timer);p.reject(new Error(message));}pending.clear();worker?.terminate();worker=null;kernelUnit=null;};
  }
  return new Promise((resolve,reject)=>{
    const id=++nextId;
    const timer=setTimeout(()=>{terminateWorker();setStatus('Execution timed out','error');},300000);
    pending.set(id,{resolve,reject,timer});worker.postMessage({id,action,code});
  });
}
let runToken=0;
async function run(indices,all=false) {
  if(busy)return;
  const token=++runToken;
  setBusy(true);$('#notice').hidden=true;
  notebook._fresh=true;$('#output-mode').textContent='Running your Python code…';
  try {
    if(all || kernelUnit!==unit.number) {await request('reset');kernelUnit=unit.number;}
    if(all) for(const i of indices){notebook.cells[i].outputs=[];notebook.cells[i].execution_count=null;renderOutputs(i);$(`#count-${i}`).textContent=' ';$(`#status-${i}`).textContent='';}
    for(let n=0;n<indices.length;n++) {
      if(token!==runToken)return;
      const index=indices[n],cell=notebook.cells[index];
      setStatus(`Running cell ${n+1} of ${indices.length}…`,'running');$(`#status-${index}`).textContent='running';
      const start=performance.now();const result=await request('execute',asText(cell.source));
      if(token!==runToken)return;
      cell.outputs=result.outputs;cell.execution_count=result.execution_count;
      $(`#count-${index}`).textContent=result.execution_count;$(`#status-${index}`).textContent=`${((performance.now()-start)/1000).toFixed(1)}s`;
      renderOutputs(index);
      if(result.failed){notice('This cell raised a Python error. Check its output below, correct the code, and rerun. Run all first if required variables are missing.',true);setStatus('Cell error','error');$('#output-mode').textContent='Your session · cell error';return;}
    }
    setStatus('Python ready');$('#output-mode').textContent='Your session · execution complete';
    if(!all)$(`#output-${indices[0]}`).scrollIntoView({behavior:'smooth',block:'nearest'});
  }catch(error){if(token===runToken){notice(`Could not run Python: ${error.message} Check your internet connection, then try Run all again.`,true);setStatus('Python unavailable','error');$('#output-mode').textContent='Run did not complete';terminateWorker();}}
  finally{if(token===runToken)setBusy(false);}
}
function stop(message='Execution stopped. Run all to restart with a fresh kernel.') {
  ++runToken;terminateWorker();setBusy(false);setStatus('Python stopped');notice(message);$('#output-mode').textContent='Your session · stopped';
  document.querySelectorAll('.cell-status').forEach(el=>{if(el.textContent==='running')el.textContent='stopped';});
}
$('#run-all').addEventListener('click',()=>run(notebook.cells.map((c,i)=>c.cell_type==='code'?i:null).filter(i=>i!==null),true));
$('#stop').addEventListener('click',()=>stop());
$('#reset').addEventListener('click',()=>{terminateWorker();setStatus('Python ready to start');notice('Kernel cleared. Run all to initialize variables again. Existing outputs remain visible.');});
$('#download').addEventListener('click',()=>{
  const exported=structuredClone(notebook);delete exported._fresh;
  const url=URL.createObjectURL(new Blob([JSON.stringify(exported,null,2)],{type:'application/x-ipynb+json'}));
  const a=document.createElement('a');a.href=url;a.download=unit.filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
document.addEventListener('click',event=>{
  const link=event.target.closest('a');const match=link?.getAttribute('href')?.match(/^\/unit-([1-5])$/);
  if(match && !event.ctrlKey && !event.metaKey){event.preventDefault();loadUnit(Number(match[1]),{navigate:true});window.scrollTo(0,0);}
});
window.addEventListener('popstate',()=>loadUnit(Number(location.pathname.match(/unit-(\d)/)?.[1]||1)));
async function start(){
  try{const response=await fetch('/units.json');if(!response.ok)throw new Error('Course files unavailable');units=await response.json();await loadUnit(Number(location.pathname.match(/unit-(\d)/)?.[1]||1));}
  catch(error){$('#page-loading').textContent=`${error.message}. Reload to retry.`;}
}
await start();
if(document.modelContext?.registerTool) {
  try {
    document.modelContext.registerTool({name:'navigate_ml_unit',title:'Open a machine learning unit',description:'Open one of the five course notebooks.',
      inputSchema:{type:'object',properties:{unit:{type:'integer',minimum:1,maximum:5}},required:['unit'],additionalProperties:false},
      annotations:{readOnlyHint:false,untrustedContentHint:false},
      async execute(input){if(!Number.isInteger(input.unit)||input.unit<1||input.unit>5)throw new Error('Unit must be 1 through 5.');await loadUnit(input.unit,{navigate:true});return {unit:unit.number,title:unit.title};}});
  } catch(error){console.warn('Optional notebook navigation tools unavailable.',error);}
}
