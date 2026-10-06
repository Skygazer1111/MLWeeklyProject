import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawn} from 'node:child_process';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../dist');
const types={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.json':'application/json','.csv':'text/csv; charset=utf-8','.ipynb':'application/x-ipynb+json','.zip':'application/zip'};
http.createServer(async(req,res)=>{
  try{
    let pathname=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
    if(pathname==='/' || /^\/unit-[1-5]\/?$/.test(pathname))pathname='/index.html';
    const target=path.resolve(root,'.'+pathname);
    if(!target.startsWith(root+path.sep)){res.writeHead(403);res.end();return;}
    const content=await fs.readFile(target);
    res.writeHead(200,{'Content-Type':types[path.extname(target)]||'application/octet-stream','Cache-Control':'no-cache'});res.end(content);
  }catch{res.writeHead(404);res.end('File not found');}
}).listen(8000,'127.0.0.1',()=>{
  console.log('Local URL: http://127.0.0.1:8000');
  if(process.argv.includes('--open') && process.platform==='win32')
    spawn('cmd.exe',['/c','start','','http://127.0.0.1:8000'],{stdio:'ignore',windowsHide:true});
});
