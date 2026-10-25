import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { contentResponse } from './content-db.mjs';
import { authResponse, notesResponse } from './editorial-db.mjs';
const allowed = new Map([
  ['/', ['index.html', 'text/html; charset=utf-8']],
  ['/index.html', ['index.html', 'text/html; charset=utf-8']],
  ['/styles.css', ['styles.css', 'text/css; charset=utf-8']],
  ['/core.js', ['core.js', 'text/javascript; charset=utf-8']],
  ['/app.js', ['app.js', 'text/javascript; charset=utf-8']],
  ['/map-data.js', ['map-data.js', 'text/javascript; charset=utf-8']],
  ...['lesson-data.js','content.js','workspace-core.js','vendor/prism.js','ui.js','art.js','launch.js','lesson-visuals.js','lesson-guide.js','editorial.js','workspace.js'].map(name=>['/'+name,[name,'text/javascript; charset=utf-8']]),
  ['/art/topic-icons.png', ['art/topic-icons.png', 'image/png']],
  ['/vendor/PRISM-LICENSE.txt', ['vendor/PRISM-LICENSE.txt', 'text/plain; charset=utf-8']]
]);
const port = Number(process.env.PORT || 4173);
createServer(async (req, res) => {
  if(new URL(req.url,'http://localhost').pathname==='/api/content'){
    const result=await contentResponse(new Request('http://localhost'+req.url,{method:req.method}));
    res.writeHead(result.status,Object.fromEntries(result.headers));res.end(await result.text());return;
  }
  const endpoint = new URL(req.url,'http://localhost').pathname;
  if(['/api/auth','/api/editorial-notes'].includes(endpoint)){
    const chunks=[];let size=0;
    for await(const chunk of req){size+=chunk.length;if(size>24000){res.writeHead(413);res.end('Too large');return;}chunks.push(chunk);}
    const request=new Request('http://localhost'+req.url,{method:req.method,headers:req.headers,...(!['GET','HEAD'].includes(req.method)?{body:Buffer.concat(chunks)}:{})});
    const result=await(endpoint==='/api/auth'?authResponse:notesResponse)(request);
    const headers=Object.fromEntries(result.headers);const cookies=result.headers.getSetCookie();if(cookies.length)headers['set-cookie']=cookies;
    res.writeHead(result.status,headers);res.end(await result.text());return;
  }
  const match = allowed.get(new URL(req.url, 'http://localhost').pathname);
  if (!match) { res.writeHead(404); res.end('Not found'); return; }
  try {
    const body = await readFile(new URL(match[0], import.meta.url));
    res.writeHead(200, { 'Content-Type': match[1], 'Cache-Control': 'no-store' });
    res.end(body);
  } catch { res.writeHead(500); res.end('Unable to read asset'); }
}).listen(port, '127.0.0.1', () => console.log(`AI Atlas: http://127.0.0.1:${port}`));
