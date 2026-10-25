import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { contentResponse } from './content-db.mjs';
import { authResponse, notesResponse } from './editorial-db.mjs';
import { agentRunsResponse } from './agent-api.mjs';
import { notificationsResponse } from './notifications-api.mjs';
import { searchResponse } from './search-api.mjs';
import { editResponse } from './content-edit-api.mjs';
import { historyResponse } from './editorial-history-api.mjs';
const allowed = new Map([
  ['/', ['index.html', 'text/html; charset=utf-8']],
  ['/index.html', ['index.html', 'text/html; charset=utf-8']],
  ['/styles.css', ['styles.css', 'text/css; charset=utf-8']],
  ['/image-viewer.css', ['image-viewer.css', 'text/css; charset=utf-8']],
  ['/core.js', ['core.js', 'text/javascript; charset=utf-8']],
  ['/app.js', ['app.js', 'text/javascript; charset=utf-8']],
  ['/map-data.js', ['map-data.js', 'text/javascript; charset=utf-8']],
  ...['lesson-data.js','content-edit.js','content.js','reader-edit.js','image-viewer.js','workspace-core.js','vendor/prism.js','ui.js','art.js','launch.js','lesson-visuals.js','lesson-guide.js','notifications.js','search.js','editorial.js','workspace.js'].map(name=>['/'+name,[name,'text/javascript; charset=utf-8']]),
  ...['ai-atlas-og-v1.png','logo.png'].map(name=>['/social/'+name,['social/'+name,'image/png']]),
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
  if(['/api/auth','/api/editorial-notes','/api/agent-runs','/api/editorial-notifications','/api/search','/api/content-edit','/api/editorial-history'].includes(endpoint)){
    const chunks=[];let size=0;
    for await(const chunk of req){size+=chunk.length;if(size>24000){res.writeHead(413);res.end('Too large');return;}chunks.push(chunk);}
    const request=new Request('http://localhost'+req.url,{method:req.method,headers:req.headers,...(!['GET','HEAD'].includes(req.method)?{body:Buffer.concat(chunks)}:{})});
    const result=await({'/api/auth':authResponse,'/api/editorial-notes':notesResponse,'/api/agent-runs':agentRunsResponse,'/api/editorial-notifications':notificationsResponse,'/api/search':searchResponse,'/api/content-edit':editResponse,'/api/editorial-history':historyResponse}[endpoint])(request);
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
