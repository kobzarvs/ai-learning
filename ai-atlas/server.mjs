import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
const allowed = new Map([
  ['/', ['index.html', 'text/html; charset=utf-8']],
  ['/index.html', ['index.html', 'text/html; charset=utf-8']],
  ['/styles.css', ['styles.css', 'text/css; charset=utf-8']],
  ['/core.js', ['core.js', 'text/javascript; charset=utf-8']],
  ['/app.js', ['app.js', 'text/javascript; charset=utf-8']],
  ['/map-data.js', ['map-data.js', 'text/javascript; charset=utf-8']]
]);
const port = Number(process.env.PORT || 4173);
createServer(async (req, res) => {
  const match = allowed.get(new URL(req.url, 'http://localhost').pathname);
  if (!match) { res.writeHead(404); res.end('Not found'); return; }
  try {
    const body = await readFile(new URL(match[0], import.meta.url));
    res.writeHead(200, { 'Content-Type': match[1], 'Cache-Control': 'no-store' });
    res.end(body);
  } catch { res.writeHead(500); res.end('Unable to read asset'); }
}).listen(port, '127.0.0.1', () => console.log(`AI Atlas: http://127.0.0.1:${port}`));
