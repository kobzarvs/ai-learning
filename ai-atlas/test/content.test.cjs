const {test}=require('node:test'),assert=require('node:assert/strict');
const content=require('../content.js');
const C=require('../core.js');
const fs=require('node:fs'),vm=require('node:vm');
const mediaBlocks=require('./fixtures/media-blocks.json');

test('published illustrations render as images with captions and source attribution inside groups',()=>{
 const html=content.renderSections([{kind:'group',columns:2,blocks:mediaBlocks}]);
 assert.equal((html.match(/<img /g)||[]).length,2);
 assert.ok(html.includes('src="'+mediaBlocks[0].src+'"'));
 assert.match(html,/alt="Архитектура нейросети с ветвлением"/);
 assert.match(html,/<figcaption/);assert.match(html,/Рисунок 1/);assert.match(html,/Авторы статьи, CC BY 4.0/);
 assert.match(html,/href="https:\/\/example.org\/paper"/);assert.match(html,/referrerpolicy="no-referrer"/);
 const svg=html.match(/src="data:image\/svg\+xml;charset=utf-8,([^"]+)"/);
 assert.ok(svg,'SVG is an isolated image, not live page markup');
 assert.equal(decodeURIComponent(svg[1].replace(/&#39;/g,"'")),mediaBlocks[1].svg);
 assert.doesNotMatch(html,/<svg|<object|<iframe|<embed/);
});

test('media cannot inject page markup or unsafe source links, even for unvalidated records',()=>{
 const html=content.renderSections([{...mediaBlocks[1],title:'<script>bad()</script>',caption:'<img onerror="bad()">',sourceUrl:'javascript:bad()',svg:'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><script>bad()</script></svg>'}]);
 assert.doesNotMatch(html,/<script|<svg|<img onerror|href="javascript:/);
 assert.match(html,/&lt;script&gt;/);assert.match(html,/&lt;img onerror=/);
 for(const src of ['javascript:bad()','data:text/html,<script>bad()</script>','file:///etc/passwd','//example.org/x.png','https://user:pass@example.org/x.png','https://example.org/x.png']){
  assert.doesNotMatch(content.renderSections([{kind:'image',src,alt:'test'}]),/<img /,src);
 }
 const png='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aDecAAAAASUVORK5CYII=';
 assert.doesNotMatch(content.renderSections([{kind:'image',src:png,alt:'Pixel'}]),/<img /);
});

test('database content updates existing topic references, lessons and presentation',()=>{
 const data=structuredClone(require('../../map-data.json')),index=C.indexTree(data.root),node=index.get('causal');
 const lessons={meta:{python:'3.14'},topics:{causal:{goal:'old'}}},presentation={};
 content.apply([{id:'causal',payload:{node:{name:'Причинные модели',what:'Новая версия',id:'other',children:[]},lesson:{goal:'new'},layout:{python:{readerWidth:1420}},sections:{article:[{kind:'text',body:'Обновлённое объяснение'}]}}}],{data,index,lessons,presentation});
 assert.equal(index.get('causal'),node);
 assert.equal(node.what,'Новая версия');assert.equal(node.id,'causal');
 assert.equal(C.indexTree(data.root).get('causal').what,'Новая версия');
 assert.equal(lessons.topics.causal.goal,'new');assert.equal(presentation.causal.layout.python.readerWidth,1420);
});

test('incremental client keeps the last successful revision and coalesces requests',async()=>{
 let time=10000,calls=[],changes=[],fail=false,resolve;
 const client=content.createClient({now:()=>time,onChange:(records,updated)=>changes.push({records,updated}),fetch:(url)=>{
  calls.push(url);return new Promise(r=>{resolve=()=>r(fail?new Response('',{status:503}):Response.json({revision:'12',records:[{id:'causal',payload:{}}]}));});
 }});
 const first=client.refresh();assert.equal(client.refresh(),first);assert.equal(calls.length,1);
 resolve();assert.equal(await first,true);assert.equal(client.revision,'12');assert.equal(changes[0].updated,false);
 time+=30000;fail=true;const second=client.refresh();resolve();assert.equal(await second,false);assert.equal(client.revision,'12');
 assert.equal(calls[1],'/api/content?since=12');
});

test('published sections escape markup, draw numeric graphs and constrain layout',()=>{
 const html=content.renderSections([{kind:'text',title:'<img onerror=alert(1)>',body:'Первый абзац\n\nВторой',emphasis:'warning',width:100000},{kind:'chart',title:'Сравнение',xLabel:'Задания',yLabel:'Минуты',series:[{name:'Инструкция',values:[{x:1,y:20},{x:2,y:10},{x:3,y:NaN}]}]}]);
 assert.ok(html.includes('&lt;img'));assert.ok(!html.includes('<img'));assert.ok(!html.includes('NaN'));
 assert.match(html,/<svg/);assert.match(html,/max-width:1600px/);assert.match(html,/<p>Первый абзац<\/p><p>Второй<\/p>/);
 assert.deepEqual(content.layoutValues({readerWidth:99999,textWidth:'url(x)',chartsColumns:50}),{contentPadding:null,sectionGap:null,readerWidth:1800,textWidth:null,chartsColumns:null});
});

test('read endpoint is read-only, parameterized, uncached and hides database errors',async()=>{
 const {contentResponse}=await import('../content-db.mjs');let query;
 const connect=async()=>({query:async(statement,params)=>{query={statement,params};return[{revision:'9',records:[]}];}});
 const response=await contentResponse(new Request('https://atlas.test/api/content?since=8'),connect);
 assert.equal(response.status,200);assert.equal(response.headers.get('cache-control'),'no-store');assert.deepEqual(query.params,['8']);
 assert.deepEqual(await response.json(),{revision:'9',records:[]});
 assert.equal((await contentResponse(new Request('https://atlas.test/api/content',{method:'POST'}),connect)).status,405);
 assert.equal((await contentResponse(new Request('https://atlas.test/api/content?since=0%20OR%201=1'),connect)).status,400);
 const failure=await contentResponse(new Request('https://atlas.test/api/content'),async()=>{throw Error('postgres://secret');});
 assert.equal(failure.status,503);assert.ok(!(await failure.text()).includes('secret'));
});

test('migration covers every current topic and framework without replacing existing publications',async()=>{
 const {seedRecords,seedDatabase}=await import('../content-db.mjs');const context={window:{}};
 vm.runInNewContext(fs.readFileSync(require.resolve('../lesson-data.js'),'utf8'),context);
 const data=require('../../map-data.json'),records=seedRecords(data,context.window.ATLAS_LESSONS);
 assert.equal(records.length,C.indexTree(data.root).size+1);
 assert.equal(records.filter(r=>r.payload.lesson).length,113);
 assert.ok(records.find(r=>r.id==='causal').payload.lesson.guide);
 const statements=[];const sql={query:(statement,params)=>{statements.push({statement,params});return statement;},transaction:async queries=>queries};
 await seedDatabase(sql,records);
 assert.match(statements.at(-1).statement,/ON CONFLICT DO NOTHING/);
 assert.doesNotMatch(statements.at(-1).statement,/DO UPDATE/);
 assert.equal(JSON.parse(statements.at(-1).params[0]).length,records.length);
});

test('sources group project documentation and GitHub together without losing distinct publications',()=>{
 const sources=[
  ['EconML: дважды устойчивая оценка','https://github.com/py-why/EconML/blob/main/doc/spec/estimation/dr.rst'],
  ['EconML: S-, T- и X-оцениватели','https://www.pywhy.org/EconML/spec/estimation/metalearners.html'],
  ['Повтор той же ссылки','https://www.pywhy.org/EconML/spec/estimation/metalearners.html'],
  ['PyTorch: обучение','https://docs.pytorch.org/tutorials/'],
  ['PyTorch: релиз','https://github.com/pytorch/pytorch/releases/latest'],
  ['Pearl: книга','https://ftp.cs.ucla.edu/pub/stat_ser/r481.pdf'],
  ['Статья','https://arxiv.org/abs/1803.09015'],
  ['Другой материал','https://example.org/reference'],
  ['Недопустимая ссылка','javascript:alert(1)']
 ];
 const groups=content.groupSources(sources);
 assert.deepEqual(groups.map(g=>g.title),['PyTorch','EconML','Другие материалы','Книги и научные публикации']);
 assert.deepEqual(groups.map(g=>g.links.length),[2,2,1,2]);
 assert.equal(groups[1].links[0][0],'Дважды устойчивая оценка');
 assert.equal(groups.at(-1).links[0][0],'Pearl: книга');
 assert.deepEqual(new Set(groups.flatMap(g=>g.links.map(([,url])=>url))),new Set(sources.slice(0,-1).map(([,url])=>url)));
});

test('publication watermark bypasses throttle and catches up after an older in-flight response',async()=>{
 let resolve,calls=0;const changes=[];
 const client=content.createClient({now:()=>10000,onChange:r=>changes.push(r),fetch:async()=>{
  calls++;if(calls===1)return new Promise(r=>resolve=()=>r(Response.json({revision:'10',records:[]})));
  return Response.json({revision:'12',records:[{id:'knn',payload:{}}]});
 }});
 const first=client.refresh();const publication=client.refresh({minRevision:'12'});resolve();
 await Promise.all([first,publication]);assert.equal(client.revision,'12');assert.equal(calls,2);assert.equal(changes.length,1);
});
test('unmet publication watermark retries after failure without waiting for throttle',async()=>{
 let calls=0;const client=content.createClient({now:()=>10000,onChange:()=>{},fetch:async()=>++calls===1?new Response('',{status:503}):Response.json({revision:'9007199254740993',records:[]})});
 assert.equal(await client.refresh({minRevision:'9007199254740993'}),false);
 assert.equal(await client.refresh({minRevision:'9007199254740993'}),true);assert.equal(client.revision,'9007199254740993');
});

test('config groups keep blocks ordered and render nested columns with working charts',()=>{
 const html=content.renderSections([{kind:'text',body:'Перед сравнением'},{kind:'group',title:'До и после',columns:2,gap:20,blocks:[
  {kind:'chart',type:'scatter',title:'До',series:[{name:'Неизвестно',color:'gray',marker:'diamond',hollow:true,values:[{x:0,y:0,label:'P?'}]}]},
  {kind:'group',columns:1,blocks:[{kind:'text',body:'После решения'},{kind:'code',code:'print(1)'}]}
 ]},{kind:'text',body:'После сравнения'}]);
 assert.match(html,/--group-columns:2;--group-gap:20px/);assert.match(html,/<svg/);assert.match(html,/P\?/);
 assert.ok(html.indexOf('Перед сравнением')<html.indexOf('До и после'));assert.ok(html.indexOf('print(1)')<html.indexOf('После сравнения'));
});
test('chart points have interior axis padding and configurable, distinct markers',()=>{
 const html=content.renderSections([{kind:'chart',type:'scatter',series:[
  {name:'Известные',color:'green',values:[{x:0,y:0},{x:2,y:2}]},
  {name:'Новые',color:'gray',marker:'diamond',hollow:true,values:[{x:.5,y:.5,label:'P?'}]}
 ]}]);
 const circles=[...html.matchAll(/<circle cx="([\d.]+)" cy="([\d.]+)"/g)];assert.equal(circles.length,2);
 for(const [,x,y]of circles){assert.ok(+x>70&&+x<530);assert.ok(+y>50&&+y<285);}
 assert.match(html,/<path[^>]+fill="#fff"/);assert.match(html,/#64748b/);assert.match(html,/P\?/);
});
test('explicit chart domains keep comparisons on the same scale',()=>{
 const block={kind:'chart',type:'scatter',xDomain:[-1,3],yDomain:[-1,3],series:[{name:'A',values:[{x:1,y:1}]}]};
 const html=content.renderSections([block]);const point=html.match(/<circle cx="([\d.]+)" cy="([\d.]+)"/);
 assert.equal(+point[1],300);assert.equal(+point[2],167.5);
});

test('near-tight explicit domains still keep the whole marker inside the axes',()=>{
 const html=content.renderSections([{kind:'chart',type:'scatter',xDomain:[-.001,1.001],yDomain:[-.001,1.001],series:[{values:[{x:0,y:0},{x:1,y:1}]}]}]);
 for(const [,x,y]of html.matchAll(/<circle cx="([\d.]+)" cy="([\d.]+)"/g)){assert.ok(+x>=80&&+x<=520);assert.ok(+y>=60&&+y<=275);}
});
test('long labels near the right edge face inward with full text retained in the tooltip',()=>{
 const label='Предсказание: класс оранжевый';const html=content.renderSections([{kind:'chart',type:'scatter',series:[{values:[{x:0,y:0},{x:1,y:1,label}]}]}]);
 assert.ok(html.includes('<title>'+label));assert.match(html,/text-anchor="end"[^>]+textLength="190"/);assert.match(html,/Предсказание: кла…/);
});


test('category outlines enclose large bright points without creating extra legend categories',()=>{
 const html=content.renderSections([{kind:'chart',type:'scatter',pointSize:11,series:[
  {name:'Категория 0: A, B, P, Q',color:'blue',outline:true,values:[{x:0,y:0},{x:0,y:2},{x:.5,y:.5,label:'P (новая)'},{x:1.5,y:.5,label:'Q (новая)'}]},
  {name:'Категория 1: C',color:'orange',outline:true,values:[{x:2,y:0}]}
 ]}]);
 assert.equal((html.match(/class="chart-category-outline"/g)||[]).length,2);
 assert.match(html,/fill="#2563eb"/);assert.match(html,/fill="#f97316"/);assert.equal((html.match(/r="11"/g)||[]).length,5);
 assert.equal((html.match(/<span style="color:/g)||[]).length,2);assert.match(html,/Q .*Z/);
});

test('valid shared domains stay identical when outline groups have different extrema',()=>{
 const make=x=>({kind:'chart',type:'scatter',xDomain:[-.61,3.39],yDomain:[-.61,3.39],series:[{outline:true,values:[{x,y:x},{x:2,y:2}]}]});
 const coords=x=>[...content.renderSections([make(x)]).matchAll(/<circle cx="([\d.]+)" cy="([\d.]+)"/g)].map(m=>m.slice(1));
 assert.deepEqual(coords(0)[1],coords(.1)[1]);
});
