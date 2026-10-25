const {test}=require('node:test'),assert=require('node:assert/strict');
const content=require('../content.js');
const C=require('../core.js');
const fs=require('node:fs'),vm=require('node:vm');

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
 assert.deepEqual(content.layoutValues({readerWidth:99999,textWidth:'url(x)',chartsColumns:50}),{readerWidth:1800,textWidth:null,chartsColumns:null});
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
  ['Недопустимая ссылка','javascript:alert(1)']
 ];
 const groups=content.groupSources(sources);
 assert.deepEqual(groups.map(g=>g.title),['PyTorch','EconML','Книги и научные публикации']);
 assert.deepEqual(groups.map(g=>g.links.length),[2,2,2]);
 assert.equal(groups[1].links[0][0],'Дважды устойчивая оценка');
 assert.equal(groups[2].links[0][0],'Pearl: книга');
 assert.deepEqual(new Set(groups.flatMap(g=>g.links.map(([,url])=>url))),new Set(sources.slice(0,-1).map(([,url])=>url)));
});
