const {test}=require('node:test');const assert=require('node:assert/strict');const fs=require('node:fs');const vm=require('node:vm');const C=require('../core.js');const ctx={window:{}};vm.runInNewContext(fs.readFileSync(require.resolve('../map-data.js'),'utf8'),ctx);const data=ctx.window.ATLAS_DATA,index=C.indexTree(data.root),map=C.layout(data.root);
test('all 113 topics are present and their references resolve',()=>{assert.equal(C.countLeaves(data.root),113);assert.equal(map.cards.length,113);assert.equal(new Set(map.cards.map(c=>c.id)).size,113);for(const n of index.values()){if(!n.children){assert.ok(n.what&&n.tasks&&n.note,n.id);assert.ok(Array.isArray(n.sources),n.id);for(const key of n.sources)assert.match(data.sources[key][1],/^https:\/\//);}}});
test('every card stays inside its group, with no overlap',()=>{for(const c of map.cards){const g=map.groups.find(g=>g.id===c.parent);assert.ok(g);assert.ok(c.x>=g.x&&c.y>=g.y&&c.x+c.w<=g.x+g.w&&c.y+c.h<=g.y+g.h,c.id);}for(let i=0;i<map.cards.length;i++)for(let j=i+1;j<map.cards.length;j++){const a=map.cards[i],b=map.cards[j];assert.ok(a.x+a.w<=b.x||b.x+b.w<=a.x||a.y+a.h<=b.y||b.y+b.h<=a.y,`${a.id} overlaps ${b.id}`);}});
test('search finds acronyms, Russian tasks, and both meanings of QNN',()=>{assert.equal(C.search(index,'CNN')[0].id,'cnn');assert.ok(C.search(index,'QNN').some(n=>n.id==='quantum'));assert.ok(C.search(index,'QNN').some(n=>n.id==='quant'));assert.ok(C.search(index,'команды').some(n=>n.id==='lab-text'));assert.equal(C.search(index,'this-will-never-match').length,0);assert.equal(C.search(index,'   ').length,0);});
test('zoom preserves the world position under the cursor and clamps scale',()=>{const before={x:31,y:-129,scale:.7},x=318,y=222;const after=C.zoomAt(before,1.4,x,y);assert.ok(Math.abs((x-before.x)/before.scale-(x-after.x)/after.scale)<1e-8);assert.ok(Math.abs((y-before.y)/before.scale-(y-after.y)/after.scale)<1e-8);assert.equal(C.zoomAt(before,100,0,0).scale,1.65);});
test('fit contains a group in the viewport',()=>{const box=map.groups.find(g=>g.id==='neural');const v=C.fitBox(box,1000,600,40);assert.ok(v.x+box.x*v.scale>=39);assert.ok(v.y+box.y*v.scale>=39);assert.ok(v.x+(box.x+box.w)*v.scale<=961);assert.ok(v.y+(box.y+box.h)*v.scale<=561);});
test('bad and stale browser storage cannot inject unknown route entries',()=>{assert.deepEqual([...C.restoreRoute('bad json',index)],[]);assert.deepEqual([...C.restoreRoute('{"x":1}',index)],[]);assert.deepEqual([...C.restoreRoute('["cnn","missing","cnn",null,"root"]',index)],['cnn']);});
const appSource=fs.readFileSync(require.resolve('../app.js'),'utf8');
const connections=vm.runInNewContext(appSource.match(/const links=(\[.*?\]);/)[1]);
test('all real connections reach their cards without crossing any card',()=>{
 for(const [from,to] of connections){
  const source=map.cards.find(c=>c.id===from),target=map.cards.find(c=>c.id===to);
  const points=C.routeConnection(map.cards,map.groups,source,target);
  assert.ok(points&&points.length>=2,`${from} → ${to}: no route`);
  const onBorder=(p,c)=>(p.x===c.x||p.x===c.x+c.w)&&p.y>=c.y&&p.y<=c.y+c.h||(p.y===c.y||p.y===c.y+c.h)&&p.x>=c.x&&p.x<=c.x+c.w;
  assert.ok(onBorder(points[0],source));assert.ok(onBorder(points.at(-1),target));
  for(let i=1;i<points.length;i++)assert.equal(C.segmentBlocked(points[i-1],points[i],map.cards),false,`${from} → ${to}: segment ${i} crosses a card`);
  const unrelated=map.groups.filter(g=>![source.parent,source.branch,target.parent,target.branch].includes(g.id));
  for(let i=1;i<points.length;i++)assert.equal(C.segmentBlocked(points[i-1],points[i],unrelated),false,`${from} → ${to}: crosses unrelated group`);
  assert.doesNotMatch(C.roundedRoute(points),/NaN|Infinity/);
 }
});
test('a blocker between aligned endpoints forces a detour',()=>{
 const source={id:'a',x:0,y:0,w:100,h:100},target={id:'b',x:400,y:0,w:100,h:100},blocker={id:'blocker',x:200,y:-30,w:100,h:160};
 const cards=[source,target,blocker],points=C.routeConnection(cards,[],source,target);
 assert.ok(points&&points.length>2);
 for(let i=1;i<points.length;i++)assert.equal(C.segmentBlocked(points[i-1],points[i],cards),false);
});
test('fully blocked endpoints do not fall back to a crossing line',()=>{
 const source={id:'a',x:0,y:0,w:100,h:100},target={id:'b',x:400,y:0,w:100,h:100};
 assert.equal(C.routeConnection([source,target,{id:'cover',x:-30,y:-30,w:160,h:160}],[],source,target),null);
});
