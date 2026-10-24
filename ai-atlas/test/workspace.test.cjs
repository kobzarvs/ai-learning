const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const C=require('../workspace-core.js'),A=require('../core.js'),data=require('../../map-data.json');
const context={window:{}};vm.runInNewContext(fs.readFileSync(require.resolve('../lesson-data.js'),'utf8'),context);
const lessons=context.window.ATLAS_LESSONS,index=A.indexTree(data.root);
test('every leaf has runnable code, instructions, expected output and a bounded claim',()=>{
 const leaves=[...index.values()].filter(n=>!n.children);assert.equal(Object.keys(lessons.topics).length,113);
 for(const n of leaves){const l=lessons.topics[n.id];for(const key of ['goal','expected','exercise','explanation','limits','code'])assert.ok(l[key]?.length>30,n.id+': '+key);assert.ok(l.steps.length>=4);assert.match(l.code,/import torch/);assert.equal(l.code,fs.readFileSync(require.resolve('../../examples/python/'+n.id+'.py'),'utf8'));}
 assert.equal(Object.values(lessons.topics).filter(l=>l.rust).length,25);
 assert.equal(lessons.meta.torch,'2.14.1');assert.equal(lessons.meta.burn,'0.22.1');
 for(const l of Object.values(lessons.topics).filter(l=>l.rust)){assert.match(l.rust.cargo,/=0\.22\.1/);assert.doesNotMatch(l.rust.code,/Tensor<B|\.to_vec::/);}
});
test('each section has a navigable board and each lesson has an article and example',()=>{
 for(const n of index.values()){if(n.id==='root')continue;const b=C.defaultBoard(n,lessons.topics);assert.ok(b.cards.some(c=>c.type==='article'));if(n.children)assert.equal(b.cards.filter(c=>c.type==='topic').length,n.children.length);else assert.ok(b.cards.some(c=>c.type==='example'));}
});
test('group movement carries members; removing the group keeps their content',()=>{
 const b=C.defaultBoard(index.get('cnn'),lessons.topics),original={...b.cards[0]};
 const g=C.groupCards(b,[original.id],'Vision');C.moveGroup(b,g.id,50,-20);
 assert.equal(b.cards[0].x,original.x+50);assert.equal(b.cards[0].y,original.y-20);
 C.removeGroup(b,g.id);assert.equal(b.groups.length,0);assert.equal(b.cards[0].groupId,null);assert.equal(b.cards[0].body,original.body);
});
test('reverse selection rectangles intersect the same cards',()=>{
 const b=C.defaultBoard(index.get('cnn'),lessons.topics);
 assert.deepEqual(C.selectRect(b.cards,{x:0,y:0,w:450,h:340}),C.selectRect(b.cards,{x:450,y:340,w:-450,h:-340}));
 assert.equal(C.selectRect(b.cards,{x:0,y:0,w:450,h:340}).length,1);
});
test('export/import round trip keeps notes, groups and positions and rejects unrelated files',()=>{
 const b=C.defaultBoard(index.get('cnn'),lessons.topics),g=C.groupCards(b,[],'Study');
 const note=C.addNote(b,'Тест','<img src=x onerror=alert(1)>',{x:20,y:30},g.id);
 const restored=C.parseImport(JSON.stringify({version:1,boards:{cnn:b}}));
 assert.deepEqual(restored.boards.cnn,b);assert.ok(restored.boards.cnn.cards.some(c=>c.id===note.id));
 assert.throws(()=>C.parseImport('{"version":3,"boards":{}}'));
 const bad=C.parseImport('{"version":1,"boards":{"__proto__":{"polluted":true}}}');assert.equal({}.polluted,undefined);assert.equal(Object.keys(bad.boards).length,0);
});
test('new typography stays at least 16px and no serif display font returns',()=>{
 const css=fs.readFileSync(require.resolve('../styles.css'),'utf8');
 for(const m of css.matchAll(/font-size:\s*(\d+(?:\.\d+)?)px/g))assert.ok(Number(m[1])>=16);
 assert.doesNotMatch(css,/Times|Georgia|(?<!-)\bserif[;,]/);
});
