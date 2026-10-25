const {test}=require('node:test'),assert=require('node:assert/strict');
const editing=require('../content-edit.js');

test('delete disjoint paragraphs and list items without shifting other selected indices',()=>{
 const current={node:{id:'test',name:'Topic'},lesson:{code:'keep()'},sections:{article:[
  {kind:'text',title:'A',body:'Один\n\nДва\n\nТри\n\nЧетыре'},
  {kind:'group',blocks:[{kind:'list',items:['a','b','c']},{kind:'svg',svg:'preserved'}]}
 ]}};
 const next=editing.remove(current,[{path:'sections.article.0.body',part:0},{path:'sections.article.0.body',part:2},
  {path:'sections.article.1.blocks.0.items.0'},{path:'sections.article.1.blocks.0.items.2'}]);
 assert.equal(next.sections.article[0].body,'Два\n\nЧетыре');
 assert.deepEqual(next.sections.article[1].blocks[0].items,['b']);
 assert.equal(next.sections.article[1].blocks[1].svg,'preserved');assert.equal(next.lesson.code,'keep()');
 assert.equal(current.sections.article[0].body,'Один\n\nДва\n\nТри\n\nЧетыре');
});
test('section selection removes nested blocks once and leaves an explicit empty tab',()=>{
 const current={sections:{article:[{kind:'group',title:'All',blocks:[{kind:'text',body:'text'}]}]}};
 assert.deepEqual(editing.remove(current,[{path:'sections.article.0'},{path:'sections.article.0.blocks.0.body',part:0}]).sections.article,[]);
});
test('direct deletion supports default material fields and guide paragraphs while preserving structure',()=>{
 const current={node:{what:'A\n\nB',tasks:'Same',example:'Same'},lesson:{goal:'G',guide:{intro:'Intro',groups:[{methods:[{id:'m',example:'E\n\nF'}]}]}}};
 const next=editing.remove(current,[{path:'node.what',part:0},{path:'node.example'},{path:'lesson.guide.groups.0.methods.0.example',part:1}]);
 assert.equal(next.node.what,'B');assert.equal(next.node.tasks,'');assert.equal(next.node.example,'');
 assert.equal(next.lesson.guide.groups[0].methods[0].example,'E');assert.equal(next.lesson.guide.groups[0].methods[0].id,'m');
});
test('invalid, protected and stale paths never modify publication structure',()=>{
 const current={node:{id:'test',name:'Topic'},sections:{article:[{kind:'text',body:'A'}]}};
 for(const target of [{path:'node.id'},{path:'node.name'},{path:'__proto__.polluted'},{path:'sections.article.9'},
  {path:'sections.article.0.kind'},{path:'sections.article.0.body',part:4},{path:'sections.article.0.body',part:-1},
  {path:'sections.article.0.body',part:0.5},{path:'lesson.code'}])assert.throws(()=>editing.remove(current,[target]),editing.EditError);
 assert.throws(()=>editing.remove(current,[]),editing.EditError);
});
test('deleting a paragraph does not prune unrelated pre-existing empty blocks',()=>{
 const current={sections:{article:[{kind:'text',title:'Keep empty title',body:''},{kind:'text',body:'A\n\nB'}]}};
 const next=editing.remove(current,[{path:'sections.article.1.body',part:0}]);
 assert.equal(next.sections.article.length,2);assert.deepEqual(next.sections.article[0],current.sections.article[0]);
});
test('whitespace-equivalent examples cannot reappear from a hidden alias after deletion',()=>{
 const current={node:{tasks:'A  B C',example:'A\n\nB\n\nC'}};
 const next=editing.remove(current,[{path:'node.example',part:1}]);
 assert.equal(next.node.example,'A\n\nC');assert.equal(next.node.tasks,'A\n\nC');
 assert.equal(editing.remove(current,[{path:'node.example'}]).node.tasks,'');
});
test('guide source annotations resolve to editable publication values',()=>{
 const fs=require('node:fs'),vm=require('node:vm'),context={window:{}};
 vm.runInNewContext(fs.readFileSync(require.resolve('../lesson-data.js'),'utf8'),context);
 const lesson=context.window.ATLAS_LESSONS.topics.causal;
 const html=require('../lesson-guide.js').render(lesson.guide);
 const paths=[...html.matchAll(/data-edit-path="([^"]+)"/g)].map(m=>m[1]);
 assert.ok(paths.length>100);
 for(const path of paths){assert.ok(editing.allowed(path),path);assert.ok(editing.lookup({lesson},path)!==undefined,path);}
});
