const {test}=require('node:test'),assert=require('node:assert/strict');
const editing=require('../reader-edit.js');
const {selection,targetFor}=editing;
test('selection toggles disjoint paragraphs and whole sections without losing other choices',()=>{
 const state=selection(),a={path:'sections.article.0.body',part:0},b={path:'sections.article.0.body',part:2},c={path:'sections.article.2'};
 state.toggle([a]);state.toggle([b,c]);assert.deepEqual(state.targets,[a,b,c]);
 state.toggle([b]);assert.deepEqual(state.targets,[a,c]);state.toggle([a,c]);assert.deepEqual(state.targets,[]);
});
test('DOM references keep zero-based paragraph indices distinct from whole fields',()=>{
 assert.deepEqual(targetFor({dataset:{editPath:'node.what',editPart:'0'}}),{path:'node.what',part:0});
 assert.deepEqual(targetFor({dataset:{editPath:'node.what'}}),{path:'node.what'});
 assert.equal(targetFor({dataset:{}}),null);
});
test('dialog deletion resolves exactly the quoted paragraph or heading, never its whole section',()=>{
 const root={contains:()=>true};
 const section={dataset:{editSection:'sections.article.2'}};
 const paragraph={dataset:{editPath:'sections.article.2.body',editPart:'1'},textContent:'Selected paragraph',
  closest:s=>s==='[data-edit-path]'?paragraph:section,matches:()=>false,parentElement:section};
 assert.deepEqual(editing.fragmentTargets(paragraph,root),[{path:'sections.article.2.body',part:1}]);
 const heading={dataset:{},textContent:'Title',closest:s=>s==='[data-edit-path]'?null:section,matches:()=>true,parentElement:section};
 assert.deepEqual(editing.fragmentTargets(heading,root),[{path:'sections.article.2.title'}]);
 const wrapper={dataset:{editPath:'lesson.guide.foundations.0'},textContent:'Term and meaning'};
 const term={dataset:{},textContent:'Term',closest:s=>s==='[data-edit-path]'?wrapper:null,matches:()=>false};
 assert.deepEqual(editing.fragmentTargets(term,root),[],'a quoted term cannot delete its unquoted definition');
 assert.deepEqual(editing.fragmentTargets(paragraph,{contains:()=>false}),[],'detached content is not deletable');
});

function controllerSetup({fail=false}={}){
 const element=()=>({dataset:{},listeners:{},classList:{toggle(){}},
  addEventListener(type,callback){this.listeners[type]=callback;},querySelectorAll(){return [];}});
 const buttons=new Map(),bar=element();bar.querySelector=selector=>{
  if(!buttons.has(selector))buttons.set(selector,{});return buttons.get(selector);
 };
 const reader={...element(),open:true,insertBefore(){}},content=element();
 const doc={...element(),createElement:()=>bar,querySelector:()=>null};
 const calls=[],messages=[],context={topicId:'knn',tab:'article',revision:'8'};
 const controller=editing.create({reader,content,document:doc,getUser:()=>({isModerator:true}),ensureUser:async()=>({isModerator:true}),
  announce:message=>messages.push(message),api:async(url,options)=>{
   calls.push(JSON.parse(options.body));if(fail)throw Error('Network unavailable');
   return {editId:'saved-edit',revision:String(8+calls.length)};
  },refresh:async revision=>{context.revision=revision;controller.update({...context});}});
 controller.update({...context});
 return {bar,reader,doc,calls,messages,context,controller,finish:()=>bar.querySelector('[data-edit-save]').onclick(),remove:()=>controller.removeFragment([{path:'node.what',part:0}],{...context})};
}

test('Save closes the editing banner without another write and retains keyboard undo',async t=>{
 t.mock.timers.enable({apis:['setTimeout']});
 const env=controllerSetup();await env.remove();assert.equal(env.bar.hidden,false);
 t.mock.timers.tick(60000);assert.equal(env.bar.hidden,false);
 env.finish();assert.equal(env.bar.hidden,true);assert.equal(env.calls.length,1);
 env.controller.refresh();env.controller.update({...env.context});assert.equal(env.bar.hidden,true,'refresh cannot revive a saved banner');
 env.doc.listeners.keydown({key:'z',metaKey:true,target:{closest:()=>null},preventDefault(){},stopImmediatePropagation(){}});
 await new Promise(setImmediate);
 assert.equal(env.calls.length,2);assert.equal(env.calls[1].action,'undo');assert.equal(env.calls[1].editId,'saved-edit');
});

test('unconfirmed saves stay visible, and reopening a reader does not revive an old confirmation',async t=>{
 t.mock.timers.enable({apis:['setTimeout']});
 const failed=controllerSetup({fail:true});await failed.remove();t.mock.timers.tick(60000);
 assert.equal(failed.bar.hidden,false,'pending writes still need an explicit retry');
 failed.finish();assert.equal(failed.bar.hidden,false,'Save cannot hide an unconfirmed write');
 const saved=controllerSetup();await saved.remove();saved.reader.open=false;saved.reader.listeners.close();
 saved.reader.open=true;saved.controller.update({...saved.context});assert.equal(saved.bar.hidden,true);
});
