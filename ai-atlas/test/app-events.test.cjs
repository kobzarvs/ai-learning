// Exercise the real app handlers with a small DOM boundary double. Geometry and
// rendering are tested separately; this covers event ordering, not browser layout.
const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const C=require('../core.js'),data=require('../../map-data.json');
function setup({spaces=false}={}){
 class Element{
  constructor(id,tag='DIV',parent=null){this.id=id;this.tagName=tag;this.parentElement=parent;this.listeners={};this.dataset={};this.attrs={};this.hidden=false;this.innerHTML='';this.clientWidth=1000;this.clientHeight=600;this.style={setProperty(){}};this.classes=new Set();this.captured=new Set();this.classList={add:c=>this.classes.add(c),remove:c=>this.classes.delete(c),contains:c=>this.classes.has(c),toggle:(c,yes)=>yes?this.classes.add(c):this.classes.delete(c)};}
  addEventListener(type,fn,options){(this.listeners[type]??=[]).push({fn,capture:options===true||options?.capture});}
  setAttribute(k,v){this.attrs[k]=v;}removeAttribute(k){delete this.attrs[k];}
  getAttribute(k){return this.attrs[k]??null;}
  matches(s){return s.split(',').some(t=>{t=t.trim();if(t.startsWith('[data-')){const key=t.slice(6,-1).replace(/-([a-z])/g,(_,c)=>c.toUpperCase());return key in this.dataset;}return t.startsWith('#')?this.id===t.slice(1):t.startsWith('.')?this.classes.has(t.slice(1)):this.tagName.toLowerCase()===t;});}
  closest(s){return this.matches(s)?this:this.parentElement?.closest(s)||null;}
  querySelectorAll(){return[];}querySelector(){return null;}
  getBoundingClientRect(){return{left:0,top:0,right:1000,bottom:600,width:1000,height:600};}
  setPointerCapture(id){this.captured.add(id);}releasePointerCapture(id){this.captured.delete(id);}hasPointerCapture(id){return this.captured.has(id);}
  focus(){}showModal(){this.open=true;}close(){this.open=false;}scrollTop=0;
 }
 const doc=new Element('document'),win=new Element('window'),body=new Element('body');doc.body=body;
 const html=fs.readFileSync(require.resolve('../index.html'),'utf8'),elements=new Map();
 for(const m of html.matchAll(/id="([^"]+)"/g))elements.set(m[1],new Element(m[1]));
 elements.set('selection-rect',new Element('selection-rect'));
 const canvas=elements.get('canvas');canvas.attrs.tabindex='0';doc.activeElement=canvas;
 doc.getElementById=id=>elements.get(id)||null;
 const button=new Element('fake-card-button','BUTTON',canvas);button.dataset.open='cnn';
 const term=new Element('fake-term','ABBR',button);term.dataset.term='CNN';term.classes.add('term');
 const location={hash:''},history={replaceState(_a,_b,hash){location.hash=hash;}};
 const storage=new Map();
 const context={structuredClone,crypto:require('node:crypto').webcrypto,window:win,document:doc,location,history,localStorage:{getItem(k){return storage.get(k)||null;},setItem(k,v){storage.set(k,v);}},setTimeout,clearTimeout,requestAnimationFrame:fn=>fn(),ResizeObserver:class{observe(){}},console};
 win.AtlasCore=C;win.ATLAS_DATA=data;win.innerWidth=1440;win.innerHeight=900;
 if(spaces){win.AtlasWorkspaceCore=require('../workspace-core.js');for(const file of ['lesson-data.js','vendor/prism.js','ui.js','art.js','launch.js','workspace.js'])vm.runInNewContext(fs.readFileSync(require.resolve('../'+file),'utf8'),context);}
 vm.runInNewContext(fs.readFileSync(require.resolve('../app.js'),'utf8'),context);
 function fire(type,target,props={}){
  const e={type,target,button:0,pointerId:1,clientX:100,clientY:100,detail:1,key:'',code:'',shiftKey:false,preventDefault(){this.defaultPrevented=true;},stopPropagation(){this.stopped=true;},stopImmediatePropagation(){this.stopped=true;this.immediate=true;},...props};
  const inCanvas=!!target.closest('#canvas');
  const call=(node,capture)=>{for(const l of node.listeners[type]||[])if(!!l.capture===capture){l.fn(e);if(e.immediate)break;}};
  call(doc,true);if(!e.stopped&&inCanvas)call(canvas,true);if(!e.stopped&&inCanvas)call(canvas,false);if(!e.stopped&&!inCanvas)call(target,false);if(!e.stopped)call(doc,false);
  return e;
 }
 return {elements,canvas,doc,win,button,term,fire,location,storage,make:(id,tag='BUTTON',parent=canvas,dataset={})=>{const e=new Element(id,tag,parent);e.dataset=dataset;return e;}};
}
test('actual handlers pan over a card, survive Shift-up and suppress its click, then open normally',()=>{
 const {elements,canvas,button,fire,location}=setup();
 elements.get('select-tool').onclick();
 const before=elements.get('world').style.transform;
 fire('keydown',canvas,{key:'Shift',code:'ShiftLeft'});
 assert.equal(elements.get('select-tool').attrs['aria-pressed'],'true');
 fire('pointerdown',button,{shiftKey:true});
 fire('pointermove',canvas,{clientX:180,clientY:130,shiftKey:true});
 assert.notEqual(elements.get('world').style.transform,before);
 fire('keyup',canvas,{key:'Shift',code:'ShiftLeft'});
 assert.equal(canvas.classes.has('dragging'),true);
 fire('pointermove',canvas,{clientX:220,clientY:160});
 fire('pointerup',canvas);const click=fire('click',button);
 assert.equal(click.defaultPrevented,true);assert.equal(location.hash,'');
 assert.equal(canvas.classes.has('pan-mode'),false);
 fire('pointerdown',button);fire('pointerup',button);fire('click',button);
 assert.equal(location.hash,'#cnn');assert.match(elements.get('detail-body').innerHTML,/КАК ЭТО РАБОТАЕТ/);
});
test('actual handlers preserve permanent Pan and ignore Shift in text input',()=>{
 const {elements,canvas,doc,win,button,fire}=setup();
 elements.get('pan-tool').onclick();fire('keydown',canvas,{key:'Shift',code:'ShiftLeft'});fire('keyup',canvas,{key:'Shift',code:'ShiftLeft'});
 assert.equal(elements.get('pan-tool').attrs['aria-pressed'],'true');assert.equal(canvas.classes.has('pan-mode'),true);
 elements.get('select-tool').onclick();elements.get('search').tagName='INPUT';
 fire('keydown',elements.get('search'),{key:'Shift',code:'ShiftLeft'});assert.equal(canvas.classes.has('pan-mode'),false);
 fire('keydown',canvas,{key:'Shift',code:'ShiftLeft'});fire('pointerdown',button,{shiftKey:true});
 for(const {fn} of win.listeners.blur)fn();assert.equal(canvas.classes.has('dragging'),false);assert.equal(canvas.classes.has('pan-mode'),false);
});
test('rendered cards keep plain attribute labels and accessible tooltip handlers explain terms',()=>{
 const {elements,term,canvas,fire}=setup();
 elements.get('select-tool').onclick();
 const markup=elements.get('nodes').innerHTML;
 assert.doesNotMatch(markup,/aria-label="[^"\n]*<abbr/);
 assert.match(markup,/aria-label="CNN — свёрточная нейросеть: открыть описание"/);
 fire('pointerover',term);assert.equal(elements.get('term-tooltip').hidden,false);
 assert.match(elements.get('term-tooltip').innerHTML,/Convolutional Neural Network/);
 assert.equal(term.attrs['aria-describedby'],'term-tooltip');
 fire('keydown',canvas,{key:'Escape'});assert.equal(elements.get('term-tooltip').hidden,true);
});
function readView(elements){
 const m=elements.get('world').style.transform.match(/translate\(([-\d.e]+)px,\s*([-\d.e]+)px\) scale\(([-\d.e]+)\)/);
 assert.ok(m,'canvas transform is valid');return{x:Number(m[1]),y:Number(m[2]),scale:Number(m[3])};
}
test('scrolling pans with or without Shift in both tools',()=>{
 const {elements,canvas,fire}=setup();
 fire('keydown',canvas,{key:'Shift',code:'ShiftLeft'});
 for(const deltaY of [-120,120]){
  const before=readView(elements);
  const event=fire('wheel',canvas,{deltaX:0,deltaY,deltaMode:0,shiftKey:true});
  const after=readView(elements);
  assert.equal(after.scale,before.scale,'Shift scrolling must never zoom');
  assert.equal(after.x,before.x);assert.equal(after.y,before.y-deltaY);
  assert.equal(event.defaultPrevented,true);
 }
 fire('keyup',canvas,{key:'Shift',code:'ShiftLeft'});
 const before=readView(elements);fire('wheel',canvas,{deltaX:0,deltaY:120,deltaMode:0});
 assert.equal(readView(elements).scale,before.scale);assert.equal(readView(elements).y,before.y-120);
 elements.get('select-tool').onclick();
 const selected=readView(elements);fire('wheel',canvas,{deltaX:20,deltaY:-60,deltaMode:0});
 assert.deepEqual(readView(elements),{...selected,x:selected.x-20,y:selected.y+60});
});
test('wheel pans in the selected Pan tool and respects horizontal and vertical deltas',()=>{
 const {elements,canvas,fire}=setup();elements.get('pan-tool').onclick();
 const before=readView(elements);fire('wheel',canvas,{deltaX:15,deltaY:80,deltaMode:0});
 const after=readView(elements);assert.equal(after.scale,before.scale);assert.equal(after.x,before.x-15);assert.equal(after.y,before.y-80);
});
test('Shift wheel flags work without keydown and line-based wheels preserve scale',()=>{
 const {elements,canvas,fire}=setup();const before=readView(elements);
 fire('wheel',canvas,{deltaX:0,deltaY:3,deltaMode:1,shiftKey:true});
 const after=readView(elements);assert.equal(after.scale,before.scale);assert.equal(after.y,before.y-48);
});

// These cases execute workspace.js and app.js together, including capture order.
test('topic opens a separate canvas; a lesson opens the full reader and Back restores the camera',()=>{
 const s=setup({spaces:true}),before=s.elements.get('world').style.transform;
 s.fire('click',s.button);
 assert.equal(s.elements.get('world').hidden,true);
 assert.equal(s.elements.get('workspace-world').hidden,false);
 assert.match(s.elements.get('workspace-world').innerHTML,/Python/);
 const card=s.make('article','BUTTON',s.canvas,{read:'article:cnn'});
 s.fire('click',card);
 assert.equal(s.elements.get('reader').open,true);
 assert.match(s.elements.get('reader-content').innerHTML,/Как это устроено/);
 const tab=s.make('python','BUTTON',null,{readerTab:'python'});s.fire('click',tab);
 assert.match(s.elements.get('reader-content').innerHTML,/Conv2d/);
 assert.match(s.elements.get('reader-content').innerHTML,/token keyword/);
 assert.match(s.elements.get('reader-content').innerHTML,/task-brief/);
 for(const block of s.elements.get('reader-content').innerHTML.matchAll(/<code[^>]*>([\s\S]*?)<\/code>/g))assert.doesNotMatch(block[1],/<abbr/);
 s.elements.get('reader-close').onclick();
 s.elements.get('space-back').onclick();
 assert.equal(s.elements.get('world').hidden,false);
 assert.equal(s.elements.get('world').style.transform,before);
});
test('exercise sections have distinct icons and the task is a separate callout',()=>{
 const s=setup({spaces:true});s.fire('click',s.button);
 s.fire('click',s.make('exercise','BUTTON',s.canvas,{read:'exercise:cnn'}));
 const html=s.elements.get('reader-content').innerHTML;
 assert.match(html,/<section class="reading-section task-brief"><h2><span class="section-icon icon-target"/);
 for(const kind of ['steps','experiment','result','info','sources'])assert.match(html,new RegExp('section-icon icon-'+kind));
 assert.equal((html.match(/<h2>/g)||[]).length,(html.match(/section-icon/g)||[]).length);
});
test('every topic article shows shared application/example text once and preserves distinct content',()=>{
 const s=setup({spaces:true});
 const plain=html=>html.replace(/<[^>]*>/g,'').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"').replace(/&#39;/g,"'").replace(/&amp;/g,'&').replace(/\s+/g,' ').trim();
 for(const n of C.indexTree(data.root).values()){
  if(n.children)continue;
  s.fire('click',s.make('topic-'+n.id,'BUTTON',s.canvas,{spaceEnter:n.id}),{detail:0});
  s.fire('click',s.make('article-'+n.id,'BUTTON',s.canvas,{read:'article:'+n.id}),{detail:0});
  const html=s.elements.get('reader-content').innerHTML;
  const paragraphs=[...html.matchAll(/<p(?:\s[^>]*)?>([\s\S]*?)<\/p>/g)].map(m=>plain(m[1]));
  for(const text of new Set([n.tasks,n.example].filter(Boolean)))assert.equal(paragraphs.filter(p=>p===text.replace(/\s+/g,' ').trim()).length,1,n.id+': duplicated or missing application/example');
 }
});
test('the rendered Rust guide places its source before cargo run and switches to an existing container example',()=>{
 const s=setup({spaces:true});s.fire('click',s.button);
 s.fire('click',s.make('rust','BUTTON',s.canvas,{read:'rust:cnn'}));
 let html=s.elements.get('reader-content').innerHTML;
 assert.ok(html.indexOf('data-copy-code="init"')<html.indexOf('data-copy-code="cargo"'));
 assert.ok(html.indexOf('data-copy-code="rust"')<html.indexOf('data-copy-code="run"'));
 assert.match(html,/src\/main.rs/);
 s.fire('click',s.make('container','BUTTON',null,{launchMode:'container'}));html=s.elements.get('reader-content').innerHTML;
 assert.match(html,/vscode.dev\/redirect/);assert.match(html,/Терминал Dev Container/);assert.match(html,/examples\/rust-burn\/Cargo.toml/);
 assert.doesNotMatch(html,/data-copy-code="init"/);
 s.fire('click',s.make('local','BUTTON',null,{launchMode:'local'}));
 assert.match(s.elements.get('reader-content').innerHTML,/data-copy-code="init"/);
});
test('real readers close on backdrop clicks while unsaved note input survives backdrop and Escape',()=>{
 const s=setup({spaces:true});s.fire('click',s.button);
 s.fire('click',s.make('article','BUTTON',s.canvas,{read:'article:cnn'}));
 const reader=s.elements.get('reader');s.fire('pointerdown',reader,{clientX:-10});s.fire('click',reader,{clientX:-10});assert.equal(reader.open,false);
 s.elements.get('space-note').onclick();const editor=s.elements.get('note-editor');s.elements.get('editor-title').value='Не потерять';
 s.fire('pointerdown',editor,{clientX:-10});s.fire('click',editor,{clientX:-10});assert.equal(editor.open,true);
 assert.equal(s.fire('cancel',editor).defaultPrevented,true);assert.equal(s.elements.get('editor-title').value,'Не потерять');
});
test('Shift wheel and Shift release while dragging preserve Pan inside topic spaces',()=>{
 const s=setup({spaces:true});s.fire('click',s.button);
 const handle=s.make('handle','BUTTON',s.canvas,{dragCard:'article:cnn'});
 const before=readView(s.elements);
 s.fire('wheel',s.canvas,{deltaX:0,deltaY:-120,deltaMode:0,shiftKey:true});
 const after=readView(s.elements);assert.equal(before.scale,after.scale);assert.equal(after.y,before.y+120);
 s.fire('pointerdown',handle,{shiftKey:true});s.fire('pointermove',s.canvas,{clientX:180});
 s.fire('keyup',s.canvas,{key:'Shift',code:'ShiftLeft'});s.fire('pointerup',s.canvas);
 assert.equal(s.fire('click',handle).defaultPrevented,true);
 const state=JSON.parse(s.storage.get('ai-atlas-workspaces-v1'));
 assert.equal(state.boards.cnn.cards[0].x,80,'pan must not move a card');
});
test('dragging a card moves only the card and saves its new coordinates',()=>{
 const s=setup({spaces:true});s.fire('click',s.button);s.elements.get('select-tool').onclick();
 const handle=s.make('handle','BUTTON',s.canvas,{dragCard:'article:cnn'}),view=readView(s.elements);
 s.fire('pointerdown',handle);s.fire('pointermove',s.canvas,{clientX:220,clientY:160});
 s.fire('keydown',s.canvas,{key:'Shift',code:'ShiftLeft'});
 s.fire('pointermove',s.canvas,{clientX:240,clientY:180});s.fire('pointerup',s.canvas);
 assert.deepEqual(readView(s.elements),view,'pressing Shift mid-card-drag cannot start panning');
 const state=JSON.parse(s.storage.get('ai-atlas-workspaces-v1'));
 assert.equal(state.boards.cnn.cards[0].x,220);assert.equal(state.boards.cnn.cards[0].y,170);
 assert.equal(s.fire('click',handle).defaultPrevented,true);
});
test('note creation, group creation and undo persist in the browser store',()=>{
 const s=setup({spaces:true});s.fire('click',s.button);
 s.elements.get('space-note').onclick();s.elements.get('editor-title').value='Мой опыт';s.elements.get('editor-body').value='<script>alert(1)</script>';
 s.fire('submit',s.elements.get('editor-form'));
 let state=JSON.parse(s.storage.get('ai-atlas-workspaces-v1'));
 const note=state.boards.cnn.cards.find(c=>c.type==='note');assert.ok(note);
 assert.match(s.elements.get('workspace-world').innerHTML,/&lt;script&gt;/);
 s.elements.get('space-group').onclick();s.elements.get('editor-title').value='Мои идеи';s.fire('submit',s.elements.get('editor-form'));
 state=JSON.parse(s.storage.get('ai-atlas-workspaces-v1'));assert.equal(state.boards.cnn.groups[0].title,'Мои идеи');
 s.elements.get('space-undo').onclick();state=JSON.parse(s.storage.get('ai-atlas-workspaces-v1'));
 assert.equal(state.boards.cnn.groups.length,0);assert.ok(state.boards.cnn.cards.some(c=>c.id===note.id));
});
test('selection frame selects intersecting cards and grouping preserves their positions',()=>{
 const s=setup({spaces:true});s.fire('click',s.button);s.elements.get('space-lasso').onclick();
 const v=readView(s.elements);
 s.fire('pointerdown',s.canvas,{clientX:v.x+70,clientY:v.y+80});
 s.fire('pointermove',s.canvas,{clientX:v.x+450,clientY:v.y+330});s.fire('pointerup',s.canvas);
 assert.match(s.elements.get('space-selection').textContent,/1/);
 s.elements.get('space-group').onclick();s.elements.get('editor-title').value='Первый блок';s.fire('submit',s.elements.get('editor-form'));
 const b=JSON.parse(s.storage.get('ai-atlas-workspaces-v1')).boards.cnn;
 assert.equal(b.groups.length,1);assert.equal(b.cards[0].groupId,b.groups[0].id);assert.equal(b.cards[0].x,80);
});

test('Pan is active on load and dragging over a card moves the canvas',()=>{
 const {elements,canvas,button,fire,location}=setup();
 assert.equal(elements.get('pan-tool').attrs['aria-pressed'],'true');
 assert.equal(elements.get('select-tool').attrs['aria-pressed'],'false');
 assert.equal(canvas.classes.has('pan-mode'),true);
 const before=readView(elements);fire('pointerdown',button);fire('pointermove',canvas,{clientX:170,clientY:150});fire('pointerup',canvas);
 assert.deepEqual(readView(elements),{...before,x:before.x+70,y:before.y+50});
 assert.equal(fire('click',button).defaultPrevented,true);assert.equal(location.hash,'');
});
test('trackpad pinch zooms around the pointer in Pan and Select, including topic spaces',()=>{
 for(const spaces of [false,true])for(const tool of ['pan','select']){
  const s=setup({spaces});if(spaces)s.fire('click',s.button);s.elements.get(tool+'-tool').onclick();
  for(const deltaY of [-20,20]){
   const before=readView(s.elements),x=320,y=240;
   const event=s.fire('wheel',s.canvas,{ctrlKey:true,deltaX:0,deltaY,deltaMode:0,clientX:x,clientY:y});
   const after=readView(s.elements);
   assert.ok(deltaY<0?after.scale>before.scale:after.scale<before.scale);
   assert.ok(Math.abs((x-before.x)/before.scale-(x-after.x)/after.scale)<1e-8);
   assert.ok(Math.abs((y-before.y)/before.scale-(y-after.y)/after.scale)<1e-8);
   assert.equal(event.defaultPrevented,true);
  }
 }
});
test('two touch pointers pinch and pan together, then one finger continues without a jump',()=>{
 const {elements,canvas,fire}=setup(),before=readView(elements);
 fire('pointerdown',canvas,{pointerId:1,clientX:100,clientY:100,pointerType:'touch'});
 fire('pointerdown',canvas,{pointerId:2,clientX:200,clientY:100,pointerType:'touch'});
 fire('pointermove',canvas,{pointerId:2,clientX:250,clientY:100,pointerType:'touch'});
 const pinched=readView(elements);assert.ok(Math.abs(pinched.scale-before.scale*1.5)<1e-8);
 fire('pointerup',canvas,{pointerId:2});
 fire('pointermove',canvas,{pointerId:1,clientX:120,clientY:130,pointerType:'touch'});
 assert.deepEqual(readView(elements),{...pinched,x:pinched.x+20,y:pinched.y+30});
 fire('pointerup',canvas,{pointerId:1});assert.equal(canvas.classes.has('dragging'),false);
});
