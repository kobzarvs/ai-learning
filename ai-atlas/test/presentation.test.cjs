const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const Prism=require('../vendor/prism.js'),Art=require('../art.js'),Core=require('../core.js'),Workspace=require('../workspace-core.js');
const context={window:{Prism}};
vm.runInNewContext(fs.readFileSync(require.resolve('../ui.js'),'utf8'),context);
vm.runInNewContext(fs.readFileSync(require.resolve('../lesson-data.js'),'utf8'),context);
const UI=context.window.AtlasUI,lessons=context.window.ATLAS_LESSONS;
const plain=html=>html.replace(/<[^>]*>/g,'').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"').replace(/&#39;/g,"'").replace(/&amp;/g,'&');

test('language grammars distinguish real syntax and preserve the complete original source',()=>{
 const samples={
  python:['def square(x):\n    # комментарий\n    return x * 2 + len("<hello>")',['keyword','function','comment','operator','number','builtin','string']],
  rust:['fn main() { let n: i32 = 42; println!("Привет"); }',['keyword','function','number','macro','string']],
  bash:['# setup\npython -m pip install torch==2.14.1\nexport PATH="$HOME/bin:$PATH"',['comment','parameter','variable','function','string']],
  toml:['[package]\nname = "atlas"\nedition = "2024"\n[dependencies]\nburn = { version = "=0.22.1", features = ["train"] }',['table','property','string','punctuation']]
 };
 for(const [language,[source,tokens]] of Object.entries(samples)){
  const html=UI.highlight(source,language);assert.equal(plain(html),source);
  for(const token of tokens)assert.match(html,new RegExp('class="token [^"]*\\b'+token+'\\b'),language+': '+token);
 }
 const hostile='<img src=x onerror="alert(1)"> & `';
 for(const language of ['python','rust','bash','toml','missing']){
  const html=UI.highlight(hostile,language);assert.doesNotMatch(html,/<img/);assert.equal(plain(html),hostile);
 }
});

test('all published Python, Rust and Cargo examples survive highlighting without changing a character',()=>{
 assert.equal(lessons.meta.python,'3.14');
 for(const [id,lesson] of Object.entries(lessons.topics)){
  assert.equal(plain(UI.highlight(lesson.code,'python')),lesson.code,id);
  if(lesson.rust){assert.equal(plain(UI.highlight(lesson.rust.code,'rust')),lesson.rust.code,id);assert.equal(plain(UI.highlight(lesson.rust.cargo,'toml')),lesson.rust.cargo,id);}
 }
});

test('every topic has an explicit semantic illustration and all sprite viewports stay inside the artwork',()=>{
 const nodes=Core.indexTree(require('../../map-data.json').root);
 for(const id of nodes.keys())assert.ok(Art.byTopic[id],id);
 for(const [id,family] of Object.entries({text:'text',vision:'vision',multimodal:'video',audio:'audio',trees:'trees',gnn:'graphs',cnn:'vision',rl:'games'}))assert.equal(Art.familyFor(id).id,family);
 for(const id of nodes.keys()){
  const html=Art.render(id),rect=html.match(/viewBox="([\d ]+)"/)[1].split(' ').map(Number);
  assert.ok(rect[0]>=0&&rect[1]>=0&&rect[0]+rect[2]<=1536&&rect[1]+rect[3]<=1024);
  assert.match(html,/role="img" aria-label="[^"]+"/);assert.match(Art.render(id,{decorative:true}),/aria-hidden="true"/);
 }
 const png=fs.readFileSync(require.resolve('../art/topic-icons.png'));
 assert.equal(png.readUInt32BE(16),1536);assert.equal(png.readUInt32BE(20),1024);
});

function dialog(){
 const handlers={};return {open:true,attrs:{},closes:0,
  setAttribute(key,value){this.attrs[key]=value;},addEventListener(type,fn){(handlers[type]??=[]).push(fn);},
  getBoundingClientRect(){return {left:100,top:100,right:800,bottom:600};},close(){this.closes++;this.open=false;},
  fire(type,props={}){const event={target:this,button:0,clientX:20,clientY:20,preventDefault(){this.defaultPrevented=true;},...props};for(const fn of handlers[type]||[])fn(event);return event;}
 };
}
test('a deliberate backdrop click dismisses the dialog and invokes its cleanup callback',()=>{
 const d=dialog();let cleaned=false;UI.bindDismiss(d,{close(){cleaned=true;d.close();}});
 d.fire('pointerdown');const e=d.fire('click');assert.equal(d.closes,1);assert.equal(cleaned,true);assert.equal(e.defaultPrevented,true);
});
test('selecting content, dialog padding, cancelled gestures and right clicks cannot dismiss a dialog',()=>{
 for(const actions of [
  [['pointerdown',{target:{}}],['click',{}]],
  [['pointerdown',{clientX:120,clientY:120}],['click',{clientX:120,clientY:120}]],
  [['pointerdown',{}],['pointercancel',{}],['click',{}]],
  [['pointerdown',{button:2}],['click',{}]],
  [['pointerdown',{}],['click',{target:{}}]],
  [['click',{}]]
 ]){const d=dialog();UI.bindDismiss(d);for(const [type,props] of actions)d.fire(type,props);assert.equal(d.closes,0);}
});
test('editable dialogs never dismiss on a backdrop click',()=>{
 const d=dialog();UI.bindDismiss(d,{editable:true});d.fire('pointerdown');d.fire('click');assert.equal(d.closes,0);assert.equal(d.attrs.closedby,'none');
});

test('larger cards migrate existing notes and groups once, preserving their content and arrangement',()=>{
 const old={version:1,cards:[{id:'a',type:'note',ref:'',title:'Мои выводы',body:'Не потерять',x:80,y:90,w:360,h:230,groupId:'g'},{id:'b',type:'article',ref:'cnn',title:'CNN',body:'',x:480,y:90,w:360,h:230,groupId:'g'}],groups:[{id:'g',title:'Зрение',x:48,y:34,w:824,h:318}]};
 const next=Workspace.normalizeBoard(old,old),[a,b]=next.cards,g=next.groups[0];
 assert.equal(next.layoutVersion,2);assert.equal(a.body,old.cards[0].body);assert.equal(a.groupId,'g');assert.equal(g.title,'Зрение');
 assert.ok(a.w>=420&&a.h>=330);assert.ok(a.x+a.w<b.x);assert.equal(a.y,b.y);
 for(const c of next.cards){assert.ok(g.x<c.x&&g.y<c.y&&g.x+g.w>c.x+c.w&&g.y+g.h>c.y+c.h);}
 assert.deepEqual(Workspace.normalizeBoard(next,next),next);assert.deepEqual(Workspace.parseImport(JSON.stringify({version:1,boards:{cnn:next}})).boards.cnn,next);
});
