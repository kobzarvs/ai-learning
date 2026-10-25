const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const context={window:{}};
vm.runInNewContext(fs.readFileSync(require.resolve('../lesson-data.js'),'utf8'),context);
const lesson=context.window.ATLAS_LESSONS.topics.causal,guide=lesson.guide;
const G=require('../lesson-guide.js');

test('causal methods and case links resolve to substantive explanations and primary sources',()=>{
 const methods=guide.groups.flatMap(group=>group.methods),ids=new Set(methods.map(method=>method.id));
 assert.equal(ids.size,methods.length);
 for(const method of methods){
  for(const field of ['title','meaning','data','assumptions','example'])assert.ok(method[field]?.trim(),method.id+': '+field);
  assert.ok(method.sources.length,method.id);
  for(const id of method.sources)assert.ok(guide.sources[id],method.id+': missing source '+id);
 }
 for(const item of guide.cases){
  for(const id of item.methods)assert.ok(ids.has(id),item.title+': missing method '+id);
  for(const id of item.sources)assert.ok(guide.sources[id],item.title+': missing source '+id);
  assert.ok(item.kind&&item.question&&item.approach&&item.lesson);
 }
 for(const [,url] of Object.values(guide.sources))assert.equal(new URL(url).protocol,'https:');
 const html=G.render(guide),targets=new Set([...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]));
 for(const match of html.matchAll(/data-guide-jump="([^"]+)"/g))assert.ok(targets.has(match[1]),match[1]);
 assert.doesNotMatch(html,/undefined|NaN|runnable/);
});

test('causal article, Python and charts use the employee task story and distinguish simulation from actual research',()=>{
 const data=require('../../map-data.json'),C=require('../core.js'),node=C.indexTree(data.root).get('causal');
 const combined=JSON.stringify(node)+JSON.stringify(lesson);
 assert.doesNotMatch(combined,/морожен|купан|ice_cream|swimming|weather/);
 assert.match(lesson.display.body,/minutes_saved = 4.0/);
 assert.match(lesson.limits,/заранее задан/);
 assert.match(guide.cases.find(item=>item.methods.includes('rootcause')).kind,/данные моделируются/);
 assert.match(guide.cases.find(item=>item.methods.includes('longitudinal')).kind,/не отчёт/);
 assert.equal(C.indexTree(data.root).get('classification').name,'Классификация и регрессия');
});

// Exercise navigation between hidden chapters, including direct links from cases.
function guideDOM(){
 const keys=[...guide.groups.map((_,i)=>String(i)),'cases','basics'];
 const panels=keys.map(key=>({dataset:{guidePanel:key},hidden:key!=='0',getBoundingClientRect:()=>({top:160-container.scrollTop})}));
 const buttons=keys.map(key=>({dataset:{guideView:key},attrs:{},setAttribute(k,v){this.attrs[k]=v;}}));
 const methods=guide.groups.flatMap((group,i)=>group.methods.map(method=>({
  id:'causal-method-'+method.id,open:false,
  closest:()=>panels[i],
  querySelector(){return{focus:options=>{this.focusOptions=options;}};},
  scrollIntoView(options){this.scrollOptions=options;container.scrollTop=600;}
 })));
 const root={dataset:{guideCurrent:'0'},querySelectorAll:selector=>selector==='[data-guide-panel]'?panels:selector==='details'?methods:buttons};
 const container={scrollTop:230,clientTop:0,getBoundingClientRect:()=>({top:40}),querySelector:()=>root,querySelectorAll:()=>methods};
 return{root,container,panels,buttons,methods};
}

test('situation navigation starts new chapters at their heading and restores each reading position',()=>{
 const dom=guideDOM();dom.methods[0].open=true;
 const keys=[...guide.groups.map((_,i)=>String(i)),'cases','basics'];
 for(const [i,key] of keys.entries()){
  assert.equal(G.selectView(dom.container,key),true);
  assert.deepEqual(dom.panels.filter(p=>!p.hidden).map(p=>p.dataset.guidePanel),[key]);
  assert.deepEqual(dom.buttons.filter(b=>b.attrs['aria-pressed']==='true').map(b=>b.dataset.guideView),[key]);
  assert.equal(dom.root.dataset.guideCurrent,key);
  assert.equal(dom.container.scrollTop,i===0?230:120);
  dom.container.scrollTop=300+i*70;
  assert.equal(dom.methods[0].open,true);
 }
 assert.equal(G.selectView(dom.container,'missing'),false);
 assert.equal(dom.root.dataset.guideCurrent,'basics');
 assert.equal(dom.container.scrollTop,930);
 for(const [i,key] of keys.entries()){
  G.selectView(dom.container,key);
  assert.equal(dom.container.scrollTop,300+i*70);
 }
 dom.container.scrollTop=800;
 G.selectView(dom.container,'basics');
 assert.equal(dom.container.scrollTop,800,'clicking the active chapter must not move it');
});

test('guide reading positions survive a rerender without leaking to another guide',()=>{
 const dom=guideDOM();dom.methods[0].open=true;
 G.selectView(dom.container,'2');dom.container.scrollTop=740;
 const snapshot=G.capture(dom.container);
 const rerendered=guideDOM();rerendered.container.scrollTop=0;
 assert.equal(G.restore(rerendered.container,snapshot),true);
 assert.equal(rerendered.root.dataset.guideCurrent,'2');
 assert.equal(rerendered.container.scrollTop,740);
 assert.equal(rerendered.methods[0].open,true);
 G.selectView(rerendered.container,'0');assert.equal(rerendered.container.scrollTop,230);
 G.selectView(rerendered.container,'3');assert.equal(rerendered.container.scrollTop,120);
 const independent=guideDOM();G.selectView(independent.container,'2');
 assert.equal(independent.container.scrollTop,120);
});

test('every case-to-method link reveals its chapter and focuses the expanded method',()=>{
 const dom=guideDOM();
 for(const item of guide.cases)for(const id of item.methods){
  G.selectView(dom.container,'cases');
  assert.equal(G.jump(dom.container,'causal-method-'+id),true);
  const method=dom.methods.find(m=>m.id==='causal-method-'+id);
  assert.equal(method.open,true);
  assert.equal(method.closest().hidden,false);
  assert.deepEqual(method.focusOptions,{preventScroll:true});
  assert.deepEqual(method.scrollOptions,{block:'start'});
 }
 assert.equal(G.jump(dom.container,'missing'),false);
});

test('jumping to a method saves the cases position and replaces the destination reading position',()=>{
 const dom=guideDOM();G.selectView(dom.container,'cases');dom.container.scrollTop=520;
 G.jump(dom.container,'causal-method-iv');assert.equal(dom.container.scrollTop,600);
 G.selectView(dom.container,'cases');assert.equal(dom.container.scrollTop,520);
 G.selectView(dom.container,'2');assert.equal(dom.container.scrollTop,600);
});

test('redesign retains every method, keeps one initial chapter and escapes editable content',()=>{
 const html=G.render(guide);
 const panels=[...html.matchAll(/<section class="guide-panel"[^>]+>/g)].map(m=>m[0]);
 assert.equal(panels.length,guide.groups.length+2);
 assert.equal(panels.filter(tag=>!tag.includes(' hidden')).length,1);
 for(const group of guide.groups)for(const method of group.methods){
  assert.equal(html.split('id="causal-method-'+method.id+'"').length-1,1);
 }
 const altered=structuredClone(guide);
 altered.groups[0].situation='<img src=x onerror=alert(1)>';
 altered.groups[0].methods[0].example='<script>bad()</script>';
 altered.sources.book=['unsafe','javascript:alert(1)'];
 const safe=G.render(altered);
 assert.doesNotMatch(safe,/<script>|<img|href="javascript:/);
 assert.match(safe,/&lt;script&gt;/);
});
