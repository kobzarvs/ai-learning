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
