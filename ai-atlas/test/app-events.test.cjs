// Exercise the real app handlers with a small DOM boundary double. Geometry and
// rendering are tested separately; this covers event ordering, not browser layout.
const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const C=require('../core.js'),data=require('../../map-data.json');
function setup(){
 class Element{
  constructor(id,tag='DIV',parent=null){this.id=id;this.tagName=tag;this.parentElement=parent;this.listeners={};this.dataset={};this.attrs={};this.hidden=false;this.innerHTML='';this.clientWidth=1000;this.clientHeight=600;this.style={setProperty(){}};this.classes=new Set();this.captured=new Set();this.classList={add:c=>this.classes.add(c),remove:c=>this.classes.delete(c),contains:c=>this.classes.has(c),toggle:(c,yes)=>yes?this.classes.add(c):this.classes.delete(c)};}
  addEventListener(type,fn,options){(this.listeners[type]??=[]).push({fn,capture:options===true||options?.capture});}
  setAttribute(k,v){this.attrs[k]=v;}removeAttribute(k){delete this.attrs[k];}
  getAttribute(k){return this.attrs[k]??null;}
  matches(s){return s.split(',').some(t=>{t=t.trim();return t.startsWith('#')?this.id===t.slice(1):t.startsWith('.')?this.classes.has(t.slice(1)):this.tagName.toLowerCase()===t;});}
  closest(s){return this.matches(s)?this:this.parentElement?.closest(s)||null;}
  querySelectorAll(){return[];}querySelector(){return null;}
  getBoundingClientRect(){return{left:0,top:0,right:1000,bottom:600,width:1000,height:600};}
  setPointerCapture(id){this.captured.add(id);}releasePointerCapture(id){this.captured.delete(id);}hasPointerCapture(id){return this.captured.has(id);}
  focus(){}scrollTop=0;
 }
 const doc=new Element('document'),win=new Element('window'),body=new Element('body');doc.body=body;
 const html=fs.readFileSync(require.resolve('../index.html'),'utf8'),elements=new Map();
 for(const m of html.matchAll(/id="([^"]+)"/g))elements.set(m[1],new Element(m[1]));
 const canvas=elements.get('canvas');canvas.attrs.tabindex='0';doc.activeElement=canvas;
 doc.getElementById=id=>elements.get(id)||null;
 const button=new Element('fake-card-button','BUTTON',canvas);button.dataset.open='cnn';
 const term=new Element('fake-term','ABBR',button);term.dataset.term='CNN';term.classes.add('term');
 const location={hash:''},history={replaceState(_a,_b,hash){location.hash=hash;}};
 const context={window:win,document:doc,location,history,localStorage:{getItem(){return null;}},setTimeout,clearTimeout,requestAnimationFrame:fn=>fn(),ResizeObserver:class{observe(){}},console};
 win.AtlasCore=C;win.ATLAS_DATA=data;win.innerWidth=1440;win.innerHeight=900;
 vm.runInNewContext(fs.readFileSync(require.resolve('../app.js'),'utf8'),context);
 function fire(type,target,props={}){
  const e={type,target,button:0,pointerId:1,clientX:100,clientY:100,detail:1,key:'',code:'',shiftKey:false,preventDefault(){this.defaultPrevented=true;},stopPropagation(){this.stopped=true;},stopImmediatePropagation(){this.stopped=true;this.immediate=true;},...props};
  const inCanvas=!!target.closest('#canvas');
  const call=(node,capture)=>{for(const l of node.listeners[type]||[])if(!!l.capture===capture){l.fn(e);if(e.immediate)break;}};
  call(doc,true);if(!e.stopped&&inCanvas)call(canvas,true);if(!e.stopped&&inCanvas)call(canvas,false);if(!e.stopped)call(doc,false);
  return e;
 }
 return {elements,canvas,doc,win,button,term,fire,location};
}
test('actual handlers pan over a card, survive Shift-up and suppress its click, then open normally',()=>{
 const {elements,canvas,button,fire,location}=setup();
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
test('Shift + wheel up and down pans vertically without zoom; releasing Shift restores zoom',()=>{
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
 assert.notEqual(readView(elements).scale,before.scale,'ordinary wheel zoom returns after Shift-up');
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
