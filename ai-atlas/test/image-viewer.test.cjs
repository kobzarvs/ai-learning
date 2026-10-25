const {test}=require('node:test'),assert=require('node:assert/strict');
const Viewer=require('../image-viewer.js');
const svg=source=>'data:image/svg+xml;charset=utf-8,'+encodeURIComponent(source);

test('fit reduces large images proportionally and never enlarges a small raster',()=>{
 assert.equal(typeof Viewer.fitSize,'function','the viewer must calculate an undistorted fit');
 assert.deepEqual(Viewer.fitSize({width:2000,height:1000},{width:900,height:600}),{width:900,height:450});
 assert.deepEqual(Viewer.fitSize({width:1000,height:2000},{width:900,height:600}),{width:300,height:600});
 assert.deepEqual(Viewer.fitSize({width:200,height:100},{width:900,height:600}),{width:200,height:100});
 assert.equal(Viewer.fitSize({width:0,height:100},{width:900,height:600}),null);
});

test('SVG original size uses its logical viewBox instead of browser fallback dimensions',()=>{
 assert.equal(typeof Viewer.originalSize,'function');
 const image={src:svg('<svg xmlns="http://www.w3.org/2000/svg" viewBox="-10, 20, 1200, 800"></svg>'),naturalWidth:300,naturalHeight:200};
 assert.deepEqual(Viewer.originalSize(image),{width:1200,height:800});
 assert.deepEqual(Viewer.originalSize({...image,src:svg('<svg width="640px" height="480"/>')}),{width:640,height:480});
 assert.deepEqual(Viewer.originalSize({...image,src:svg('<svg viewBox="0 0 0 800"/>')}),{width:300,height:200});
 assert.deepEqual(Viewer.originalSize({...image,src:'data:image/svg+xml,%BAD'}),{width:300,height:200});
 assert.deepEqual(Viewer.originalSize({src:'https://atlas.test/photo.webp',naturalWidth:2400,naturalHeight:1800}),{width:2400,height:1800});
 assert.equal(Viewer.originalSize({src:'photo.webp',naturalWidth:0,naturalHeight:0}),null);
});

// Browser boundary double: native dialog layout/focus trapping is verified in
// the browser; these tests exercise our actual delegated handlers and cleanup.
function setup(){
 class Element{
  constructor(tag='div'){this.tagName=tag.toUpperCase();this.attrs={};this.listeners={};this.children=[];this.style={};this.dataset={};this.scrollTop=0;this.scrollLeft=0;this.clientWidth=960;this.clientHeight=640;this.open=false;this.className='';}
  addEventListener(type,fn,options){(this.listeners[type]??=[]).push({fn,capture:options===true||options?.capture});}
  setAttribute(key,value){this.attrs[key]=String(value);if(key==='id')this.id=String(value);}
  getAttribute(key){return this.attrs[key]??null;}
  hasAttribute(key){return key in this.attrs;}
  removeAttribute(key){delete this.attrs[key];}
  append(...nodes){for(const node of nodes){this.children.push(node);node.parentElement=this;}}
  matches(selector){return selector.split(',').some(part=>{const s=part.trim();if(s==='.content-illustration img')return this.tagName==='IMG'&&!!this.parentElement?.closest('.content-illustration');if(s[0]==='#')return this.id===s.slice(1);if(s[0]==='.')return this.className.split(' ').includes(s.slice(1));return this.tagName.toLowerCase()===s;});}
  closest(selector){return this.matches(selector)?this:this.parentElement?.closest(selector)||null;}
  querySelectorAll(selector){return this.children.flatMap(child=>[...(child.matches(selector)?[child]:[]),...child.querySelectorAll(selector)]);}
  querySelector(selector){return this.querySelectorAll(selector)[0]||null;}
  get isConnected(){return this===doc||!!this.parentElement?.isConnected;}
  focus(options){this.focusOptions=options;doc.activeElement=this;}
  showModal(){this.open=true;}
  close(){this.open=false;fire('close',this);}
  getBoundingClientRect(){return {left:20,top:20,right:980,bottom:780,width:960,height:760};}
 }
 const doc=new Element('document');doc.body=new Element('body');doc.append(doc.body);doc.documentElement=doc.body;doc.createElement=tag=>new Element(tag);doc.getElementById=id=>doc.querySelector('#'+id);
 const win=new Element('window');win.innerWidth=1000;win.innerHeight=800;win.getComputedStyle=()=>({paddingLeft:'0',paddingRight:'0',paddingTop:'0',paddingBottom:'0'});
 const observers=[];win.MutationObserver=class{constructor(callback){observers.push(callback);}observe(){}};
 const reader=doc.createElement('dialog');reader.id='reader';reader.open=true;doc.body.append(reader);
 const content=doc.createElement('div');content.id='reader-content';content.scrollTop=427;content.scrollLeft=23;reader.append(content);
 function addImage({width=2000,height=1000,src='https://atlas.test/photo.webp'}={}){
  const figure=doc.createElement('figure');figure.className='content-illustration';const image=doc.createElement('img');image.src=src;image.alt='Схема процесса';image.naturalWidth=width;image.naturalHeight=height;image.complete=!!width;figure.append(image);content.append(figure);return image;
 }
 const image=addImage();image.focus();
 function fire(type,target,props={}){
  const event={type,target,button:0,key:'',defaultPrevented:false,preventDefault(){this.defaultPrevented=true;},stopPropagation(){this.stopped=true;},stopImmediatePropagation(){this.stopped=true;this.immediate=true;},...props};
  const path=[];for(let node=target;node;node=node.parentElement)path.push(node);
  const call=(node,capture)=>{for(const listener of node.listeners[type]||[]){if(!!listener.capture===capture)listener.fn(event);if(event.immediate)break;}};
  for(const node of [...path].reverse()){call(node,true);if(event.stopped)break;}
  if(!event.stopped)for(const node of path){call(node,false);if(event.stopped)break;}
  return event;
 }
 return {doc,win,reader,content,image,addImage,fire,mutate:()=>observers.forEach(callback=>callback([{addedNodes:[content]}]))};
}

test('dynamic illustrations open with Enter and Space and expose accessible controls',()=>{
 assert.equal(typeof Viewer.init,'function');const env=setup();Viewer.init(env.doc,env.win);
 assert.equal(env.image.getAttribute('tabindex'),'0');assert.equal(env.image.getAttribute('aria-haspopup'),'dialog');
 const next=env.addImage();env.mutate();assert.equal(next.getAttribute('tabindex'),'0');
 for(const key of ['Enter',' ']){
  const event=env.fire('keydown',next,{key}),dialog=env.doc.querySelector('.image-viewer');
  assert.equal(event.defaultPrevented,true);assert.equal(dialog.open,true);assert.ok(dialog.getAttribute('aria-label'));
  assert.ok(dialog.querySelector('.image-viewer-close').getAttribute('aria-label'));
  env.fire('click',dialog.querySelector('.image-viewer-close'));assert.equal(dialog.open,false);
 }
});

test('Escape closes only the viewer, restores focus and preserves reader scroll',()=>{
 assert.equal(typeof Viewer.init,'function');const env=setup();Viewer.init(env.doc,env.win);
 let escaped=0;env.doc.addEventListener('keydown',event=>{if(event.key==='Escape'){escaped++;env.reader.close();}});
 env.fire('click',env.image);const dialog=env.doc.querySelector('.image-viewer');
 assert.equal(dialog.open,true);const event=env.fire('keydown',dialog.querySelector('.image-viewer-close'),{key:'Escape'});
 assert.equal(event.defaultPrevented,true);assert.equal(dialog.open,false);assert.equal(escaped,0);assert.equal(env.reader.open,true);
 assert.equal(env.doc.activeElement,env.image);assert.deepEqual(env.image.focusOptions,{preventScroll:true});
 assert.equal(env.content.scrollTop,427);assert.equal(env.content.scrollLeft,23);
});

test('modifier clicks and modified keyboard activations stay available to editorial selection',()=>{
 assert.equal(typeof Viewer.init,'function');const env=setup();Viewer.init(env.doc,env.win);
 for(const modifier of ['metaKey','ctrlKey','shiftKey','altKey'])for(const type of ['click','keydown']){
  const event=env.fire(type,env.image,{[modifier]:true,key:'Enter'});
  assert.equal(env.doc.querySelector('.image-viewer').open,false);assert.equal(event.defaultPrevented,false);
 }
});

test('100 percent shows full natural dimensions then returns to viewport fit',()=>{
 assert.equal(typeof Viewer.init,'function');const env=setup();Viewer.init(env.doc,env.win);env.fire('click',env.image);
 const dialog=env.doc.querySelector('.image-viewer'),image=dialog.querySelector('img'),toggle=dialog.querySelector('.image-viewer-size');
 assert.ok(Number.parseFloat(image.style.width)<=960);assert.ok(Number.parseFloat(image.style.height)<=640);
 env.fire('click',toggle);assert.equal(image.style.width,'2000px');assert.equal(image.style.height,'1000px');assert.equal(toggle.getAttribute('aria-pressed'),'true');
 env.fire('click',toggle);assert.ok(Number.parseFloat(image.style.width)<=960);assert.equal(toggle.getAttribute('aria-pressed'),'false');
});

test('opening SVG and raster illustrations keeps 100 percent size when they fit',()=>{
 const env=setup();Viewer.init(env.doc,env.win);
 const vector=env.addImage({width:300,height:150,src:svg('<svg viewBox="0 0 900 450"/>')});env.fire('click',vector);
 const dialog=env.doc.querySelector('.image-viewer'),image=dialog.querySelector('img'),toggle=dialog.querySelector('.image-viewer-size');
 assert.equal(image.style.width,'900px');assert.equal(image.style.height,'450px');
 env.fire('click',toggle);assert.equal(image.style.width,'900px');assert.equal(image.style.height,'450px');
 env.fire('click',toggle);assert.equal(image.style.width,'900px');assert.equal(image.style.height,'450px');
 env.fire('click',dialog.querySelector('.image-viewer-close'));
 const raster=env.addImage({width:900,height:450});env.fire('click',raster);
 assert.equal(image.style.width,'900px');assert.equal(image.style.height,'450px');
});

test('oversized SVG opens fitted to the window and can still be viewed at 100 percent',()=>{
 const env=setup();Viewer.init(env.doc,env.win);
 const vector=env.addImage({src:svg('<svg viewBox="0 0 2400 1600"/>')});env.fire('click',vector);
 const dialog=env.doc.querySelector('.image-viewer'),image=dialog.querySelector('img'),toggle=dialog.querySelector('.image-viewer-size');
 assert.equal(image.style.width,'960px');assert.equal(image.style.height,'640px');
 env.fire('click',toggle);assert.equal(image.style.width,'2400px');assert.equal(image.style.height,'1600px');
});

test('only an intentional backdrop gesture dismisses the image',()=>{
 assert.equal(typeof Viewer.init,'function');const env=setup();Viewer.init(env.doc,env.win);env.fire('click',env.image);
 const dialog=env.doc.querySelector('.image-viewer'),image=dialog.querySelector('img');
 env.fire('pointerdown',image);env.fire('click',dialog,{clientX:5,clientY:5});assert.equal(dialog.open,true);
 env.fire('pointerdown',dialog,{clientX:5,clientY:5});env.fire('pointercancel',dialog);env.fire('click',dialog,{clientX:5,clientY:5});assert.equal(dialog.open,true);
 env.fire('pointerdown',dialog,{clientX:5,clientY:5});env.fire('click',dialog,{clientX:5,clientY:5});assert.equal(dialog.open,false);assert.equal(env.reader.open,true);
});

test('an image that finishes loading updates the viewer without retaining a previous image size',()=>{
 assert.equal(typeof Viewer.init,'function');const env=setup();Viewer.init(env.doc,env.win);env.fire('click',env.image);
 const dialog=env.doc.querySelector('.image-viewer'),preview=dialog.querySelector('img'),toggle=dialog.querySelector('.image-viewer-size');
 env.fire('click',dialog.querySelector('.image-viewer-close'));
 const next=env.addImage({width:0,height:0});env.fire('click',next);assert.equal(toggle.disabled,true);
 preview.naturalWidth=240;preview.naturalHeight=120;env.fire('load',preview);
 assert.equal(preview.style.width,'240px');assert.equal(preview.style.height,'120px');assert.equal(toggle.disabled,false);
});

test('viewport changes refit to 80 percent height while 100 percent preserves the image size',()=>{
 const env=setup();Viewer.init(env.doc,env.win);env.fire('click',env.image);
 const dialog=env.doc.querySelector('.image-viewer'),image=dialog.querySelector('img'),toggle=dialog.querySelector('.image-viewer-size');
 env.win.innerWidth=500;env.win.innerHeight=400;env.fire('resize',env.win);
 assert.equal(image.style.width,'460px');assert.equal(image.style.height,'230px');
 env.fire('click',toggle);env.win.innerWidth=400;env.fire('resize',env.win);
 assert.equal(image.style.width,'2000px');assert.equal(image.style.height,'1000px');
});

test('failed images have an accessible error and do not allow a misleading original-size toggle',()=>{
 const env=setup();Viewer.init(env.doc,env.win);env.fire('click',env.image);
 const dialog=env.doc.querySelector('.image-viewer'),preview=dialog.querySelector('img');env.fire('error',preview);
 assert.equal(preview.hidden,true);assert.equal(dialog.querySelector('.image-viewer-size').disabled,true);
 const status=dialog.querySelector('.image-viewer-status');assert.equal(status.hidden,false);assert.equal(status.getAttribute('role'),'status');assert.ok(status.textContent);
});

test('closing the parent reader also dismisses the viewer',()=>{
 const env=setup();Viewer.init(env.doc,env.win);env.fire('click',env.image);env.reader.close();
 assert.equal(env.doc.querySelector('.image-viewer').open,false);
});
