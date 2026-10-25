(function(scope){
'use strict';
const selector='.content-illustration img',instances=new WeakMap();
const positive=value=>Number.isFinite(Number(value))&&Number(value)>0;
function size(width,height){return positive(width)&&positive(height)?{width:Number(width),height:Number(height)}:null;}
function fitSize(original,bounds){
 if(!original||!bounds||!size(original.width,original.height)||!size(bounds.width,bounds.height))return null;
 const scale=Math.min(1,bounds.width/original.width,bounds.height/original.height);
 return {width:original.width*scale,height:original.height*scale};
}
function svgSize(source){
 if(!/^data:image\/svg\+xml(?:;[^,]*)?,/i.test(source))return null;
 try{
  const comma=source.indexOf(','),header=source.slice(0,comma);
  // Published SVGs are URL-encoded image documents. Inspect dimensions as text;
  // their markup must never become nodes in the application document.
  if(/;base64/i.test(header))return null;
  const svg=decodeURIComponent(source.slice(comma+1)).replace(/<!--[\s\S]*?-->/g,'');
  const root=svg.match(/<svg\b([^>]*)>/i)?.[1];if(!root)return null;
  const attribute=name=>root.match(new RegExp('(?:^|\\s)'+name+'\\s*=\\s*(["\'])(.*?)\\1','i'))?.[2];
  const viewBox=attribute('viewBox')?.trim().split(/[\s,]+/).map(Number);
  if(viewBox?.length===4&&viewBox.every(Number.isFinite)&&size(viewBox[2],viewBox[3]))return size(viewBox[2],viewBox[3]);
  const pixels=value=>/^\s*\d+(?:\.\d+)?(?:px)?\s*$/i.test(value||'')?parseFloat(value):0;
  return size(pixels(attribute('width')),pixels(attribute('height')));
 }catch{return null;}
}
function originalSize(image){return svgSize(image.currentSrc||image.src||'')||size(image.naturalWidth,image.naturalHeight);}
function plainActivation(event){return !event.defaultPrevented&&!event.metaKey&&!event.ctrlKey&&!event.altKey&&!event.shiftKey;}
function init(doc=scope.document,win=scope){
 if(!doc?.body)return null;
 if(instances.has(doc))return instances.get(doc);
 const element=(tag,className,text)=>{const node=doc.createElement(tag);node.className=className;if(text)node.textContent=text;return node;};
 const dialog=element('dialog','image-viewer');dialog.setAttribute('aria-label','Просмотр иллюстрации');dialog.setAttribute('closedby','closerequest');
 const header=element('header','image-viewer-header'),title=element('strong','image-viewer-title','Иллюстрация');
 const toggle=element('button','image-viewer-size','100%');toggle.type='button';toggle.setAttribute('aria-pressed','false');toggle.setAttribute('aria-label','Показать в исходном размере, 100%');
 const closeButton=element('button','image-viewer-close','Закрыть');closeButton.type='button';closeButton.setAttribute('aria-label','Закрыть просмотр иллюстрации');
 const stage=element('div','image-viewer-stage');stage.setAttribute('tabindex','0');stage.setAttribute('role','region');stage.setAttribute('aria-label','Иллюстрация; в исходном размере доступна прокрутка');
 const image=element('img','image-viewer-image');image.decoding='async';image.referrerPolicy='no-referrer';image.draggable=false;
 const status=element('p','image-viewer-status');status.setAttribute('role','status');status.hidden=true;
 header.append(title,toggle,closeButton);stage.append(image,status);dialog.append(header,stage);doc.body.append(dialog);
 let source=null,original=null,actualSize=false,previousFocus=null,scroll=null,beganOutside=false;
 function prepare(){
  for(const img of doc.querySelectorAll(selector)){
   if(!img.hasAttribute('tabindex'))img.setAttribute('tabindex','0');
   img.setAttribute('role','button');img.setAttribute('aria-haspopup','dialog');
   img.setAttribute('aria-label',img.alt?'Увеличить: '+img.alt:'Увеличить иллюстрацию');
  }
 }
 function layout(){
  if(!dialog.open)return;
  const viewport=win.visualViewport,viewportWidth=viewport?.width||win.innerWidth,viewportHeight=viewport?.height||win.innerHeight;
  const bounds={width:Math.min(stage.clientWidth||viewportWidth-40,viewportWidth-40),height:Math.min(stage.clientHeight||viewportHeight*.8,viewportHeight*.8)};
  const dimensions=actualSize?original:fitSize(original,bounds);
  if(dimensions){image.style.width=dimensions.width+'px';image.style.height=dimensions.height+'px';}
  toggle.disabled=!original;
  toggle.textContent=actualSize?'Вписать':'100%';toggle.setAttribute('aria-pressed',String(actualSize));
  toggle.setAttribute('aria-label',actualSize?'Вписать изображение в окно':'Показать в исходном размере, 100%');
 }
 function restore(){
  beganOutside=false;
  if(previousFocus?.isConnected)previousFocus.focus({preventScroll:true});
  if(scroll?.node.isConnected){scroll.node.scrollTop=scroll.top;scroll.node.scrollLeft=scroll.left;}
  source=null;original=null;previousFocus=null;scroll=null;
  image.removeAttribute('src');
 }
 function close(){if(dialog.open)dialog.close();}
 function open(img){
  if(!img?.matches(selector)||dialog.open)return;
  prepare();source=img;original=originalSize(img);actualSize=false;previousFocus=doc.activeElement;
  const readerContent=img.closest('#reader-content');scroll=readerContent?{node:readerContent,top:readerContent.scrollTop,left:readerContent.scrollLeft}:null;
  image.style.width='';image.style.height='';image.hidden=false;image.alt=img.alt||'Иллюстрация';status.hidden=true;
  image.src=img.currentSrc||img.src;stage.scrollTop=0;stage.scrollLeft=0;
  dialog.showModal();layout();closeButton.focus({preventScroll:true});
 }
 closeButton.addEventListener('click',close);
 toggle.addEventListener('click',()=>{actualSize=!actualSize;stage.scrollTop=0;stage.scrollLeft=0;layout();});
 image.addEventListener('load',()=>{if(!source)return;original=originalSize(source)||originalSize(image);layout();});
 image.addEventListener('error',()=>{if(!source)return;image.hidden=true;status.hidden=false;status.textContent='Не удалось загрузить иллюстрацию.';toggle.disabled=true;});
 dialog.addEventListener('close',restore);
 dialog.addEventListener('cancel',event=>{event.preventDefault();event.stopPropagation();close();});
 const outside=event=>{const r=dialog.getBoundingClientRect();return event.target===dialog&&(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom);};
 dialog.addEventListener('pointerdown',event=>{beganOutside=event.button===0&&outside(event);});
 dialog.addEventListener('pointercancel',()=>{beganOutside=false;});
 dialog.addEventListener('click',event=>{const dismiss=beganOutside&&outside(event);beganOutside=false;if(dismiss){event.preventDefault();event.stopPropagation();close();}});
 // Capture prevents the existing reader/app Escape handlers from closing the
 // underlying material. Native modal dialogs provide the focus trap/inertness.
 doc.addEventListener('keydown',event=>{
  if(dialog.open&&event.key==='Escape'){event.preventDefault();event.stopImmediatePropagation();close();return;}
  if(!dialog.open&&plainActivation(event)&&(event.key==='Enter'||event.key===' ')&&event.target?.matches?.(selector)){
   event.preventDefault();event.stopPropagation();open(event.target);
  }
 },true);
 doc.addEventListener('click',event=>{
  if(!dialog.open&&plainActivation(event)&&event.button===0&&event.target?.matches?.(selector)){
   event.preventDefault();event.stopPropagation();open(event.target);
  }
 },true);
 doc.getElementById('reader')?.addEventListener('close',close);
 win.addEventListener('resize',layout);win.visualViewport?.addEventListener('resize',layout);
 prepare();
 if(win.MutationObserver)new win.MutationObserver(prepare).observe(doc.body,{childList:true,subtree:true});
 const api={open,close};instances.set(doc,api);return api;
}
const api={fitSize,originalSize,init};
if(typeof module!=='undefined'&&module.exports)module.exports=api;
else{scope.AtlasImageViewer=api;if(scope.document.readyState==='loading')scope.document.addEventListener('DOMContentLoaded',()=>init());else init();}
})(typeof window!=='undefined'?window:globalThis);
