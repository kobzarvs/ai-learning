(function(scope){
'use strict';
class EditError extends Error{}
const parts=value=>String(value??'').split(/\n\s*\n/);
const esc=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const attr=(path,part)=>path?` data-edit-path="${esc(path)}"${part===undefined?'':` data-edit-part="${part}"`}`:'';
function allowed(path){
 return /^node\.(what|how|tasks|example|note|lab|input|output|training|criteria)$/.test(path)||
  /^lesson\.(?:variants\.\d+\.lesson\.)?(?:(?:rust|mlx)\.)?(goal|explanation|expected|limits|parity|exercise|deviceNote)$/.test(path)||
  /^lesson\.guide\.(intro|route\.\d+|foundations\.\d+|groups\.\d+\.purpose|groups\.\d+\.methods\.\d+\.(meaning|example|data|assumptions)|cases\.\d+\.(kind|question|approach|lesson))$/.test(path)||
  /^sections\.(article|methods|python|rust|mlx|exercise)\.\d+(?:\.blocks\.\d+)*(?:\.(body|title|caption|sourceLabel)|\.items\.\d+)?$/.test(path);
}
function lookup(payload,path){
 let value=payload;
 for(const key of path.split('.')){
  if(value===null||typeof value!=='object'||!Object.hasOwn(value,key))throw new EditError('Текст уже изменился. Обнови материал и выдели его снова.');
  value=value[key];
 }
 return value;
}
function remove(payload,targets){
 if(!Array.isArray(targets)||targets.length<1||targets.length>80)throw new EditError('Выбери от 1 до 80 фрагментов.');
 const planned=new Map(),aliases=new Map();
 for(const item of targets){
  if(!item||typeof item.path!=='string'||item.path.length>300||!allowed(item.path))throw new EditError('Этот фрагмент нельзя удалить напрямую.');
  const value=lookup(payload,item.path);
  if(item.part!==undefined&&(typeof value!=='string'||!Number.isInteger(item.part)||item.part<0||item.part>=parts(value).length))throw new EditError('Выделенный абзац изменился. Выдели его снова.');
  if(item.part===undefined&&typeof value!=='string'&&!Array.isArray(value)&&!(value&&typeof value==='object'&&typeof value.kind==='string'))throw new EditError('Неподдерживаемый фрагмент.');
  if(!planned.has(item.path))planned.set(item.path,{whole:false,parts:new Set()});
  const entry=planned.get(item.path);if(item.part===undefined)entry.whole=true;else entry.parts.add(item.part);
 }
 // A shared example is displayed once but backed by two equal legacy fields.
 const normalized=value=>typeof value==='string'?value.replace(/\s+/g,' ').trim():null;
 for(const [path,entry]of [...planned])if(['node.example','node.tasks'].includes(path)&&normalized(payload.node?.example)===normalized(payload.node?.tasks)){
  const alias=path==='node.example'?'node.tasks':'node.example';if(!planned.has(alias)){planned.set(alias,entry);aliases.set(alias,path);}
 }
 const paths=[...planned.keys()].filter(path=>![...planned].some(([parent,entry])=>parent!==path&&entry.whole&&path.startsWith(parent+'.')));
 const replacements=new Map(paths.map(path=>{
  const {whole,parts:selected}=planned.get(path),value=lookup(payload,aliases.get(path)||path);
  return [path,whole?null:parts(value).filter((_,index)=>!selected.has(index)).join('\n\n')];
 }));
 function visit(value,path=''){
  if(replacements.has(path)){
   const replacement=replacements.get(path);
   return replacement===null?(typeof value==='string'?'':undefined):replacement;
  }
  if(Array.isArray(value))return value.flatMap((item,index)=>{
   const childPath=path+'.'+index;
   // Array strings are entries, so remove them, not an empty bullet.
   if(replacements.has(childPath)&&replacements.get(childPath)===null)return [];
   const next=visit(item,childPath);return next===undefined?[]:[next];
  });
  if(value&&typeof value==='object'){
   const next=Object.fromEntries(Object.entries(value).map(([key,item])=>[key,visit(item,path?path+'.'+key:key)]));
   if(value.kind==='group'&&value.blocks?.length&&!next.blocks?.length)return undefined;
   if(value.kind==='text'&&value.body?.trim()&&!next.body?.trim())return undefined;
   if(value.kind==='list'&&value.items?.length&&!next.items?.length)return undefined;
   return next;
  }
  return value;
 }
 const updated=visit(payload);
 if(JSON.stringify(updated)===JSON.stringify(payload))throw new EditError('Выбранные фрагменты уже удалены.');
 return updated;
}
const api={EditError,parts,attr,allowed,lookup,remove};
if(typeof module!=='undefined'&&module.exports)module.exports=api;else scope.AtlasContentEdit=api;
})(typeof window!=='undefined'?window:globalThis);
