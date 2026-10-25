(function(scope){
'use strict';
const key=target=>target.path+'#'+(target.part??'all');
function selection(){
 const chosen=new Map();
 return {get targets(){return [...chosen.values()];},clear(){chosen.clear();},
  toggle(targets){const all=targets.every(target=>chosen.has(key(target)));
   for(const target of targets)if(all)chosen.delete(key(target));else chosen.set(key(target),target);
  },has(target){return chosen.has(key(target));}};
}
function targetFor(element){
 if(!element?.dataset?.editPath)return null;
 return {path:element.dataset.editPath,...(element.dataset.editPart===undefined?{}:{part:Number(element.dataset.editPart)})};
}
function fragmentTargets(block,content){
 if(!block||!content.contains(block))return [];
 const source=block.closest('[data-edit-path]'),normalize=value=>String(value||'').replace(/\s+/g,' ').trim();
 if(source&&(source===block||normalize(source.textContent)===normalize(block.textContent)))return [targetFor(source)];
 const section=block.closest('[data-edit-section]');
 if(section&&block.parentElement===section&&block.matches('h1,h2,h3,h4'))return [{path:section.dataset.editSection+'.title'}];
 return [];
}
function create({reader,content,api,getUser,ensureUser,announce,refresh,document:doc=scope.document}){
 let context=null,busy=false,pending=null,editingSession=false,completionText='Удаление сохранено';const chosen=selection(),undo=new Map();
 const bar=doc.createElement('div');bar.className='reader-edit-bar';bar.hidden=true;
 bar.innerHTML='<span data-edit-count role="status"></span><button type="button" data-edit-delete>Удалить · Delete</button><button type="button" data-edit-clear>Снять выделение · Esc</button><button type="button" data-edit-undo>Отменить · ⌘Z</button><button type="button" data-edit-save>Сохранить</button>';
 reader.insertBefore(bar,content);
 const button=name=>bar.querySelector('[data-edit-'+name+']');
 const active=()=>reader.open&&context&&context.tab!=='note'&&!doc.querySelector('dialog[open]:not(#reader)');
 function paint(){
  for(const element of content.querySelectorAll('[data-edit-path]'))element.classList.toggle('reader-edit-selected',chosen.has(targetFor(element)));
  for(const element of content.querySelectorAll('[data-edit-section]'))element.classList.toggle('reader-edit-selected',chosen.has({path:element.dataset.editSection}));
  const count=chosen.targets.length,canUndo=Boolean(undo.get(context?.topicId)?.length);
  bar.hidden=!getUser()?.isModerator||(!count&&!editingSession&&!pending&&!busy)||context?.tab==='note';
  button('count').textContent=busy?'Сохраняю…':pending?'Сохранение не подтверждено. Повтори запрос.':count?'Выбрано фрагментов: '+count:completionText;
  button('delete').textContent=pending?'Повторить сохранение':'Удалить · Delete';
  button('delete').disabled=busy||(!count&&!pending);button('clear').disabled=busy||!!pending||!count;
  button('undo').disabled=busy||!!pending||!canUndo;
  button('save').disabled=busy||!!pending;
  reader.classList.toggle('reader-can-edit',Boolean(getUser()?.isModerator));
 }
 async function save(action){
  if(busy||!active()||!getUser()?.isModerator)return;
  if(!pending){
   const history=undo.get(context.topicId)||[];
   if(action==='undo'&&!history.length||action==='delete'&&!chosen.targets.length)return;
   pending={requestId:scope.crypto.randomUUID(),topicId:context.topicId,expectedRevision:context.revision,action,
    ...(action==='delete'?{targets:chosen.targets}:{editId:history.at(-1)})};
  }
  const operation=pending,from={...context};busy=true;paint();
  try{
   const result=await api('/api/content-edit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(operation)});
   const history=undo.get(operation.topicId)||[];
   if(operation.action==='delete')history.push(result.editId);else history.pop();
   undo.set(operation.topicId,history);pending=null;chosen.clear();
   await refresh(result.revision);
   editingSession=reader.open&&context?.topicId===from.topicId&&context?.tab===from.tab&&context?.view===from.view;
   completionText=operation.action==='delete'?'Удаление сохранено':'Удаление отменено';
   announce(operation.action==='delete'?'Удалено. ⌘Z отменит правку.':'Удаление отменено.');
  }catch(error){
   if([400,401,403,404,409].includes(error.status)){pending=null;chosen.clear();if(error.status===409)await refresh();}
   announce(error.message);
  }finally{busy=false;paint();}
 }
 button('delete').onclick=()=>save('delete');button('undo').onclick=()=>save('undo');
 button('clear').onclick=()=>{chosen.clear();paint();};
 button('save').onclick=()=>{if(busy||pending)return;editingSession=false;chosen.clear();paint();};
 content.addEventListener('click',async event=>{
  if(!event.shiftKey||!(event.metaKey||event.ctrlKey)||!active()||busy||pending)return;
  if(event.target.closest('input,textarea,select,button,pre,code,.reading-sources,.reading-topics'))return;
  let targets=[];
  const heading=event.target.closest('h1,h2,h3,h4'),section=heading?.closest('section');
  if(section?.dataset.editSection)targets=[{path:section.dataset.editSection}];
  else if(section)targets=[...section.querySelectorAll('[data-edit-path]')].map(targetFor);
  else{const element=event.target.closest('[data-edit-path]');if(element&&content.contains(element))targets=[targetFor(element)];}
  if(!targets.length)return;
  event.preventDefault();event.stopImmediatePropagation();
  const before=context;
  const user=await ensureUser();
  if(!user?.isModerator){announce('Прямое редактирование доступно владельцу после входа.');return;}
  if(context?.topicId!==before.topicId||context?.tab!==before.tab||context?.revision!==before.revision)return;
  if(!targets.every(target=>chosen.has(target))&&new Set([...chosen.targets,...targets].map(key)).size>80){announce('За один раз можно выбрать до 80 фрагментов.');return;}
  chosen.toggle(targets);scope.getSelection()?.removeAllRanges();paint();
 },true);
 doc.addEventListener('keydown',event=>{
  if(!active()||!getUser()?.isModerator||event.target.closest?.('input,textarea,select,[contenteditable="true"]'))return;
  const remove=['Delete','Backspace'].includes(event.key)&&(chosen.targets.length||pending);
  const revert=(event.metaKey||event.ctrlKey)&&event.key.toLowerCase()==='z'&&!event.shiftKey&&(undo.get(context.topicId)?.length||pending);
  const clear=event.key==='Escape'&&chosen.targets.length;
  if(!remove&&!revert&&!clear)return;
  event.preventDefault();event.stopImmediatePropagation();
  if(clear){if(!busy&&!pending){chosen.clear();paint();}return;}
  if(revert&&pending){announce('Сначала подтверди сохранение кнопкой «Повторить сохранение».');return;}
  if(!event.repeat)save(revert?'undo':'delete');
 },true);
 reader.addEventListener('close',()=>{chosen.clear();editingSession=false;paint();});
 return {refresh:paint,canRemove:()=>!busy&&!pending&&Boolean(getUser()?.isModerator),
  async removeFragment(targets,from){
   if(busy||pending){announce('Сначала дождись сохранения предыдущей правки или повтори её.');return;}
   if(!active()||!getUser()?.isModerator)return;
   if(!from||context.topicId!==from.topicId||context.tab!==from.tab||context.revision!==from.revision||context.view!==from.view){
    announce('Материал изменился. Выбери фрагмент снова.');return;
   }
   if(!Array.isArray(targets)||!targets.length)return;
   chosen.clear();chosen.toggle(targets);paint();await save('delete');
  },reset(){chosen.clear();undo.clear();pending=null;editingSession=false;paint();},update(value){
  if(context?.topicId!==value?.topicId||context?.tab!==value?.tab||context?.revision!==value?.revision||context?.view!==value?.view){chosen.clear();editingSession=false;}
  context=value;paint();
 }};
}
const api={selection,targetFor,fragmentTargets,create};
if(typeof module!=='undefined'&&module.exports)module.exports=api;else scope.AtlasReaderEdit=api;
})(typeof window!=='undefined'?window:globalThis);
