(function(scope){
'use strict';
const labels={clarify:'Расшифровать',rewrite:'Переписать по-человечески',custom:'Своя инструкция'};
const statuses={pending:'На рассмотрении',accepted:'Принято',rejected:'Отклонено',done:'Выполнено'};
const tabNames={article:'Теория',methods:'Методы и случаи',python:'Python',rust:'Rust + Burn',mlx:'Python + MLX',exercise:'Эксперимент / задание'};
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const normalize=value=>String(value||'').replace(/\s+/g,' ').trim();
const blockSelector='p,h1,h2,h3,h4,li,dt,dd,blockquote,figcaption,.guide-method-heading>strong,.guide-method-heading>span,.guide-case>summary>strong';

function textBlock(target,content){
 if(!target?.closest||!content.contains(target)||target.closest('button,input,textarea,select,pre,code,svg,.reading-sources,.reading-topics'))return null;
 const block=target.closest(blockSelector);
 if(!block||!content.contains(block)||!normalize(block.textContent))return null;
 // A list item containing whole sections is not a single text block.
 if(block.matches('li')&&block.querySelector('p,h2,h3,pre,ul,ol'))return null;
 return block;
}
function snapshot(block,content,context){
 const section=block.closest('section'),method=block.closest('.guide-method'),panel=block.closest('[data-guide-panel]');
 return {topicId:context.topicId,tab:context.tab,contentRevision:context.revision||'0',quote:normalize(block.textContent),
  anchor:{methodId:method?.id?.replace(/^causal-method-/,'')||'',panel:panel?.dataset.guidePanel||'',
   section:normalize(section?.querySelector('h2,h3')?.textContent).slice(0,300),
   block:String([...content.querySelectorAll(blockSelector)].indexOf(block))}};
}

function create({document:doc=scope.document,fetch:fetcher=scope.fetch.bind(scope)}={}){
 const reader=doc.getElementById('reader'),content=doc.getElementById('reader-content');
 if(!reader||!content)return null;
 let context=null,user=null,configured=false,authLoaded=false,hover=null,selected=null,point=null,meta=false,draft=null,saving=false;
 let filter='pending',notes=[],next=null,loading=false,queueGeneration=0;
 const menu=doc.createElement('dialog');menu.className='editorial-menu';menu.setAttribute('aria-label','Замечание к тексту');doc.body.append(menu);
 const queue=doc.createElement('dialog');queue.className='editorial-queue';queue.setAttribute('aria-label','Замечания к материалам');doc.body.append(queue);
 const top=doc.createElement('button');top.className='editorial-entry';top.textContent='Замечания';top.title='Замечания к текстам и модерация';doc.querySelector('.header-right')?.append(top);
 const entry=doc.createElement('button');entry.className='editorial-entry editorial-reader-entry';entry.textContent='Замечания';entry.title='⌘ + клик по тексту — оставить замечание';reader.querySelector('header')?.append(entry);
 const status=doc.createElement('div');status.className='editorial-toast';status.setAttribute('role','status');status.hidden=true;doc.body.append(status);
 const announce=text=>{(reader.open?reader:doc.body).append(status);status.textContent=text;status.hidden=false;setTimeout(()=>{status.hidden=true;},5000);};
 const setHover=block=>{if(hover===block)return;hover?.classList.remove('editorial-hover');hover=block;hover?.classList.add('editorial-hover');};
 const canPick=()=>reader.open&&context&&context.tab!=='note'&&!menu.open&&!queue.open;
 const loginURL=()=>'/api/auth?action=login&returnTo='+encodeURIComponent('/'+scope.location.hash);
 const loginHTML=()=>configured?`<a class="editorial-primary" href="${esc(loginURL())}" data-editorial-login>Войти через GitHub</a>`:'<p>Вход ещё настраивается. Пока можно свободно читать материалы.</p>';
 const remember=()=>{try{if(draft)scope.sessionStorage.setItem('atlas-editorial-draft',JSON.stringify(draft));}catch{}};
 async function api(url,options){
  const response=await fetcher(url,{credentials:'same-origin',cache:'no-store',...options});
  const data=await response.json();if(!response.ok)throw Error(data.error||'Запрос не выполнен.');return data;
 }
 async function session(){
  try{const data=await api('/api/auth');user=data.user;configured=data.configured;}
  catch{user=null;configured=false;}
  authLoaded=true;top.textContent=user?.isModerator?'Модерация':user?'Мои замечания':'Войти / замечания';
  entry.textContent=user?.isModerator?'Модерация':'Замечания';
 }
 function closeMenu(){if(saving)return;menu.close();selected?.classList.remove('editorial-selected');selected=null;draft=null;setHover(null);}
 function showMenu(data,block){
  draft={...data,requestId:scope.crypto.randomUUID()};selected=block;setHover(null);block?.classList.add('editorial-selected');
  menu.innerHTML=`<header><h2>Доработать текст</h2><button type="button" data-editorial-close aria-label="Закрыть замечание">×</button></header>
   <blockquote>${esc(draft.quote)}</blockquote>
   ${user?`<p>${user.isModerator?'Команда от твоего аккаунта. Автоматический исполнитель ещё не подключён; пока она сохранится в базе.':'Замечание попадёт в базу на рассмотрение модератору.'}</p><div class="editorial-actions"><button data-editorial-action="clarify">Расшифровать</button><button data-editorial-action="rewrite">Переписать по-человечески</button></div>
   <form><label for="editorial-instruction">Своя инструкция</label><textarea id="editorial-instruction" rows="3" maxlength="3000" placeholder="Например: объясни на примере сотрудников, без сокращений">${esc(data.instruction||'')}</textarea><button class="editorial-primary" type="submit">Отправить инструкцию</button></form>`:
   `<p>Чтобы оставить замечание, войди в свой аккаунт. Читать материалы можно без входа.</p>${loginHTML()}`}
   <p class="editorial-message" role="status"></p>`;
  const rect=block?.getBoundingClientRect(),width=Math.min(500,scope.innerWidth-32);
  menu.style.width=width+'px';menu.style.left=Math.max(16,Math.min(rect?.left??(scope.innerWidth-width)/2,scope.innerWidth-width-16))+'px';
  menu.style.top=Math.max(16,Math.min((rect?.top??120)+35,scope.innerHeight-620))+'px';
  if(!menu.open)menu.showModal();
 }
 async function submit(action){
  if(saving||!draft||!user)return;
  const input=menu.querySelector('textarea'),instruction=input?.value.trim()||'',message=menu.querySelector('.editorial-message');
  if(action==='custom'&&!instruction){message.textContent='Напиши, что нужно изменить.';input.focus();return;}
  saving=true;menu.querySelectorAll('button').forEach(b=>b.disabled=true);message.textContent='Сохраняю…';
  try{
   await api('/api/editorial-notes',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...draft,action,instruction})});
   try{scope.sessionStorage.removeItem('atlas-editorial-draft');}catch{}
   saving=false;closeMenu();announce(user.isModerator?'Команда сохранена. Исполнитель ещё не подключён.':'Замечание сохранено и отправлено на рассмотрение.');
  }catch(error){message.textContent=error.message;}
  finally{saving=false;menu.querySelectorAll('button').forEach(b=>b.disabled=false);}
 }
 menu.addEventListener('click',event=>{
  if(event.target.closest('[data-editorial-close]'))closeMenu();
  const action=event.target.closest('[data-editorial-action]')?.dataset.editorialAction;if(action)submit(action);
  if(event.target.closest('[data-editorial-login]'))remember();
  if(event.target===menu){const r=menu.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)closeMenu();}
 });
 menu.addEventListener('submit',event=>{event.preventDefault();submit('custom');});
 menu.addEventListener('cancel',event=>{event.preventDefault();event.stopPropagation();if(!saving)closeMenu();});
 menu.addEventListener('input',()=>{if(draft)draft.instruction=menu.querySelector('textarea')?.value||'';});

 function queueShell(){
  queue.innerHTML=`<header><div><h2>${user?.isModerator?'Модерация замечаний':'Мои замечания'}</h2><p>${user?esc(user.name)+' · @'+esc(user.login):'Для отправки замечаний нужен вход'}</p></div><button data-queue-close aria-label="Закрыть список замечаний">×</button></header>
   ${user?`<div class="editorial-queue-tools"><label>Показывать <select data-note-filter>${Object.entries({pending:'Новые',accepted:'Принятые',rejected:'Отклонённые',done:'Выполненные',all:'Все'}).map(([key,name])=>`<option value="${key}"${filter===key?' selected':''}>${name}</option>`).join('')}</select></label><button data-queue-refresh>Обновить</button><button data-logout>Выйти</button></div>
   <p class="editorial-help">Для новой пометки зажми ⌘ и нажми на нужный блок текста. ${user.isModerator?'Замечания читателей проходят твою модерацию. Их одобрение не запускает автоматическую правку.':'Модератор рассмотрит замечание; его решение появится здесь.'}</p><div class="editorial-note-list"></div><button data-load-more hidden>Показать ещё</button>`:
   `<p>Пометки сохраняются с твоим именем. Другие читатели видят только свои заявки; модератор — все.</p>${loginHTML()}`}
   <p class="editorial-message" role="status"></p>`;
 }
 function renderNotes(){
  const list=queue.querySelector('.editorial-note-list');if(!list)return;
  list.innerHTML=notes.length?notes.map(note=>`<article class="editorial-note" data-note-id="${esc(note.id)}"><header><div><strong>${esc(note.topic_title||note.topic_id)}</strong><p>${esc(tabNames[note.tab]||note.tab)}${note.anchor?.section?' · '+esc(note.anchor.section):''}</p></div><span class="editorial-status status-${esc(note.status)}">${esc(statuses[note.status])}</span></header>
   <p class="editorial-author">@${esc(note.login||user.login)} · ${esc(new Date(note.created_at).toLocaleString('ru-RU'))}${note.execution_mode==='owner_request'?' · Команда владельца':''}</p>
   <blockquote>${esc(note.quote)}</blockquote><p><strong>${esc(labels[note.action])}</strong></p>${note.instruction?`<p class="editorial-instruction">${esc(note.instruction)}</p>`:''}
   ${note.moderator_note?`<p class="editorial-moderator-comment"><strong>Модератор:</strong> ${esc(note.moderator_note)}</p>`:''}
   ${user.isModerator?`<label>Комментарий к решению<textarea data-moderator-comment rows="2" maxlength="3000">${esc(note.moderator_note)}</textarea></label><div class="editorial-actions">${Object.entries({accepted:'Принять',rejected:'Отклонить',done:'Выполнено',pending:'Вернуть на рассмотрение'}).filter(([key])=>key!==note.status).map(([key,title])=>`<button data-moderate="${key}">${title}</button>`).join('')}</div>`:''}</article>`).join(''):'<p class="editorial-empty">Здесь пока нет замечаний.</p>';
  queue.querySelector('[data-load-more]').hidden=!next;
 }
 async function loadNotes(append=false){
  if(!user)return;const generation=++queueGeneration;loading=true;
  const message=queue.querySelector('.editorial-message');message.textContent='Загружаю…';
  try{
   const params=new URLSearchParams({scope:user.isModerator?'all':'mine',status:filter});if(append&&next)params.set('before',next);
   const data=await api('/api/editorial-notes?'+params);
   if(generation!==queueGeneration)return;notes=append?[...notes,...data.notes]:data.notes;next=data.next;renderNotes();message.textContent='';
  }catch(error){if(generation===queueGeneration)message.textContent=error.message;}
  finally{if(generation===queueGeneration)loading=false;}
 }
 async function openQueue(){
  setHover(null);if(!authLoaded)await session();filter=user?.isModerator?'pending':'all';queueShell();
  if(!queue.open)queue.showModal();await loadNotes();
 }
 top.onclick=entry.onclick=openQueue;
 queue.addEventListener('cancel',event=>{event.stopPropagation();});
 queue.addEventListener('change',event=>{if(event.target.matches('[data-note-filter]')){filter=event.target.value;loadNotes();}});
 queue.addEventListener('click',async event=>{
  const button=event.target.closest('button');if(!button)return;
  if(button.hasAttribute('data-queue-close')){queue.close();return;}
  if(button.hasAttribute('data-queue-refresh')){loadNotes();return;}
  if(button.hasAttribute('data-load-more')){if(!loading)loadNotes(true);return;}
  if(button.hasAttribute('data-logout')){
   try{await api('/api/auth',{method:'POST'});user=null;await session();queueShell();}catch(error){queue.querySelector('.editorial-message').textContent=error.message;}return;
  }
  if(button.dataset.moderate){
   const card=button.closest('[data-note-id]'),note=notes.find(n=>n.id===card.dataset.noteId);
   card.querySelectorAll('button').forEach(b=>b.disabled=true);
   try{
    await api('/api/editorial-notes',{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:note.id,version:note.version,status:button.dataset.moderate,comment:card.querySelector('textarea').value})});
    await loadNotes();
   }catch(error){queue.querySelector('.editorial-message').textContent=error.message;card.querySelectorAll('button').forEach(b=>b.disabled=false);}
  }
 });
 content.addEventListener('pointermove',event=>{point={x:event.clientX,y:event.clientY};meta=event.metaKey;setHover(meta&&canPick()?textBlock(event.target,content):null);});
 content.addEventListener('pointerleave',()=>{point=null;setHover(null);});
 doc.addEventListener('keydown',event=>{
  if(event.key==='Meta'&&!event.target.closest?.('input,textarea,[contenteditable="true"]')){meta=true;if(point&&canPick())setHover(textBlock(doc.elementFromPoint(point.x,point.y),content));}
 });
 doc.addEventListener('keyup',event=>{if(event.key==='Meta'||!event.metaKey){meta=false;setHover(null);}});
 scope.addEventListener('blur',()=>{meta=false;setHover(null);});
 content.addEventListener('scroll',()=>{setHover(null);point=null;});
 content.addEventListener('click',async event=>{
  if(!event.metaKey||!canPick())return;
  const block=textBlock(event.target,content);if(!block)return;
  event.preventDefault();event.stopImmediatePropagation();
  const data=snapshot(block,content,context);
  if(data.quote.length>6000){announce('Выбери более короткий абзац.');return;}
  if(!authLoaded)await session();
  if(reader.open)showMenu(data,block);
 },true);
 reader.addEventListener('close',()=>{setHover(null);point=null;});
 session().then(()=>{
  if(!user)return;
  try{const saved=JSON.parse(scope.sessionStorage.getItem('atlas-editorial-draft')||'null');if(saved?.quote&&saved?.topicId){scope.sessionStorage.removeItem('atlas-editorial-draft');showMenu(saved,null);}}catch{}
 });
 return {update(value){context=value;setHover(null);entry.hidden=value?.tab==='note';}};
}
const api={textBlock,snapshot,create};
if(typeof module!=='undefined'&&module.exports)module.exports=api;
else{scope.AtlasEditorial=api;scope.ATLAS_EDITORIAL=create();}
})(typeof window!=='undefined'?window:globalThis);
