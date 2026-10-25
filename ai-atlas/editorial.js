(function(scope){
'use strict';
const labels={clarify:'Расшифровать',rewrite:'Переписать по-человечески',custom:'Своя инструкция'};
const statuses={pending:'На рассмотрении',accepted:'Принято',rejected:'Отклонено',done:'Выполнено'};
const tabNames={topic:'Весь раздел',article:'Теория',methods:'Методы и случаи',python:'Python',rust:'Rust + Burn',mlx:'Python + MLX',exercise:'Эксперимент / задание'};
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const normalize=value=>String(value||'').replace(/\s+/g,' ').trim();
const activeRun=run=>['queued','running'].includes(run?.status);
function runMarkup(run,worker={online:false}){
 if(!run)return '';
 const status=['queued','running','completed','failed'].includes(run.status)?run.status:'queued';
 const label={queued:'В очереди',running:'Агент выполняет',completed:'Выполнено',failed:'Ошибка выполнения'}[status];
 const time=value=>value&&Number.isFinite(Date.parse(value))?new Date(value).toLocaleTimeString('ru-RU',{hour:'2-digit',minute:'2-digit',second:'2-digit'}):'';
 const warning=status==='queued'&&!worker.online?'Исполнитель недоступен. Задание сохранено и запустится после подключения.':'';
 return `<div class="agent-run-state agent-${status}" role="status"><div class="agent-run-heading"><span class="agent-indicator" aria-hidden="true"></span><strong>${label}</strong></div>
  ${run.phase?`<p>${esc(run.phase)}</p>`:''}${warning?`<p class="agent-run-warning">${warning}</p>`:''}
  ${run.started_at?`<p class="agent-run-time">Начало: ${esc(time(run.started_at))}${run.heartbeat_at&&status==='running'?' · Последний сигнал: '+esc(time(run.heartbeat_at)):''}</p>`:''}
  ${status==='completed'&&run.summary?`<p class="agent-run-result">${esc(run.summary)}</p>`:''}
  ${status==='failed'?`<p class="agent-run-error">${esc(run.error||'Задание остановлено. Можно повторить запуск.')}</p>`:''}</div>`;
}
function executionButton(note,busy=false){
 const active=activeRun(note.run);
 const rework=note.status==='done'&&note.run?.status==='completed';
 if(!active&&!rework&&['done','rejected'].includes(note.status))return '';
 return `<button type="button" class="editorial-execute" data-execute="${rework?'rework':'execute'}"${active||busy?' disabled':''}>${rework?'Доработать':note.run?.status==='running'?'Агент выполняет':note.run?.status==='queued'?'В очереди':note.run?.status==='failed'?'Повторить запуск':'Выполнить сейчас'}</button>`;
}
function resultLink(note){
 const tab=Object.hasOwn(tabNames,note.tab)&&note.tab!=='topic'?note.tab:'article';
 return `<a class="editorial-result-link" href="/?tab=${tab}#${esc(encodeURIComponent(note.topic_id))}" target="_blank" rel="noopener noreferrer">Открыть результат ↗</a>`;
}
async function syncPublishedContent(client,runs){
 if(!client)return;
 const revision=runs.filter(run=>run.status==='completed'&&/^\d+$/.test(run.result_revision)).reduce((latest,run)=>BigInt(run.result_revision)>BigInt(latest)?String(run.result_revision):latest,'0');
 if(BigInt(revision)>BigInt(client.revision||'0'))await client.refresh({minRevision:revision});
}

function moderationButtons(note,busy=false){
 return Object.entries({accepted:'Принять',rejected:'Отклонить',done:'Отметить выполненным',pending:'Вернуть на рассмотрение'})
  .filter(([key])=>key!==note.status).map(([key,title])=>`<button data-moderate="${key}"${activeRun(note.run)||busy?' disabled':''}>${title}</button>`).join('');
}
async function fetchRunUpdates(api,noteIds=[]){
 const main=await api('/api/agent-runs'),runsById=new Map(main.runs.map(run=>[run.id,run]));
 for(let offset=0;offset<noteIds.length;offset+=50){
  const page=await api('/api/agent-runs?notes='+encodeURIComponent(noteIds.slice(offset,offset+50).join(',')));
  for(const run of page.runs)runsById.set(run.id,run);
 }
 return {...main,runs:[...runsById.values()]};
}
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
function topicSnapshot(context){
 return {topicId:context.topicId,tab:'topic',contentRevision:context.revision||'0',
  quote:normalize(context.title||context.topicId),anchor:{methodId:'',panel:'',section:'',block:''}};
}

function improvementInstruction(wishes=''){
 const base='Улучши весь раздел. Сначала прочитай полную текущую публикацию через read_topic: теорию, методы, примеры, код и задания. Вычитай материал целиком и при необходимости перестрой структуру и порядок объяснения. Убери мусор, повторы, лишние отступления и пустые формулировки; перефразируй сложные места понятным русским языком. Сохрани проверенные факты, существенные пояснения, работающий код и полезные источники. Проверь иллюстрации и SVG-схемы: соответствие тексту, подписи, читаемость, размеры и расположение; исправь неточности и выровняй оформление. Используй настоящие изображения или SVG, если нужна иллюстрация. Не раздувай материал без необходимости. Опубликуй улучшенный раздел и кратко перечисли реальные изменения.';
 return base+(wishes.trim()?'\n\nДополнительные пожелания владельца (имеют приоритет):\n'+wishes.trim():'');
}

function accountMarkup(user,id,loading=false,{reader=false}={}){
 if(loading||(reader&&user))return '';
 if(!user)return `<button class="account-login" data-account-login${loading?' disabled aria-busy="true"':''}>Войти</button>`;
 const name=user.name||user.login,initials=name.trim().split(/\s+/).slice(0,2).map(part=>part[0]).join('').toUpperCase();
 const avatar=/^\d+$/.test(String(user.id))?`<img src="https://avatars.githubusercontent.com/u/${user.id}?s=80" alt="" width="36" height="36" referrerpolicy="no-referrer">`:'';
 return `<button class="account-trigger" popovertarget="${id}" aria-controls="${id}" aria-expanded="false" aria-label="Аккаунт: ${esc(name)}" title="${esc(name)} · @${esc(user.login)}">
  <span class="account-avatar" aria-hidden="true"><span>${esc(initials)}</span>${avatar}</span><span class="account-name">${esc(name)}</span><svg class="account-chevron" viewBox="0 0 20 20" aria-hidden="true"><path d="m5 7.5 5 5 5-5"/></svg></button>
  <div id="${id}" class="account-menu" popover="auto" aria-label="Меню пользователя">
   <div class="account-identity"><strong>${esc(name)}</strong><span>@${esc(user.login)}</span></div>
   <button data-account-notes>${user.isModerator?'Модерация замечаний':'Мои замечания'}</button>
   <button data-account-logout>Выйти</button><p class="account-error" role="status"></p>
  </div>`;
}

function loginReturnTo(context,readerOpen,hash=''){
 if(readerOpen&&context&&/^[a-zA-Z0-9_-]+$/.test(context.topicId)&&['article','methods','python','rust','mlx','exercise'].includes(context.tab))return '/?tab='+context.tab+'#'+context.topicId;
 return '/'+(/^#[a-zA-Z0-9_-]+$/.test(hash)?hash:'');
}

function create({document:doc=scope.document,fetch:fetcher=scope.fetch.bind(scope)}={}){
 const reader=doc.getElementById('reader'),content=doc.getElementById('reader-content');
 if(!reader||!content)return null;
 let context=null,user=null,configured=false,authLoaded=false,hover=null,selected=null,point=null,meta=false,draft=null,saving=false;
 let menuDeletion=null;
 let notes=[],historyEntries=[],next=null,loading=false,queueGeneration=0,queueTarget='';
 let queueView='inbox',historyKind='changes',query='',cursors=[''],pageIndex=0,lastQueueLoad=0;
 let publicNoteGeneration=0,notificationSelection=0;
 let worker={online:false},runs=[],pollBusy=false,pollTimer=null;
 const commentDrafts=new Map(),dispatching=new Set();
 const menu=doc.createElement('dialog');menu.className='editorial-menu';menu.setAttribute('aria-label','Замечание к тексту');doc.body.append(menu);
 const queue=doc.createElement('dialog');queue.className='editorial-queue';queue.setAttribute('aria-label','Замечания к материалам');doc.body.append(queue);
 const accounts=[doc.querySelector('.header-right'),reader.querySelector('header')].filter(Boolean).map((parent,index)=>{
  const element=doc.createElement('div');element.className='atlas-account'+(index?' reader-account':'');parent.append(element);return element;
 });
 const activityButtons=accounts.map(element=>{const button=doc.createElement('button');button.className='agent-activity';button.type='button';button.hidden=true;element.before(button);button.addEventListener('click',()=>openQueue(true));return button;});
 const notifications=scope.AtlasNotifications?.create({anchors:accounts.filter(element=>element.closest('.header-right')),api,onSelect:openNotification,document:doc});
 const publicDialog=doc.createElement('dialog');publicDialog.className='editorial-public-note';publicDialog.setAttribute('aria-label','Замечание по разделу');doc.body.append(publicDialog);
 scope.AtlasUI?.bindDismiss(publicDialog);
 publicDialog.addEventListener('close',()=>{publicNoteGeneration++;});
 publicDialog.addEventListener('cancel',event=>event.stopPropagation());
 publicDialog.addEventListener('click',event=>{if(event.target.closest('[data-public-note-close]'))publicDialog.close();});
 const status=doc.createElement('div');status.className='editorial-toast';status.setAttribute('role','status');status.hidden=true;doc.body.append(status);
 const announce=text=>{(reader.open?reader:doc.body).append(status);status.textContent=text;status.hidden=false;setTimeout(()=>{status.hidden=true;},5000);};
 const direct=scope.AtlasReaderEdit?.create({reader,content,api,getUser:()=>user,ensureUser:async()=>{if(!authLoaded)await session();return user;},announce,
  refresh:revision=>scope.ATLAS_CONTENT_CLIENT?.refresh(revision?{minRevision:revision}:{}),document:doc});
 const setHover=block=>{if(hover===block)return;hover?.classList.remove('editorial-hover');hover=block;hover?.classList.add('editorial-hover');};
 const canPick=()=>reader.open&&context&&context.tab!=='note'&&!menu.open&&!queue.open;
 const returnTo=()=>loginReturnTo(context,reader.open,scope.location.hash);
 const loginURL=()=>'/api/auth?action=login&returnTo='+encodeURIComponent(returnTo());
 const loginHTML=()=>configured?`<a class="editorial-primary" href="${esc(loginURL())}" data-editorial-login>Войти через GitHub</a>`:'<p>Вход ещё настраивается. Пока можно свободно читать материалы.</p>';
 let returnView=null,returnReady=false,returnDraft=null;
 try{
  const saved=JSON.parse(scope.sessionStorage.getItem('atlas-reader-return')||'null');
  scope.sessionStorage.removeItem('atlas-reader-return');
  if(saved&&Date.now()-saved.createdAt<900000&&saved.returnTo===scope.location.pathname+scope.location.search+scope.location.hash)returnView=saved;
 }catch{}
 const remember=()=>{try{
  if(draft)scope.sessionStorage.setItem('atlas-editorial-draft',JSON.stringify(draft));
  if(reader.open&&context)scope.sessionStorage.setItem('atlas-reader-return',JSON.stringify({createdAt:Date.now(),returnTo:returnTo(),topicId:context.topicId,tab:context.tab,
   scrollTop:content.scrollTop,guideView:content.querySelector('.causal-guide')?.dataset.guideCurrent,
   details:[...content.querySelectorAll('details')].map(el=>el.open),codeScroll:[...content.querySelectorAll('pre')].map(el=>({left:el.scrollLeft,top:el.scrollTop}))}));
 }catch{}};
 function restoreAfterLogin(){
  if(!returnReady||!reader.open||!context)return;
  if(returnView){
   if(context.topicId===returnView.topicId&&context.tab===returnView.tab){
    if(returnView.guideView!==undefined)scope.AtlasLessonGuide?.selectView(content,returnView.guideView);
    [...content.querySelectorAll('details')].forEach((el,i)=>{if(typeof returnView.details?.[i]==='boolean')el.open=returnView.details[i];});
    [...content.querySelectorAll('pre')].forEach((el,i)=>{const position=returnView.codeScroll?.[i];if(position){el.scrollLeft=position.left;el.scrollTop=position.top;}});
    if(Number.isFinite(returnView.scrollTop))content.scrollTop=returnView.scrollTop;
   }
   returnView=null;
  }
  if(returnDraft&&user&&context.topicId===returnDraft.topicId){const saved=returnDraft;returnDraft=null;showMenu(saved,null);}
 }

 async function api(url,options){
  const response=await fetcher(url,{credentials:'same-origin',cache:'no-store',signal:AbortSignal.timeout(12000),...options});
  const data=await response.json();if(!response.ok){const error=Error(data.error||'Запрос не выполнен.');error.status=response.status;throw error;}return data;
 }
 async function session(){
  try{const data=await api('/api/auth');user=data.user;configured=data.configured;}
  catch{user=null;configured=false;}
  authLoaded=true;renderAccounts();
 }
 function renderAccounts(){
  direct?.refresh();
  accounts.forEach((element,index)=>{
   element.innerHTML=accountMarkup(user,'atlas-account-menu-'+index,!authLoaded,{reader:element.classList.contains('reader-account')});
   element.hidden=!element.innerHTML;
   const popup=element.querySelector('[popover]'),trigger=element.querySelector('.account-trigger');
   element.querySelector('img')?.addEventListener('error',event=>{event.target.hidden=true;},{once:true});
   popup?.addEventListener('beforetoggle',event=>{
    trigger.setAttribute('aria-expanded',String(event.newState==='open'));
    if(event.newState!=='open')return;
    const rect=trigger.getBoundingClientRect(),width=Math.min(320,scope.innerWidth-24);
    popup.style.width=width+'px';popup.style.left=Math.max(12,Math.min(rect.right-width,scope.innerWidth-width-12))+'px';
    popup.style.top=Math.max(12,Math.min(rect.bottom+8,scope.innerHeight-280))+'px';
   });
  });
 }
 async function logout(){
  await api('/api/auth',{method:'POST'});
  user=null;notes=[];runs=[];next=null;queueGeneration++;loading=false;commentDrafts.clear();clearTimeout(pollTimer);renderAccounts();renderActivity();
  direct?.reset();
  if(queue.open)queueShell();notifications?.refresh();
 }
 accounts.forEach(element=>{
 element.addEventListener('keydown',event=>{if(event.key==='Escape'&&element.querySelector(':popover-open'))event.stopPropagation();});
 element.addEventListener('click',async event=>{
  const button=event.target.closest('button');if(!button)return;
  if(button.hasAttribute('data-account-login')){
   button.disabled=true;if(!authLoaded||!configured)await session();
   if(user)return;
   if(configured){remember();scope.location.assign(loginURL());}
   else{button.disabled=false;announce('Вход сейчас недоступен. Попробуй обновить страницу.');}return;
  }
  if(button.hasAttribute('data-account-notes')){element.querySelector('[popover]').hidePopover();await openQueue();return;}
  if(button.hasAttribute('data-account-logout')){
   button.disabled=true;
   try{await logout();element.querySelector('[data-account-login]')?.focus();announce('Ты вышел из аккаунта.');}
   catch(error){element.querySelector('.account-error').textContent=error.message;button.disabled=false;}
  }
 });
 });
 renderAccounts();
 const topicButton=doc.getElementById('reader-feedback');
 topicButton?.addEventListener('click',async()=>{
  if(!context||context.tab==='note')return;
  const data=topicSnapshot(context);
  if(!authLoaded)await session();
  if(reader.open&&context?.topicId===data.topicId)showMenu(data,null);
 });
 function closeMenu(){if(saving)return;menu.close();selected?.classList.remove('editorial-selected');selected=null;draft=null;menuDeletion=null;setHover(null);}
 function showMenu(data,block){
  draft={...data,requestId:scope.crypto.randomUUID()};selected=block;setHover(null);block?.classList.add('editorial-selected');
  const wholeTopic=data.tab==='topic';
  const targets=scope.AtlasReaderEdit?.fragmentTargets(block,content)||[];
  menuDeletion=!wholeTopic&&targets.length?{targets,context:{...context}}:null;
  const canDelete=Boolean(menuDeletion&&direct?.canRemove());
  const deletion=user?.isModerator&&!wholeTopic?`<button type="button" class="editorial-delete" data-editorial-delete${canDelete?'':` disabled title="${menuDeletion?'Сначала заверши сохранение предыдущей правки.':'Этот фрагмент нельзя удалить отдельно. Выбери абзац или раздел через Cmd+Shift.'}"`}>Удалить</button>`:'';
  menu.setAttribute('aria-label',wholeTopic?'Замечание по разделу':'Замечание к тексту');
  menu.innerHTML=`<header><h2>${wholeTopic?'Замечание по разделу':'Доработать текст'}</h2><button type="button" data-editorial-close aria-label="Закрыть замечание">×</button></header>
   ${wholeTopic?`<p class="editorial-topic">Раздел: <strong>${esc(draft.quote)}</strong></p>`:`<blockquote>${esc(draft.quote)}</blockquote>`}
   ${user?`<p>${user.isModerator?(canDelete?'«Удалить» сразу сохранит правку без агента; ⌘Z отменит удаление. Остальные действия передают инструкцию агенту.':'Твоя инструкция будет передана агенту. Ход работы и результат появятся в модерации.'):'Краткое описание нерешённого замечания будет видно всем. Решение примет владелец атласа.'}</p>${wholeTopic?(user.isModerator?'<div class="editorial-actions"><button type="button" data-editorial-improve>Улучшить раздел</button></div>':''):`<div class="editorial-actions"><button data-editorial-action="clarify">Расшифровать</button><button data-editorial-action="rewrite">Переписать по-человечески</button>${deletion}</div>`}
   <form><label for="editorial-instruction">${wholeTopic?'Что нужно добавить или изменить?':'Своя инструкция'}</label><textarea id="editorial-instruction" rows="${wholeTopic?5:3}" maxlength="3000" placeholder="${wholeTopic?'Например: не хватает графиков. Покажи на схеме, как выбираются ближайшие соседи и как от их числа зависит ответ.':'Например: объясни на примере сотрудников, без сокращений'}">${esc(data.instruction||'')}</textarea><button class="editorial-primary" type="submit">${wholeTopic?'Отправить замечание':'Отправить инструкцию'}</button></form>`:
   `<p>Чтобы оставить замечание, войди в свой аккаунт. Читать материалы можно без входа.</p>${loginHTML()}`}
   <p class="editorial-message" role="status"></p>`;
  const rect=block?.getBoundingClientRect(),width=Math.min(1000,scope.innerWidth-32);
  menu.style.width=width+'px';menu.style.left=Math.max(16,Math.min(rect?.left??(scope.innerWidth-width)/2,scope.innerWidth-width-16))+'px';
  menu.style.top=Math.max(16,Math.min((rect?.top??120)+35,scope.innerHeight-620))+'px';
  if(!menu.open)menu.showModal();
 }
 async function submit(action,instructionOverride){
  if(saving||!draft||!user)return;
  const input=menu.querySelector('textarea'),instruction=instructionOverride??(input?.value.trim()||''),message=menu.querySelector('.editorial-message');
  if(action==='custom'&&!instruction){message.textContent='Напиши, что нужно изменить.';input.focus();return;}
  if(instruction.length>3000){message.textContent='Сократи пожелания: вместе с поручением они должны занимать не больше 3000 символов.';input.focus();return;}
  saving=true;menu.querySelectorAll('button').forEach(b=>b.disabled=true);message.textContent='Сохраняю…';
  try{
   const data=await api('/api/editorial-notes',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...draft,action,instruction})});
   try{scope.sessionStorage.removeItem('atlas-editorial-draft');}catch{}
   saving=false;closeMenu();
   if(data.run){runs=[data.run,...runs.filter(r=>r.note_id!==data.run.note_id)];worker=data.worker||worker;renderActivity();}
   announce(data.dispatchError||(user.isModerator?(data.worker?.online?'Задание передано агенту. Ход работы — в модерации.':'Задание в очереди. Исполнитель сейчас недоступен.'):'Замечание сохранено и отправлено на рассмотрение.'));
   schedulePoll(500);notifications?.refresh();
  }catch(error){message.textContent=error.message;}
  finally{saving=false;menu.querySelectorAll('button').forEach(b=>b.disabled=false);}
 }
 menu.addEventListener('click',event=>{
  if(event.target.closest('[data-editorial-improve]')&&user?.isModerator&&draft?.tab==='topic'){submit('custom',improvementInstruction(menu.querySelector('textarea')?.value||''));return;}
  if(event.target.closest('[data-editorial-delete]')){
   if(saving||!user?.isModerator||!menuDeletion||!direct?.canRemove())return;
   const operation=menuDeletion;closeMenu();void direct.removeFragment(operation.targets,operation.context);return;
  }
  if(event.target.closest('[data-editorial-close]'))closeMenu();
  const action=event.target.closest('[data-editorial-action]')?.dataset.editorialAction;if(action)submit(action);
  if(event.target.closest('[data-editorial-login]'))remember();
  if(event.target===menu){const r=menu.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)closeMenu();}
 });
 menu.addEventListener('submit',event=>{event.preventDefault();submit('custom');});
 menu.addEventListener('cancel',event=>{event.preventDefault();event.stopPropagation();if(!saving)closeMenu();});
 menu.addEventListener('input',()=>{if(draft)draft.instruction=menu.querySelector('textarea')?.value||'';});

 const showingChanges=()=>queueView==='history'&&historyKind==='changes'&&user?.isModerator;
 function resetPages(){cursors=[''];pageIndex=0;next=null;notes=[];historyEntries=[];}
 function queueShell(){
  queue.innerHTML=`<header><div><h2>${user?.isModerator?'Модерация замечаний':'Мои замечания'}</h2><p>${user?esc(user.name)+' · @'+esc(user.login):'Для отправки замечаний нужен вход'}</p></div><button data-queue-close aria-label="Закрыть список замечаний">×</button></header>
   ${user?`${user.isModerator?`<nav class="editorial-queue-views" aria-label="Режим списка"><button data-queue-view="inbox" aria-pressed="${queueView==='inbox'}">Новые и ошибки</button><button data-queue-view="history" aria-pressed="${queueView==='history'}">История</button></nav>`:''}
   <div class="editorial-queue-tools">${queueView==='history'&&user.isModerator?`<label>История <select data-history-kind><option value="changes"${historyKind==='changes'?' selected':''}>Изменения материалов</option><option value="notes"${historyKind==='notes'?' selected':''}>Все замечания</option></select></label>`:''}<button data-queue-refresh>Обновить</button>${queueTarget?'<button data-clear-note-filter>Все замечания</button>':''}<button data-logout>Выйти</button></div>
   ${queueView==='history'?`<form class="editorial-history-search" data-history-search><label for="editorial-history-query">Поиск в истории</label><div><input id="editorial-history-query" type="search" maxlength="200" value="${esc(query)}" placeholder="Тема, инструкция или результат"><button type="submit">Найти</button></div></form>`:''}
   <p class="editorial-help">${user.isModerator?(queueView==='inbox'?'Здесь только новые замечания и ошибки. Текущие задания и прежние результаты доступны в истории замечаний.':showingChanges()?'Найди изменение, открой результат или отправь уточнение через «Скорректировать».':'Найди прежнее замечание и уточни, что нужно доработать.'):'Модератор рассмотрит замечание; решение и ход выполнения появятся здесь.'}</p><p class="agent-connection" role="status"></p><div class="editorial-note-list"></div><nav class="editorial-pagination" aria-label="Страницы истории"><button data-page-prev disabled>Назад</button><span data-page-number>Страница 1</span><button data-page-next disabled>Далее</button></nav>`:
   `<p>Пометки сохраняются с твоим именем. Краткие описания нерешённых замечаний видны всем; решение принимает владелец.</p>${loginHTML()}`}
   <p class="editorial-message" role="status"></p>`;
 }
 function renderPagination(){
  const prev=queue.querySelector('[data-page-prev]'),following=queue.querySelector('[data-page-next]');if(!prev)return;
  prev.disabled=loading||pageIndex===0;following.disabled=loading||!next;
  queue.querySelector('[data-page-number]').textContent='Страница '+(pageIndex+1);
 }
 function renderChanges(){
  const list=queue.querySelector('.editorial-note-list');if(!list)return;
  list.innerHTML=historyEntries.length?historyEntries.map(entry=>`<article class="editorial-note"><header><strong>${esc(entry.topic_title||entry.topic_id)}</strong><span class="editorial-status">${esc({agent:'Агент',manual:'Ручная правка',publication:'Публикация'}[entry.kind]||'Публикация')}</span></header>
   <p class="editorial-author">${esc(new Date(entry.created_at).toLocaleString('ru-RU'))} · Версия ${esc(entry.revision)}</p>
   <p>${esc(entry.summary||'Материал обновлён.')}</p>${entry.instruction?`<details><summary>Инструкция</summary><p class="editorial-instruction">${esc(entry.instruction)}</p></details>`:''}
   <div class="editorial-result-actions">${resultLink(entry)}<button data-history-correct="${esc(entry.revision)}">Скорректировать</button></div></article>`).join(''):'<p class="editorial-empty">Изменения не найдены.</p>';
  renderPagination();renderActivity();
 }
 function renderNotes(){
  const list=queue.querySelector('.editorial-note-list');if(!list)return;
  const scrollTop=queue.scrollTop,focused=doc.activeElement?.matches?.('[data-moderator-comment]')?{id:doc.activeElement.closest('[data-note-id]').dataset.noteId,start:doc.activeElement.selectionStart,end:doc.activeElement.selectionEnd}:null;
  list.innerHTML=notes.length?notes.map(note=>`<article class="editorial-note" data-note-id="${esc(note.id)}"><header><div><strong>${esc(note.topic_title||note.topic_id)}</strong><p>${esc(tabNames[note.tab]||note.tab)}${note.anchor?.section?' · '+esc(note.anchor.section):''}</p></div><span class="editorial-status status-${esc(note.status)}">${esc(statuses[note.status])}</span></header>
   <p class="editorial-author">@${esc(note.login||user.login)} · ${esc(new Date(note.created_at).toLocaleString('ru-RU'))}${note.execution_mode==='owner_request'?' · Команда владельца':''}</p>
   ${note.tab==='topic'?'':`<blockquote>${esc(note.quote)}</blockquote>`}<p><strong>${note.tab==='topic'?'Замечание по разделу':esc(labels[note.action])}</strong></p>${note.instruction?`<p class="editorial-instruction">${esc(note.instruction)}</p>`:''}
   ${note.moderator_note?`<p class="editorial-moderator-comment"><strong>Модератор:</strong> ${esc(note.moderator_note)}</p>`:''}
   <div data-run-status>${runMarkup(note.run,worker)}</div>
   <div class="editorial-result-actions">${resultLink(note)}</div>
   <details class="agent-run-history" data-run-history><summary>История запусков</summary><div data-history-content></div></details>
   ${user.isModerator?`<label>Комментарий / что доработать<textarea data-moderator-comment rows="2" maxlength="3000">${esc(commentDrafts.has(note.id)?commentDrafts.get(note.id):note.status==='done'?'':note.moderator_note)}</textarea></label><div class="editorial-actions"><span data-run-action>${executionButton(note,dispatching.has(note.id))}</span><span data-moderation-actions>${moderationButtons(note,dispatching.has(note.id))}</span></div>`:''}</article>`).join(''):'<p class="editorial-empty">Здесь пока нет замечаний.</p>';
  renderPagination();
  queue.scrollTop=scrollTop;
  if(focused){const input=queue.querySelector(`[data-note-id="${focused.id}"] [data-moderator-comment]`);input?.focus({preventScroll:true});input?.setSelectionRange(focused.start,focused.end);}
  renderActivity();
 }
 async function loadNotes(){
  if(!user)return;const generation=++queueGeneration;loading=true;renderPagination();
  const message=queue.querySelector('.editorial-message');message.textContent='Загружаю…';
  try{
   const changes=showingChanges(),params=new URLSearchParams(changes?{q:query}:{scope:user.isModerator?'all':'mine',status:queueView==='inbox'?'inbox':'all',q:query,limit:'20'});
   if(cursors[pageIndex])params.set('before',cursors[pageIndex]);if(queueTarget&&!changes)params.set('id',queueTarget);
   const data=await api((changes?'/api/editorial-history?':'/api/editorial-notes?')+params);
   if(generation!==queueGeneration)return;next=data.next;lastQueueLoad=Date.now();
   if(changes){notes=[];historyEntries=data.entries;renderChanges();}
   else{historyEntries=[];notes=data.notes;worker=data.worker||worker;renderNotes();}
   message.textContent='';
  }catch(error){if(generation===queueGeneration){notes=[];historyEntries=[];next=null;if(showingChanges())renderChanges();else renderNotes();message.textContent=error.message;}}
  finally{if(generation===queueGeneration){loading=false;renderPagination();}}
 }
 async function openQueue(activity=false,noteId=''){
  setHover(null);if(!authLoaded)await session();queueTarget=noteId;queueView=noteId||!user?.isModerator?'history':'inbox';historyKind=noteId||!user?.isModerator?'notes':'changes';query='';resetPages();queueShell();
  if(!queue.open)queue.showModal();await loadNotes();schedulePoll(500);
 }

 async function openNotification(id){
  const selection=++notificationSelection;
  if(!authLoaded)await session();
  if(selection!==notificationSelection)return;
  if(user?.isModerator)return openQueue(true,id);
  const generation=++publicNoteGeneration;
  const heading='<header><h2>Замечание по разделу</h2><button type="button" data-public-note-close aria-label="Закрыть замечание">×</button></header>';
  publicDialog.innerHTML=heading+'<p role="status">Загружаю…</p>';
  if(!publicDialog.open)publicDialog.showModal();
  try{
   const {note}=await api('/api/editorial-notifications?note='+encodeURIComponent(id));
   if(generation!==publicNoteGeneration||!publicDialog.open)return;
   publicDialog.innerHTML=heading+`<h3>${esc(note.topic_title||note.topic_id)}</h3><p class="editorial-author">${esc(tabNames[note.tab]||note.tab)} · ${esc(statuses[note.status]||note.status)}</p><p class="public-note-summary">${esc(note.summary)}</p><p class="editorial-help">Решения по замечаниям принимает владелец атласа.</p>${resultLink(note)}`;
   publicDialog.querySelector('[data-public-note-close]')?.focus({preventScroll:true});
  }catch(error){if(generation===publicNoteGeneration&&publicDialog.open)publicDialog.innerHTML=heading+`<p role="status">${esc(error.message)}</p>`;notifications?.refresh();}
 }

 async function loadHistory(details,note){
  if(!note||!user)return;
  const key=[note.run?.id,note.run?.status].join(':'),body=details.querySelector('[data-history-content]');
  if(details.dataset.loaded===key||details.dataset.loading)return;
  details.dataset.loading='true';
  if(!body.innerHTML)body.textContent='Загружаю историю…';
  try{
   const userId=user.id,data=await api('/api/agent-runs?history='+encodeURIComponent(note.id));
   if(user?.id!==userId)return;
   body.innerHTML=data.runs.length?data.runs.map(run=>`<article><p>${esc(new Date(run.created_at).toLocaleString('ru-RU'))}</p>${run.instruction?`<p><strong>Уточнение:</strong> ${esc(run.instruction)}</p>`:''}${runMarkup(run,worker)}</article>`).join(''):'Запусков пока нет.';
   details.dataset.loaded=key;
  }catch(error){body.textContent=error.message;}
  finally{delete details.dataset.loading;}
 }
 queue.addEventListener('toggle',event=>{
  if(event.target.matches('[data-run-history]')&&event.target.open){
   const note=notes.find(n=>n.id===event.target.closest('[data-note-id]').dataset.noteId);loadHistory(event.target,note);
  }
 },true);
 queue.addEventListener('cancel',event=>{event.stopPropagation();});
 queue.addEventListener('close',()=>{queueGeneration++;loading=false;});
 queue.addEventListener('input',event=>{if(event.target.matches('[data-moderator-comment]'))commentDrafts.set(event.target.closest('[data-note-id]').dataset.noteId,event.target.value);});
 queue.addEventListener('change',event=>{if(event.target.matches('[data-history-kind]')){historyKind=event.target.value;queueTarget='';resetPages();queueShell();loadNotes();}});
 queue.addEventListener('submit',event=>{if(event.target.matches('[data-history-search]')){event.preventDefault();query=queue.querySelector('[type="search"]').value.trim();queueTarget='';resetPages();loadNotes();}});
 queue.addEventListener('click',async event=>{
  const button=event.target.closest('button');if(!button)return;
  if(button.hasAttribute('data-queue-close')){queue.close();return;}
  if(button.hasAttribute('data-queue-refresh')){loadNotes();notifications?.refresh();return;}
  if(button.dataset.queueView){queueView=button.dataset.queueView;historyKind='changes';queueTarget='';query='';resetPages();queueShell();loadNotes();return;}
  if(button.hasAttribute('data-clear-note-filter')){queueTarget='';resetPages();queueShell();loadNotes();return;}
  if(button.hasAttribute('data-page-prev')){if(!loading&&pageIndex>0){pageIndex--;loadNotes();}return;}
  if(button.hasAttribute('data-page-next')){if(!loading&&next){cursors=cursors.slice(0,pageIndex+1);cursors.push(next);pageIndex++;loadNotes();}return;}
  if(button.dataset.historyCorrect){
   const entry=historyEntries.find(item=>item.revision===button.dataset.historyCorrect);if(!entry||!user?.isModerator)return;
   queue.close();showMenu({topicId:entry.topic_id,tab:'topic',contentRevision:entry.current_revision,
    quote:(entry.topic_title+' · Версия '+entry.revision+' · '+entry.summary).slice(0,6000),
    anchor:{methodId:'',panel:'',section:'История · версия '+entry.revision,block:''}},null);return;
  }
  if(button.hasAttribute('data-logout')){
   button.disabled=true;
   try{await logout();}catch(error){queue.querySelector('.editorial-message').textContent=error.message;button.disabled=false;}return;
  }
  if(button.hasAttribute('data-execute')){
   const card=button.closest('[data-note-id]'),note=notes.find(n=>n.id===card.dataset.noteId);
   if(!note||dispatching.has(note.id)||activeRun(note.run))return;
   const kind=button.dataset.execute||'execute',input=card.querySelector('textarea'),comment=input.value.trim()||(note.run?.status==='failed'?(note.run.instruction||note.moderator_note||''):'');
   if(kind==='rework'&&!comment){queue.querySelector('.editorial-message').textContent='Напиши, что нужно доработать.';input.focus();return;}
   const key='atlas-execute:'+user.id+':'+note.id+':'+note.version+':'+kind;
   let requestId;try{requestId=scope.sessionStorage.getItem(key);}catch{}
   requestId=requestId||scope.crypto.randomUUID();try{scope.sessionStorage.setItem(key,requestId);}catch{}
   dispatching.add(note.id);button.disabled=true;button.textContent='Отправляю агенту…';queue.querySelector('.editorial-message').textContent='';
   try{
    const data=await api('/api/agent-runs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({noteId:note.id,version:note.version,requestId,kind,comment})});
    try{scope.sessionStorage.removeItem(key);}catch{}
    note.run=data.run;if(activeRun(data.run))note.status='accepted';note.moderator_note=comment;worker=data.worker||worker;runs=[data.run,...runs.filter(r=>r.note_id!==note.id)];
    if(input.value.trim()===comment){input.value='';commentDrafts.set(note.id,'');}
    if(queueView==='inbox')await loadNotes();else updateRunCards();renderActivity();schedulePoll(500);notifications?.refresh();
    announce(worker.online?'Задание передано агенту.':'Задание сохранено в очереди. Исполнитель сейчас недоступен.');
   }catch(error){queue.querySelector('.editorial-message').textContent=error.message;}
   finally{dispatching.delete(note.id);updateRunCards();}
   return;
  }
  if(button.dataset.moderate){
   const card=button.closest('[data-note-id]'),note=notes.find(n=>n.id===card.dataset.noteId);
   card.querySelectorAll('button').forEach(b=>b.disabled=true);
   try{
    await api('/api/editorial-notes',{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:note.id,version:note.version,status:button.dataset.moderate,comment:card.querySelector('textarea').value})});
    await loadNotes();notifications?.refresh();
   }catch(error){queue.querySelector('.editorial-message').textContent=error.message;card.querySelectorAll('button').forEach(b=>b.disabled=false);}
  }
 });
 function renderActivity(){
  const active=runs.filter(activeRun),running=active.filter(r=>r.status==='running').length;
  const title=running?`Агент выполняет · ${running}`:active.length?`В очереди · ${active.length}`:worker.online?'Агент готов':'Агент недоступен';
  activityButtons.forEach(button=>{button.hidden=!user||(!user.isModerator&&!active.length);button.textContent=title;button.classList.toggle('is-running',running>0);button.title='Открыть задания и результаты';});
  const connection=queue.querySelector('.agent-connection');if(connection)connection.textContent=worker.online?'Агент подключён. Статусы обновляются автоматически.':'Агент сейчас недоступен. Новые задания сохраняются в очереди.';
 }
 function updateRunCards(){
  if(!queue.open)return;
  for(const note of notes){
   const card=queue.querySelector(`[data-note-id="${note.id}"]`);if(!card)continue;
   const statusElement=card.querySelector('[data-run-status]'),action=card.querySelector('[data-run-action]');
   const markup=runMarkup(note.run,worker);if(statusElement.innerHTML!==markup)statusElement.innerHTML=markup;
   const state=[note.status,note.run?.status||'',dispatching.has(note.id)].join(':');
   if(action&&action.dataset.state!==state){action.innerHTML=executionButton(note,dispatching.has(note.id));action.dataset.state=state;}
   const moderation=card.querySelector('[data-moderation-actions]');if(moderation&&moderation.dataset.state!==state){moderation.innerHTML=moderationButtons(note,dispatching.has(note.id));moderation.dataset.state=state;}
   const label=card.querySelector('.editorial-status');label.textContent=statuses[note.status]||note.status;label.className='editorial-status status-'+note.status;
   const history=card.querySelector('[data-run-history]');if(history?.open)loadHistory(history,note);
  }
 }
 function schedulePoll(delay){clearTimeout(pollTimer);if(user)pollTimer=setTimeout(pollRuns,delay);}
 async function pollRuns(){
  if(!user||pollBusy)return;
  if(doc.hidden){schedulePoll(10000);return;}
  const userId=user.id;pollBusy=true;
  try{
   const ids=queue.open?notes.map(n=>n.id):[];
   const data=await fetchRunUpdates(api,ids);
   if(user?.id!==userId)return;
   const completed=data.runs.some(run=>run.status==='completed'&&runs.some(old=>old.id===run.id&&activeRun(old)));
   const newFailure=data.runs.some(run=>run.status==='failed'&&!runs.some(old=>old.id===run.id&&old.status==='failed'));
   runs=data.runs;worker=data.worker;
   for(const note of notes){const run=runs.find(r=>r.note_id===note.id);if(run){note.run=run;if(run.note_version)note.version=run.note_version;if(run.note_status)note.status=run.note_status;}}
   if(queue.open&&queueView==='inbox'&&!loading&&(newFailure||completed||Date.now()-lastQueueLoad>15000))await loadNotes();
   else updateRunCards();renderActivity();
   if(completed){announce('Агент завершил работу. Изменения опубликованы.');notifications?.refresh();}
   await syncPublishedContent(scope.ATLAS_CONTENT_CLIENT,runs);
  }catch{if(user?.id===userId){const connection=queue.querySelector('.agent-connection');if(connection)connection.textContent='Не удалось обновить статусы. Повторяем подключение…';}}
  finally{pollBusy=false;schedulePoll(queue.open||runs.some(activeRun)?3000:30000);}
 }
 doc.addEventListener('visibilitychange',()=>{if(!doc.hidden)schedulePoll(100);});
 content.addEventListener('pointermove',event=>{point={x:event.clientX,y:event.clientY};meta=event.metaKey;setHover(meta&&!event.shiftKey&&canPick()?textBlock(event.target,content):null);});
 content.addEventListener('pointerleave',()=>{point=null;setHover(null);});
 doc.addEventListener('keydown',event=>{
  if(event.key==='Meta'&&!event.target.closest?.('input,textarea,[contenteditable="true"]')){meta=true;if(point&&canPick())setHover(textBlock(doc.elementFromPoint(point.x,point.y),content));}
 });
 doc.addEventListener('keyup',event=>{if(event.key==='Meta'||!event.metaKey){meta=false;setHover(null);}});
 scope.addEventListener('blur',()=>{meta=false;setHover(null);});
 content.addEventListener('scroll',()=>{setHover(null);point=null;});
 content.addEventListener('click',async event=>{
  if(!event.metaKey||event.shiftKey||!canPick())return;
  const block=textBlock(event.target,content);if(!block)return;
  event.preventDefault();event.stopImmediatePropagation();
  const data=snapshot(block,content,context);
  if(data.quote.length>6000){announce('Выбери более короткий абзац.');return;}
  if(!authLoaded)await session();
  if(reader.open)showMenu(data,block);
 },true);
 reader.addEventListener('close',()=>{setHover(null);point=null;});
 session().then(async()=>{
  if(user)schedulePoll(100);
  // Wait until the published material has replaced its bundled fallback, so
  // restoring a deep scroll position is not clamped by the shorter old text.
  if(returnView)await scope.ATLAS_CONTENT_CLIENT?.refresh();
  returnReady=true;
  if(user)try{const saved=JSON.parse(scope.sessionStorage.getItem('atlas-editorial-draft')||'null');if(saved?.quote&&saved?.topicId){scope.sessionStorage.removeItem('atlas-editorial-draft');returnDraft=saved;}}catch{}
  restoreAfterLogin();
 });
 return {update(value){context=value;direct?.update(value);setHover(null);if(topicButton)topicButton.hidden=!value||value.tab==='note';queueMicrotask(restoreAfterLogin);}};
}
const api={improvementInstruction,textBlock,snapshot,topicSnapshot,accountMarkup,loginReturnTo,runMarkup,executionButton,resultLink,syncPublishedContent,moderationButtons,fetchRunUpdates,create};
if(typeof module!=='undefined'&&module.exports)module.exports=api;
else{scope.AtlasEditorial=api;scope.ATLAS_EDITORIAL=create();}
})(typeof window!=='undefined'?window:globalThis);
