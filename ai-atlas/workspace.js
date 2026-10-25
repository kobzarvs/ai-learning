(function(scope){
'use strict';
function create(o){
 const {index,rich,esc,toast,getView,setView,closeDetail,hideTerm}=o,C=scope.AtlasWorkspaceCore;
 const $=id=>document.getElementById(id),canvas=$('canvas'),world=$('workspace-world');
 const art=(id,options)=>scope.AtlasArt?.render(id,options)||'';
 const lessons=scope.ATLAS_LESSONS?.topics||{},meta=scope.ATLAS_LESSONS?.meta||{};
 const contentBlocks=new Map();
 let launchMode='local',currentLaunch=null,exerciseLanguage='python';
 const launchDevices={python:'cpu',rust:'cpu',mlx:'metal'};
 const causalExperiment={minutesSaved:4,assignment:'biased'};
 const lessonVisual=id=>scope.AtlasLessonVisuals?.render(id,causalExperiment)||'';
 const storageKey='ai-atlas-workspaces-v1';let boards={},active=null,stack=[],views=new Map(),selection=new Set(),selectedGroup=null,drag=null,suppressClick=false,lasso=false,edit=null,readerCard=null,undo=[];
 try{const saved=localStorage.getItem(storageKey);if(saved)boards=C.parseImport(saved).boards;}catch{toast('Не удалось прочитать сохранённый холст.');}
 const board=()=>boards[active];
 function save(){try{localStorage.setItem(storageKey,JSON.stringify({version:1,boards}));$('space-status').textContent='Сохранено в этом браузере';}catch{$('space-status').textContent='Не удалось сохранить изменения в браузере';toast('Не удалось сохранить изменения в браузере. Не закрывай вкладку, пока не освободишь место.');}}
 function checkpoint(){undo.push({id:active,board:structuredClone(board())});if(undo.length>30)undo.shift();$('space-undo').disabled=false;}
 function redrawSelection(){world.querySelectorAll('[data-card]').forEach(el=>el.classList.toggle('is-selected',selection.has(el.dataset.card)));world.querySelectorAll('[data-group]').forEach(el=>el.classList.toggle('is-selected',el.dataset.group===selectedGroup));$('space-selection').textContent=selection.size?`Выбрано: ${selection.size}`:selectedGroup?'Выбрана группа':'Перетаскивай за ⠿ · Ctrl / ⌘ + клик — выбрать';}
 function render(){
  if(!active)return;
  const b=board();world.innerHTML=b.groups.map(g=>`<section class="space-group" data-group="${esc(g.id)}" style="left:${g.x}px;top:${g.y}px;width:${g.w}px;height:${g.h}px"><header><button class="group-handle" data-drag-group="${esc(g.id)}" title="Переместить группу">⠿</button><button class="group-name" data-select-group="${esc(g.id)}">${esc(g.title)}</button><button data-group-menu="${esc(g.id)}" title="Изменить группу" aria-label="Изменить группу ${esc(g.title)}">•••</button></header></section>`).join('')+b.cards.map(c=>{
   const topic=index.get(c.ref),lesson=lessons[c.ref];
   const title=c.type==='article'?'Теория':c.type==='topic'?(topic?.name||c.title):c.title;
   const body=({topic:topic?.what,article:topic?.what,exercise:lesson?.exercise,example:lesson?.goal,rust:lesson?.rust?.goal})[c.type]??c.body;
   const type={article:'Статья',topic:'Подпространство',example:'Python + PyTorch',rust:'Rust + Burn',exercise:'Практика',note:'Моя заметка'}[c.type];
   return `<article class="space-card type-${c.type}" data-card="${esc(c.id)}" style="left:${c.x}px;top:${c.y}px;width:${c.w}px;height:${c.h}px"><header><span>${type}</span><button data-drag-card="${esc(c.id)}" aria-label="Переместить ${esc(title)}" title="Перетащи карточку. Стрелки — сдвиг на 20, Alt + стрелки — на 100">⠿</button></header><button class="space-card-body" data-read="${esc(c.id)}"><div class="space-card-heading">${art(c.type==='note'?'projects':c.ref,{decorative:true})}<h2>${c.type==='note'?esc(c.title):rich(title)}</h2></div><p>${c.type==='note'?esc(c.body):rich(body)}</p><span>${c.type==='topic'?'Войти в пространство':'Открыть'} <b>↗</b></span></button></article>`;
  }).join('')+'<div id="selection-rect" class="selection-rect" hidden></div>';
  redrawSelection();
 }
 function enter(id,{remember=true,hash=true}={}){
  if(!index.has(id)||id==='root')return false;hideTerm();closeDetail();closeReader();
  if(active===id)return true;
  views.set(active||'root',{...getView()});if(remember)stack.push(active);
  active=id;if(!boards[id])boards[id]=C.defaultBoard(index.get(id),lessons);
  selection.clear();selectedGroup=null;lasso=false;drag=null;
  $('world').hidden=true;world.hidden=false;$('space-tools').hidden=false;$('space-back').hidden=false;canvas.classList.add('in-space');canvas.classList.remove('lasso-mode');
  $('breadcrumb-current').innerHTML=rich(index.get(id).name);$('space-lasso').setAttribute('aria-pressed','false');
  render();setView(views.get(id)||{x:32,y:24,scale:1});
  if(hash)history.replaceState(null,'','#'+encodeURIComponent(id));save();return true;
 }
 function leave({restore=true}={}){if(!active)return;views.set(active,{...getView()});active=null;stack=[];selection.clear();drag=null;world.hidden=true;$('space-tools').hidden=true;$('space-back').hidden=true;canvas.classList.remove('in-space','lasso-mode');$('world').hidden=false;closeReader();if(restore&&views.has('root'))setView(views.get('root'));}
 function back(){const target=stack.pop();if(target){enter(target,{remember:false});}else{const rootView=views.get('root');leave();o.returnToMap(rootView);}}
 function fit(){if(!active)return;setView(scope.AtlasCore.fitBox(C.bounds([...board().cards,...board().groups]),canvas.clientWidth,canvas.clientHeight,80,1));}
 function point(e){const r=canvas.getBoundingClientRect(),v=getView();return{x:(e.clientX-r.left-v.x)/v.scale,y:(e.clientY-r.top-v.y)/v.scale};}
 function stop(e){e.preventDefault();e.stopPropagation();}
 function pointerDown(e,panMode){
  suppressClick=false;if(!active||e.button!==0||panMode==='pan'||e.shiftKey)return false;
  const c=e.target.closest('[data-card]'),g=e.target.closest('[data-drag-group]');
  if((e.ctrlKey||e.metaKey)&&c){selection.has(c.dataset.card)?selection.delete(c.dataset.card):selection.add(c.dataset.card);selectedGroup=null;redrawSelection();suppressClick=true;stop(e);return true;}
  const handle=e.target.closest('[data-drag-card]');
  if(handle||g){
   checkpoint();if(handle&&!selection.has(handle.dataset.dragCard))selection=new Set([handle.dataset.dragCard]);
   if(g){selectedGroup=g.dataset.dragGroup;selection.clear();}
   drag={pointer:e.pointerId,start:point(e),last:point(e),type:g?'group':'cards',id:g?.dataset.dragGroup,moved:false};redrawSelection();
  }else if(lasso&&!e.target.closest('button,a')){const p=point(e);selection.clear();selectedGroup=null;drag={pointer:e.pointerId,start:p,last:p,type:'lasso',moved:false};}
  else return false;
  stop(e);canvas.setPointerCapture(e.pointerId);return true;
 }
 function pointerMove(e){
  if(!drag||drag.pointer!==e.pointerId)return false;const p=point(e),dx=p.x-drag.last.x,dy=p.y-drag.last.y;
  drag.moved||=Math.hypot(p.x-drag.start.x,p.y-drag.start.y)>3;drag.last=p;
  if(drag.type==='lasso'){const r={x:drag.start.x,y:drag.start.y,w:p.x-drag.start.x,h:p.y-drag.start.y};selection=new Set(C.selectRect(board().cards,r));const el=$('selection-rect');el.hidden=false;Object.assign(el.style,{left:Math.min(r.x,r.x+r.w)+'px',top:Math.min(r.y,r.y+r.h)+'px',width:Math.abs(r.w)+'px',height:Math.abs(r.h)+'px'});redrawSelection();}
  else{if(drag.type==='group')C.moveGroup(board(),drag.id,dx,dy);else for(const c of board().cards)if(selection.has(c.id)){c.x+=dx;c.y+=dy;}render();}
  stop(e);return true;
 }
 function pointerUp(e){
  if(!drag||drag.pointer!==e.pointerId)return false;suppressClick=drag.moved||drag.type==='lasso';
  if(drag.type==='cards'&&drag.moved){for(const c of board().cards.filter(c=>selection.has(c.id))){const group=[...board().groups].reverse().find(g=>c.x+c.w/2>=g.x&&c.x+c.w/2<=g.x+g.w&&c.y+c.h/2>=g.y&&c.y+c.h/2<=g.y+g.h);const old=c.groupId;c.groupId=group?.id||null;if(old)C.fitGroup(board(),old);if(group)C.fitGroup(board(),group.id);}}
  if(drag.type!=='lasso'&&drag.moved)save();drag=null;render();return true;
 }
 function download(name,content,type='text/plain;charset=utf-8'){const url=URL.createObjectURL(new Blob([content],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
 const sectionKinds={
  'Задача':'target','Что будем делать руками':'target','Что покажет этот код':'target',
  'Порядок работы':'steps','Как это устроено':'mechanism','Где пригодится':'target',
  'Представь такую задачу':'example','Пример применения':'example','Что важно понимать':'info','На что смотреть в результате':'result',
  'Измени одно условие':'experiment','Проверь себя':'experiment','Как оценить результат':'result','Границы этого опыта':'info',
  'Подготовка и запуск':'terminal','Разбор':'book','Ожидаемый результат':'result',
  'Попробуй изменить':'experiment','Что пример не доказывает':'info','Первоисточники':'sources','Темы внутри':'topics'
 };
 const taskTitles=new Set(['Задача','Что будем делать руками','Что покажет этот код','Измени одно условие']);
 const sectionHeading=title=>`<h2>${scope.AtlasUI?.icon(sectionKinds[title])||''}<span>${esc(title)}</span></h2>`;
 function textSection(title,body){return body?`<section class="reading-section${taskTitles.has(title)?' task-brief':''}">${sectionHeading(title)}<p>${rich(body,true)}</p></section>`:'';}

 function codeBlock(code,id,label='Python',language={cargo:'toml',python:'python',rust:'rust'}[id]||'bash'){
  const highlighted=scope.AtlasUI?.highlight(code,language)||esc(code);
  return `<div class="code-block"><header><span>${esc(label)}</span><div><button data-copy-code="${id}">Копировать</button><button data-download-code="${id}">Скачать ↓</button></div></header><pre tabindex="0" class="language-${language}" aria-label="Код: ${esc(label)}"><code class="language-${language}">${highlighted}</code></pre></div>`;
 }
 function launchGuide(id,lesson,mode){
  currentLaunch=scope.AtlasLaunch.plan(id,lesson,meta,{language:mode,mode:launchMode,device:launchDevices[mode]});
  const choice=`<div class="launch-choice" role="group" aria-label="Способ запуска">${[['local','На компьютере'],['container','Dev Container']].map(([key,label])=>`<button data-launch-mode="${key}" aria-pressed="${launchMode===key}">${label}</button>`).join('')}</div>`;
  const devices=`<div class="launch-devices" role="group" aria-label="Устройство вычислений">${scope.AtlasLaunch.deviceChoices[mode].map(([key,label])=>`<button data-launch-device="${key}" aria-pressed="${launchDevices[mode]===key}">${label}</button>`).join('')}</div>`;
  const steps=currentLaunch.steps.map(step=>`<li><h3>${esc(step.title)}</h3><p>${rich(step.body,true)}</p>${(step.links||[]).map(([label,url])=>`<a class="setup-link" href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(label)} ↗</a>`).join('')}${step.action?`<a class="inline-primary container-open" href="${esc(step.action.url)}" target="_blank" rel="noopener noreferrer">${esc(step.action.label)} ↗</a>`:''}${(step.blocks||[]).map(b=>codeBlock(b.code,b.id,b.label,b.language)).join('')}</li>`).join('');
  return `<section class="reading-section launch-guide">${sectionHeading('Подготовка и запуск')}${choice}${devices}${currentLaunch.prerequisite?`<p class="setup-prerequisite">${rich(currentLaunch.prerequisite,true)}</p>`:''}${currentLaunch.blocked?`<p class="setup-prerequisite" role="status">${esc(currentLaunch.blocked)}</p>`:''}<ol class="setup-steps">${steps}</ol>${currentLaunch.source?`<details class="container-source"><summary>Посмотреть исходный код этого примера</summary>${codeBlock(currentLaunch.source.code,currentLaunch.source.id,currentLaunch.source.label,currentLaunch.source.language)}</details>`:''}</section>`;
 }

 function read(card,tab){
  if(card.type==='topic'){enter(card.ref);return;}
  if(readerCard?.ref!==card.ref)exerciseLanguage='python';
  readerCard=card;const n=index.get(card.ref)||index.get(active),l=lessons[n.id],mode=tab||({example:'python',rust:'rust',exercise:'exercise',note:'note'}[card.type]||'article');
  contentBlocks.clear();
  if(['python','rust','mlx'].includes(mode))exerciseLanguage=mode;
  $('reader-title').innerHTML=art(card.type==='note'?'projects':n.id,{decorative:true})+'<span>'+esc(card.type==='note'?card.title:n.name)+'</span>';
  $('reader-tabs').innerHTML=card.type==='note'?'':[['article','Теория'],...(l?.guide?[['methods','Методы и случаи']]:[]),...(l?[['python','Python'],...(l.rust?[['rust','Rust + Burn']]:[]),...(l.mlx?[['mlx','Python + MLX']]:[]),['exercise',n.id==='causal'?'Эксперимент':'Задание']]:[])].map(([key,label])=>`<button data-reader-tab="${key}" aria-pressed="${key===mode}">${label}</button>`).join('');
  let html='';
  if(mode==='note'){html=`<div class="personal-note">${esc(card.body||'Пока пусто. Добавь свои наблюдения.').replace(/\n/g,'<br>')}</div><button class="inline-primary" data-edit-note="${esc(card.id)}">Редактировать заметку</button>`;}
  else if(mode==='article'){
   const normalize=text=>String(text||'').replace(/\s+/g,' ').trim();
   const sharedExample=normalize(n.tasks)===normalize(n.example);
   html=`<div class="reading-meta">${esc(n.status||'Область знаний')} · ${l?'Материал и пример':'Путеводитель по области'}</div><p class="reading-lead">${rich(n.what,true)}</p>${textSection('Как это устроено',n.how)}${textSection('Где пригодится',sharedExample?'':n.tasks)}${textSection(sharedExample?'Пример применения':'Представь такую задачу',n.example)}${textSection('Что важно понимать',n.note)}`;
   if(l)html+=textSection('Что будем делать руками',l.goal);
   if(n.children)html+=`<section class="reading-section">${sectionHeading('Темы внутри')}<div class="reading-topics">${n.children.map(c=>`<button data-space-enter="${c.id}"><div class="reading-topic-heading">${art(c.id,{decorative:true})}<b>${rich(c.name)}</b></div><span>${rich(c.what)}</span><span>Открыть пространство ↗</span></button>`).join('')}</div></section>`;
  }else if(l?.guide&&mode==='methods'){
   html=scope.AtlasLessonGuide.render(l.guide);
  }else if(l&&mode==='exercise'){
   const exerciseExample=l[exerciseLanguage]||l;
   const exerciseLabel={python:'Python',rust:'Rust + Burn',mlx:'Python + MLX'}[exerciseLanguage];
   html=(n.id==='causal'?lessonVisual(n.id)+textSection('Проверь себя','Отключи экономию и выбери «Чаще на сложных». Сравни распределения сложности: почему у групп разное среднее время, хотя эффект инструкции равен нулю? Затем выбери случайное назначение: почему распределения сближаются, но разница времени не становится ровно нулевой? Включи экономию 4 минуты и проверь свой прогноз.'):
    textSection('Измени одно условие',exerciseExample.exercise)+`<section class="reading-section">${sectionHeading('Порядок работы')}<ol><li>Перед запуском запиши, как, по твоему предположению, изменится результат.</li><li>Во вкладке ${exerciseLabel} запусти исходный вариант и сохрани вывод.</li><li>Внеси указанное изменение, повтори запуск и сравни результаты.</li><li>Объясни различие: чем наблюдение подтверждает или опровергает твоё предположение?</li></ol></section>`)+`<button class="inline-primary" id="exercise-note">Записать гипотезу и результат</button>`;
  }else if(l&&(mode==='python'||mode==='rust'||mode==='mlx')){
   const rust=mode==='rust',mlx=mode==='mlx',example=rust?l.rust:mlx?l.mlx:l;
   html=`<div class="reading-meta">${rust?'Burn '+meta.burn:mlx?'MLX '+meta.mlx:'Python '+meta.python+' · PyTorch '+meta.torch} · ${esc(example.kind||'Учебный эксперимент')}</div>`+textSection('Что покажет этот код',example.goal||l.goal)+launchGuide(n.id,l,mode);
   html+=textSection('Разбор',example.explanation||l.explanation)+textSection('Ожидаемый результат',example.expected||l.expected)+textSection('Что пример не доказывает',example.limits||l.limits);
   if(example.parity)html+=textSection('Соответствие вариантов',example.parity);
   html+=`<p class="validation-note">${esc(meta.validation||'Версии зафиксированы; результат зависит от данных и случайной инициализации.')} ${meta.validationUrl?`<a href="${esc(meta.validationUrl)}" target="_blank" rel="noopener noreferrer">Посмотреть проверку ↗</a>`:''}</p>`;
  }
  const presentation=scope.ATLAS_PRESENTATION?.[n.id]||{};
  if(card.type!=='note'&&Array.isArray(presentation.sections?.[mode])){
   html=scope.AtlasContent.renderSections(presentation.sections[mode],{rich,
    code:block=>{const id='content-'+contentBlocks.size;contentBlocks.set(id,block.code||'');return codeBlock(block.code||'',id,block.filename||block.language||'Код',block.language||'python');},
    experiment:id=>id==='causal'?lessonVisual('causal'):''});
  }
  const layout=scope.AtlasContent?.layoutValues(presentation.layout?.[mode]||{})||{};
  for(const [key,property] of [['readerWidth','--content-reader-width'],['textWidth','--content-text-width'],['chartsColumns','--content-charts-columns']]){
   $('reader').style.setProperty(property,layout[key]?String(layout[key])+(key==='chartsColumns'?'':'px'):'');
  }
  if(card.type!=='note'&&mode==='article'){
   const sources=[...(n.sources||[]).map(k=>o.sources[k]).filter(Boolean),...(l?.sources||[]),...Object.values(l?.guide?.sources||{})];
   const groups=scope.AtlasContent.groupSources(sources);
   html+=`<section class="reading-section reading-sources">${sectionHeading('Первоисточники')}<div class="source-groups">${groups.map(group=>`<section class="source-group"><h3>${esc(group.title)}</h3><ul>${group.links.map(([name,url])=>`<li><a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(name)} ↗</a></li>`).join('')}</ul></section>`).join('')}</div></section>`;
  }
  const codeMode=['python','rust','mlx'].includes(mode);
  $('reader').classList.toggle('reader-code',codeMode);
  $('reader').classList.toggle('reader-wide',codeMode||(n.id==='causal'&&['exercise','methods'].includes(mode)));
  $('reader-content').innerHTML=html;$('reader-content').scrollTop=0;$('reader').dataset.tab=mode;if(!$('reader').open)$('reader').showModal();hideTerm();
  scope.ATLAS_EDITORIAL?.update({topicId:n.id,title:n.name,tab:mode,revision:presentation.revision||'0'});
  scope.ATLAS_CONTENT_CLIENT?.refresh();
 }
 function refreshLaunch(){
  const content=$('reader-content'),scroll=content.scrollTop,charts=content.querySelector('.lesson-charts'),horizontal=charts?.scrollLeft||0;
  const codeScroll=[...content.querySelectorAll('pre')].map(el=>({left:el.scrollLeft,top:el.scrollTop}));
  const openDetails=[...content.querySelectorAll('details')].map(el=>el.open);
  const guideView=content.querySelector('.causal-guide')?.dataset.guideCurrent;
  const focused=document.activeElement,focusKey=['causalSaving','causalAssignment','launchMode','launchDevice','guideView'].find(key=>focused?.dataset?.[key]!==undefined),focusValue=focusKey?focused.dataset[focusKey]:null;
  read(readerCard,$('reader').dataset.tab);
  if(guideView!==undefined)scope.AtlasLessonGuide.selectView(content,guideView);
  const updated=content.querySelector('.lesson-charts');if(updated)updated.scrollLeft=horizontal;
  [...content.querySelectorAll('pre')].forEach((el,i)=>{if(codeScroll[i]){el.scrollLeft=codeScroll[i].left;el.scrollTop=codeScroll[i].top;}});
  [...content.querySelectorAll('details')].forEach((el,i)=>{if(i<openDetails.length)el.open=openDetails[i];});
  content.scrollTop=scroll;
  if(focusKey)[...content.querySelectorAll('button')].find(button=>button.dataset[focusKey]===focusValue)?.focus({preventScroll:true});
 }
 function closeReader(){hideTerm();if($('reader').open)$('reader').close();readerCard=null;}
 function codeFor(id){return contentBlocks.has(id)?contentBlocks.get(id):currentLaunch?scope.AtlasLaunch.findBlock(currentLaunch,id)?.code||'':'';}

 function editor(mode,id){
  if(!active)return;edit={mode,id};const item=mode==='group'?board().groups.find(g=>g.id===id):board().cards.find(c=>c.id===id);
  $('editor-heading').textContent=mode==='group'?(item?'Изменить группу':'Новая группа'):(item?'Редактировать заметку':'Новая заметка');
  $('editor-title').value=item?.title||'';$('editor-body').value=item?.body||'';$('editor-body-label').hidden=mode==='group';$('editor-group-label').hidden=mode==='group';
  $('editor-group').innerHTML='<option value="">Без группы</option>'+board().groups.map(g=>`<option value="${esc(g.id)}">${esc(g.title)}</option>`).join('');$('editor-group').value=item?.groupId||selectedGroup||'';
  $('editor-delete').hidden=!item;$('editor-delete').textContent=mode==='group'?'Убрать рамку группы':'Удалить заметку';$('note-editor').showModal();$('editor-title').focus();
 }
 scope.AtlasUI?.bindDismiss($('reader'),{close:closeReader});
 scope.AtlasUI?.bindDismiss($('topic-picker'));
 scope.AtlasUI?.bindDismiss($('note-editor'),{editable:true});
 $('note-editor').addEventListener('cancel',e=>e.preventDefault());
 $('editor-form').addEventListener('submit',e=>{e.preventDefault();if(!edit)return;checkpoint();const title=$('editor-title').value.trim(),body=$('editor-body').value,groupId=$('editor-group').value||null;
  if(edit.mode==='group'){const g=board().groups.find(g=>g.id===edit.id);if(g)g.title=title||'Группа';else{const created=C.groupCards(board(),[...selection],title);selectedGroup=created.id;if(!selection.size){const v=getView();created.x=(canvas.clientWidth/2-v.x)/v.scale-200;created.y=(canvas.clientHeight/2-v.y)/v.scale-130;}}}
  else{const c=board().cards.find(c=>c.id===edit.id);if(c){const old=c.groupId;Object.assign(c,{title:title||'Без названия',body,groupId});if(old)C.fitGroup(board(),old);if(groupId)C.fitGroup(board(),groupId);}else{const v=getView();C.addNote(board(),title,body,{x:(canvas.clientWidth/2-v.x)/v.scale-180,y:(canvas.clientHeight/2-v.y)/v.scale-115},groupId);}}
  $('note-editor').close();save();render();if(readerCard?.type==='note'){const c=board().cards.find(c=>c.id===readerCard.id);if(c)read(c);}edit=null;
 });
 $('editor-delete').onclick=()=>{checkpoint();if(edit.mode==='group')C.removeGroup(board(),edit.id);else{const c=board().cards.find(c=>c.id===edit.id);board().cards=board().cards.filter(c=>c.id!==edit.id);if(c?.groupId)C.fitGroup(board(),c.groupId);closeReader();}$('note-editor').close();edit=null;selection.clear();selectedGroup=null;save();render();toast('Удалено. Можно отменить кнопкой ↶.');};
 $('editor-cancel').onclick=()=>$('note-editor').close();$('reader-close').onclick=closeReader;$('reader').addEventListener('close',hideTerm);$('reader-content').addEventListener('scroll',hideTerm);$('space-back').onclick=back;
 $('space-note').onclick=()=>editor('note');$('space-group').onclick=()=>editor('group');
 $('space-lasso').onclick=()=>{lasso=!lasso;$('space-lasso').setAttribute('aria-pressed',String(lasso));canvas.classList.toggle('lasso-mode',lasso);if(lasso)o.selectTool();};
 $('space-undo').onclick=()=>{const last=undo.pop();if(!last)return;boards[last.id]=last.board;if(last.id===active){selection.clear();selectedGroup=null;render();}save();$('space-undo').disabled=!undo.length;};
 $('space-link').onclick=()=>{$('topic-picker-list').innerHTML=[...index.values()].filter(n=>n.parent&&n.id!==active).map(n=>`<button data-add-topic="${n.id}">${rich(n.name)}</button>`).join('');$('topic-picker').showModal();};
 $('topic-picker-close').onclick=()=>$('topic-picker').close();$('topic-picker-search').addEventListener('input',e=>{const q=e.target.value.toLocaleLowerCase('ru');$('topic-picker-list').querySelectorAll('button').forEach(b=>b.hidden=!b.textContent.toLocaleLowerCase('ru').includes(q));});
 document.addEventListener('click',e=>{
  const b=e.target.closest('button');if(!b)return;
  if(b.dataset.read){if(suppressClick||o.panBlocks(e.detail===0))return;const c=board()?.cards.find(c=>c.id===b.dataset.read);if(c)read(c);}
  if(b.dataset.spaceEnter){enter(b.dataset.spaceEnter);}
  if(b.dataset.selectGroup){selectedGroup=b.dataset.selectGroup;selection.clear();redrawSelection();}
  if(b.dataset.groupMenu)editor('group',b.dataset.groupMenu);
  if(b.dataset.editNote)editor('note',b.dataset.editNote);
  if(b.dataset.guideView!==undefined&&readerCard)scope.AtlasLessonGuide.selectView($('reader-content'),b.dataset.guideView);
  if(b.dataset.guideJump&&readerCard)scope.AtlasLessonGuide.jump($('reader-content'),b.dataset.guideJump);
  if(b.dataset.readerTab&&readerCard)read(readerCard,b.dataset.readerTab);
  if(readerCard?.ref==='causal'&&$('reader').dataset.tab==='exercise'){
   if(b.dataset.causalSaving!==undefined&&[0,4,8].includes(Number(b.dataset.causalSaving))){causalExperiment.minutesSaved=Number(b.dataset.causalSaving);refreshLaunch();}
   if(['biased','random'].includes(b.dataset.causalAssignment)){causalExperiment.assignment=b.dataset.causalAssignment;refreshLaunch();}
  }
  if(b.dataset.launchMode&&readerCard){launchMode=b.dataset.launchMode==='container'?'container':'local';refreshLaunch();}
  if(b.dataset.launchDevice&&readerCard){const mode=$('reader').dataset.tab;if(scope.AtlasLaunch.deviceChoices[mode]?.some(([key])=>key===b.dataset.launchDevice)){launchDevices[mode]=b.dataset.launchDevice;refreshLaunch();}}
  if(b.id==='exercise-note'){const n=index.get(readerCard?.ref||active),l=lessons[n.id],example=l[exerciseLanguage]||l;closeReader();editor('note');$('editor-title').value='Опыт: '+n.name;$('editor-body').value='Мой вопрос\n'+example.exercise+'\n\nЧто изменил\n\nЧто получилось\n\nВывод\n';}
  if(b.dataset.copyCode){if(navigator.clipboard){navigator.clipboard.writeText(codeFor(b.dataset.copyCode)).then(()=>{b.textContent='Скопировано ✓';setTimeout(()=>b.textContent='Копировать',1800);},()=>{b.textContent='Используй «Скачать»';});}else b.textContent='Используй «Скачать»';}
  if(b.dataset.downloadCode){const id=b.dataset.downloadCode;const filename=scope.AtlasLaunch.findBlock(currentLaunch,id)?.label.split('/').pop();download(id==='cargo'?'Cargo.toml':id==='rust'?'main.rs':['python','mlx'].includes(id)?filename:id+'.txt',codeFor(id));}
  if(b.dataset.addTopic){checkpoint();const n=index.get(b.dataset.addTopic),v=getView(),c=C.addNote(board(),n.name,n.what,{x:(canvas.clientWidth/2-v.x)/v.scale-180,y:(canvas.clientHeight/2-v.y)/v.scale-115},selectedGroup);c.type='topic';c.ref=n.id;$('topic-picker').close();save();render();}
 });
 canvas.addEventListener('click',e=>{if(active&&suppressClick&&e.detail!==0){e.preventDefault();e.stopImmediatePropagation();}},true);
 document.addEventListener('keydown',e=>{if(!active||e.target.closest('input,textarea,select,dialog')||e.shiftKey)return;const moves={ArrowLeft:[-20,0],ArrowRight:[20,0],ArrowUp:[0,-20],ArrowDown:[0,20]},handle=e.target.closest('[data-drag-card],[data-drag-group]');if(handle&&moves[e.key]){e.preventDefault();checkpoint();const [dx,dy]=moves[e.key].map(v=>v*(e.altKey?5:1));if(handle.dataset.dragGroup)C.moveGroup(board(),handle.dataset.dragGroup,dx,dy);else{const c=board().cards.find(c=>c.id===handle.dataset.dragCard);c.x+=dx;c.y+=dy;if(c.groupId)C.fitGroup(board(),c.groupId);}const cardId=handle.dataset.dragCard,groupId=handle.dataset.dragGroup;save();render();[...world.querySelectorAll('[data-drag-card],[data-drag-group]')].find(x=>cardId?x.dataset.dragCard===cardId:x.dataset.dragGroup===groupId)?.focus();}});
 return {enter,leave,back,fit,pointerDown,pointerMove,pointerUp,get active(){return active;},refreshContent(){render();if(active)$('breadcrumb-current').innerHTML=rich(index.get(active)?.name||'');if(readerCard&&readerCard.type!=='note')refreshLaunch();},updateView(v){if(active)world.style.transform=`translate(${v.x}px,${v.y}px) scale(${v.scale})`;},reset(){if(drag){drag=null;render();save();}}};
}
scope.AtlasSpaces={create};
})(window);
