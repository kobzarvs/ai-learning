(() => {
'use strict';
function readHash(){try{return decodeURIComponent(location.hash.slice(1));}catch{return '';}}
const C=window.AtlasCore,data=window.ATLAS_DATA,index=C.indexTree(data.root),map=C.layout(data.root);
const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const colors={tasks:['#c2e0ca','#e1ebe2','#d1e2d3','#a6c7ab','#315d40'],models:['#d8baf1','#e9e4ee','#e0d6ea','#c7b8d5','#6c4686'],training:['#b8d8eb','#e2eaee','#d4e4ec','#adc8d6','#416e88'],small:['#ead99e','#eeeadc','#e9e1c1','#d3c69b','#85702d'],practice:['#ecc7ad','#efe6df','#ebd9ca','#d5bdaa','#895d3f']};
const css=b=>{const c=colors[b]||colors.models;return `--label:${c[0]};--tint:${c[1]};--nested:${c[2]};--stroke:${c[3]};--accent:${c[3]};--ink:${c[4]}`;};
const links=[['mlp','cnn','Работа с локальными признаками'],['rnn','lstm','Управление памятью'],['cnn','specialblocks','Блоки и архитектуры для зрения'],['ae','vae','Вероятностное скрытое пространство'],['diffusion','flowmatching','Родственные пути генерации'],['ssm','linearattention','Альтернативы полному attention'],['linear','lab-text','Первый классификатор команд'],['cnn','lab-vision','Классификация изображений'],['transfer','lora','Экономная адаптация'],['quant','binary','Предельно низкая точность'],['supervised','lab-table','Обучение по размеченным примерам'],['transformer','lora','Адаптация предобученной модели'],['distill','compactarch','Обучение компактного ученика']];
let route;try{route=C.restoreRoute(localStorage.getItem('ai-atlas-route-v1'),index);}catch{route=new Set();}
let view={scale:.85,x:0,y:0},current='neural',selected=null,toastTimer,lastDetailTrigger=null;
let viewportWidth=$('canvas').clientWidth,viewportHeight=$('canvas').clientHeight;
const pan=C.createPanController(),rich=C.createTermRenderer(data.glossary);
const cardById=new Map(map.cards.map(c=>[c.id,c]));const groupById=new Map(map.groups.map(g=>[g.id,g]));
function updateView(){ viewportWidth=$('canvas').clientWidth;viewportHeight=$('canvas').clientHeight; $('world').style.transform=`translate(${view.x}px,${view.y}px) scale(${view.scale})`;$('zoom-value').textContent=Math.round(view.scale*100)+'%';$('canvas').classList.toggle('overview-mode',view.scale<.25);$('world').style.setProperty('--root-label-scale',Math.min(4,Math.max(1,.43/view.scale)));  }
function boxFor(id){return cardById.get(id)||groupById.get(id);}
function navigate(id,{fit=false,details=false,hash=true}={}){
hideTerm();
const node=index.get(id);if(!node)return;current=id;$('breadcrumb-current').innerHTML=rich(node.name);
if(details)showDetail(id);const box=boxFor(id);const canvas=$('canvas');
if(box){if(fit||cardById.has(id))view=C.fitBox(box,canvas.clientWidth,canvas.clientHeight,45,cardById.has(id)?1.15:1);
else{const scale=1;view={scale,x:Math.max(90,(canvas.clientWidth-box.w)/2)-box.x,y:32-box.y};}updateView();}
if(hash)history.replaceState(null,'','#'+encodeURIComponent(id));
document.querySelectorAll('.nav-child').forEach(b=>b.classList.toggle('active',b.dataset.nav===id));
closeNav();hideSearch();
}
function fitAll(){current='root';const w=Math.max(...map.groups.map(g=>g.x+g.w)),h=Math.max(...map.groups.map(g=>g.y+g.h));view=C.fitBox({x:0,y:-30,w,h:h+30},$('canvas').clientWidth,$('canvas').clientHeight,55,1);updateView();$('breadcrumb-current').textContent='Вся карта';history.replaceState(null,'','#root');closeNav();}
function zoom(factor,x=$('canvas').clientWidth/2,y=$('canvas').clientHeight/2){view=C.zoomAt(view,view.scale*factor,x,y);updateView();}
function renderBoard(){
$('regions').innerHTML=map.groups.map(g=>{const n=index.get(g.id);return `<section class="region ${g.nested?'nested':'root-region'}" style="${css(g.branch)};left:${g.x}px;top:${g.y}px;width:${g.w}px;height:${g.h}px"><button class="region-label" data-nav="${n.id}">${rich(n.name)} <small>${C.countLeaves(n)} тем ↗</small></button><div class="region-intro">${rich(n.what)}</div></section>`;}).join('');
$('nodes').innerHTML=map.cards.map((c,i)=>{const n=index.get(c.id);return `<article class="note" id="card-${n.id}" style="${css(c.branch)};left:${c.x}px;top:${c.y}px;width:${c.w}px;height:${c.h}px"><button class="note-main" data-open="${n.id}" aria-label="${esc(n.name)}: открыть описание"><span class="note-serial">${String(i+1).padStart(3,'0')} / ${rich('AI ATLAS')} <b>${esc(n.status||'Область')}</b></span><h2>${rich(n.name)}</h2>${n.short?`<span class="short">${rich(n.short)}</span>`:''}<p class="description">${rich(n.what)}</p><div class="task-preview"><b>Пример</b> · ${rich(n.example||n.tasks)}</div></button><div class="note-footer"><span>${n.lab?'↗ Есть идея эксперимента':'Открыть конспект ↗'}</span><button data-save="${n.id}" aria-label="${esc(n.name)}: добавить в маршрут" aria-pressed="false">☆</button></div></article>`;}).join('');
$('edges').innerHTML='<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10z" style="fill:#628574;stroke:none;opacity:1"/></marker></defs>'+links.map(([a,b,label])=>{
  const p=cardById.get(a),q=cardById.get(b);if(!p||!q)return '';
  const points=C.routeConnection(map.cards,map.groups,p,q);if(!points)return '';
  return `<path class="connection ${p.branch!==q.branch?'cross-link':''}" data-source="${a}" data-target="${b}" d="${C.roundedRoute(points)}" marker-end="url(#arrow)"><title>${esc(label)}</title></path>`;
}).join('');
$('branch-nav').innerHTML=data.root.children.map(b=>`<div class="nav-branch" style="${css(b.id)}"><button data-nav="${b.id}"><span class="nav-dot"></span>${rich(b.name)}</button>${b.children.some(c=>c.children)?b.children.map(c=>`<button class="nav-child" data-nav="${c.id}">${rich(c.name)}</button>`).join(''):''}</div>`).join('');
updateSaved();
}
function updateSaved(){ $('route-count').textContent=route.size;document.querySelectorAll('[data-save]').forEach(b=>{const saved=route.has(b.dataset.save);b.classList.toggle('saved',saved);b.setAttribute('aria-pressed',String(saved));b.setAttribute('aria-label',index.get(b.dataset.save).name+(saved?': убрать из маршрута':': добавить в маршрут'));if(b.classList.contains('primary-button')){b.classList.toggle('is-saved',saved);b.textContent=saved?'✓ В моём маршруте':'☆ Добавить в маршрут';}else b.textContent=saved?'★':'☆';}); }
function toast(message){clearTimeout(toastTimer);$('toast').textContent=message;$('toast').hidden=false;toastTimer=setTimeout(()=>$('toast').hidden=true,2600);}
function toggleSaved(id){if(route.has(id))route.delete(id);else route.add(id);try{localStorage.setItem('ai-atlas-route-v1',JSON.stringify([...route]));toast(route.has(id)?'Тема добавлена в маршрут':'Тема убрана из маршрута');}catch{toast('Выбор доступен до закрытия страницы. Скачай маршрут.');}updateSaved();if($('route-dialog').open)renderRoute();}
function highlightConnections(id){
  $('edges').querySelectorAll('.connection').forEach(edge=>edge.classList.toggle('highlighted',edge.dataset.source===id||edge.dataset.target===id));
}
function showDetail(id){
  const n=index.get(id);if(!n)return;hideTerm();highlightConnections(id);
  lastDetailTrigger=document.activeElement;selected=id;$('inspector').hidden=false;document.body.classList.add('detail-open');
  $('detail-category').innerHTML=rich(index.get(n.parent)?.name||'ПУТЕВОДИТЕЛЬ');$('detail-body').style=css(n.branch);
  const related=links.filter(l=>l[0]===id||l[1]===id).map(l=>({id:l[0]===id?l[1]:l[0],label:l[2]}));
  const section=(title,text)=>text?`<section class="detail-section"><h3>${title}</h3><p>${rich(text,true)}</p></section>`:'';
  $('detail-body').innerHTML=`<span class="detail-tag">${esc(n.status||'Область знаний')}</span>
    <h2 class="detail-title">${rich(n.name,true)}</h2><p class="detail-intro">${rich(n.what,true)}</p>
    ${section('КАК ЭТО РАБОТАЕТ',n.how)}
    ${n.example?`<section class="example-box"><h3>КОНКРЕТНЫЙ ПРИМЕР</h3><p>${rich(n.example,true)}</p></section>`:''}
    ${section('ЧТО ПОДАЁМ НА ВХОД',n.input)}${section('ЧТО ПОЛУЧАЕМ',n.output)}
    ${section('НА ЧЁМ БУДЕМ УЧИТЬ',n.training)}
    ${n.steps?`<section class="detail-section"><h3>ПЛАН ПЕРВОГО ОПЫТА</h3><ol class="experiment-steps">${n.steps.map(step=>`<li>${rich(step,true)}</li>`).join('')}</ol></section>`:''}
    ${n.lab?`<section class="lab-box"><h3>↗ Попробуем на практике</h3><p>${rich(n.lab,true)}</p></section>`:''}
    ${section('КАК ПРОВЕРИМ РЕЗУЛЬТАТ',n.criteria)}${section('ЧТО УЧЕСТЬ',n.note)}
    <button class="primary-button" data-save="${n.id}"></button>
    ${n.children?`<section class="detail-section related-links"><h3>ВНУТРИ ЭТОЙ ОБЛАСТИ</h3>${n.children.map(c=>`<button data-open="${c.id}">${rich(c.name)} ↗</button>`).join('')}</section>`:''}
    ${related.length?`<section class="detail-section related-links"><h3>СВЯЗАННЫЕ ИДЕИ</h3>${related.map(l=>`<button data-open="${l.id}">${rich(index.get(l.id).name)} ↗<div class="detail-full">${rich(l.label)}</div></button>`).join('')}</section>`:''}
    ${n.sources?.length?`<section class="detail-section sources"><h3>ДОКУМЕНТАЦИЯ И СТАТЬИ</h3>${n.sources.map(k=>data.sources[k]).filter(Boolean).map(([title,url])=>`<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${rich(title)} ↗</a>`).join('')}</section>`:''}`;
  $('inspector').scrollTop=0;document.querySelectorAll('.note.selected').forEach(e=>e.classList.remove('selected'));$('card-'+id)?.classList.add('selected');updateSaved();
}
function closeDetail(){hideTerm();highlightConnections(null);document.body.classList.remove('detail-open');$('inspector').hidden=true;lastDetailTrigger?.focus?.({preventScroll:true});}
function about(){highlightConnections(null);lastDetailTrigger=document.activeElement;selected=null;$('inspector').hidden=false;document.body.classList.add('detail-open');$('detail-category').textContent='ПУТЕВОДИТЕЛЬ';$('detail-body').style=css('practice');$('detail-body').innerHTML=`<div class="welcome-art" aria-hidden="true"><span>↗</span><span>✳</span><span>∿</span></div><span class="detail-tag">113 тем · 5 областей</span><h2 class="detail-title">Большая карта.<br>Маленькие шаги.</h2><p class="detail-intro">Здесь можно разобраться, какие модели существуют, как они учатся и что попробовать на своём компьютере.</p><div class="legend"><b>01</b> Выбери область в меню слева.</div><div class="legend"><b>02</b> Приблизь холст и открой карточку.</div><div class="legend"><b>03</b> Отметь звёздочкой интересные темы.</div><section class="lab-box"><h3>Начни с небольшого результата</h3><p>Классификатор команд, распознавание изображений, поиск аномалий или простой прогноз — у каждой задачи есть несколько путей.</p><button class="primary-button" data-nav="projects">Выбрать первый эксперимент →</button></section><section class="detail-section"><h3>КАК ЧИТАТЬ СВЯЗИ</h3><p>Цвет обозначает область. Сплошные линии связывают близкие идеи внутри неё, пунктир — темы из разных областей. Точный смысл связи указан в карточке. Это не обязательный порядок обучения.</p></section><section class="detail-section"><h3>ГРАНИЦЫ КАРТЫ</h3><p>Карта охватывает основные семейства и практические подходы, но не каждую публикацию. Архитектура, способ обучения и способ сжатия — разные оси: их можно сочетать. «Исследования» не означает преимущество перед простым методом.</p><p>Материалы проверены 24 октября 2026. Ссылки на документацию и статьи находятся в карточках.</p></section>`;}
function openNav(){ $('navigation').hidden=false;$('toggle-nav').setAttribute('aria-expanded','true'); }
function closeNav(){ $('navigation').hidden=true;$('toggle-nav').setAttribute('aria-expanded','false'); }
function hideSearch(){ $('search-results').hidden=true;$('search').setAttribute('aria-expanded','false'); }
function renderSearch(){const q=$('search').value.trim();if(!q){hideSearch();return;}const results=C.search(index,q);$('search-results').innerHTML=results.length?results.slice(0,15).map(n=>`<button data-result="${n.id}">${rich(n.name)}<small>${rich(index.get(n.parent)?.name||'Карта')}</small></button>`).join(''):'<p>Ничего не найдено. Попробуй «звук», «изображения», «сжатие» или «команды».</p>';$('search-results').hidden=false;$('search').setAttribute('aria-expanded','true');}
function renderRoute(){ $('route-list').innerHTML=route.size?[...route].map(id=>{const n=index.get(id);return `<div class="route-item"><button class="route-open" data-route-open="${id}">${rich(n.name)}<small>${rich(index.get(n.parent)?.name||'')}</small></button><button class="icon-button" data-remove="${id}" aria-label="Убрать ${esc(n.name)} из маршрута">×</button></div>`;}).join(''):'<div class="empty-route">Здесь пока чистый лист.<br>Нажимай ☆ на карточках, чтобы собрать свой маршрут.</div>';$('download-route').disabled=!route.size; }
function downloadRoute(){const text='# Мой маршрут AI Atlas\n\n'+[...route].map((id,i)=>{const n=index.get(id);return `## ${i+1}. ${n.name}\n\n${n.what}\n\n${n.how?'Как работает: '+n.how+'\n\n':''}${n.example?'Пример: '+n.example+'\n\n':''}${n.training?'Данные для обучения: '+n.training+'\n\n':''}${n.steps?n.steps.map((step,j)=>`${j+1}. ${step}`).join('\n')+'\n\n':''}${n.criteria?'Проверка результата: '+n.criteria+'\n\n':''}${n.lab?'Идея проекта: '+n.lab+'\n\n':''}${n.note?'Учесть: '+n.note+'\n\n':''}${(n.sources||[]).map(k=>data.sources[k]).filter(Boolean).map(([t,u])=>`- [${t}](${u})`).join('\n')}`;}).join('\n\n');const url=URL.createObjectURL(new Blob([text],{type:'text/markdown;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='ai-atlas-route.md';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
const tooltip=$('term-tooltip');let activeTerm=null,termTimer;
function hideTerm(){clearTimeout(termTimer);if(activeTerm){activeTerm.removeAttribute('aria-describedby');const term=data.glossary[activeTerm.dataset.term];activeTerm.setAttribute('title',term.full+' — '+term.description);activeTerm=null;}tooltip.hidden=true;}
function showTerm(element){
  const term=data.glossary?.[element.dataset.term];if(!term||pan.dragging||(pan.effective==='pan'&&element.closest('#canvas')))return;
  hideTerm();activeTerm=element;element.removeAttribute('title');element.setAttribute('aria-describedby','term-tooltip');
  tooltip.innerHTML=`<strong>${esc(term.full)}</strong><p>${esc(term.description)}</p>`;tooltip.hidden=false;
  const r=element.getBoundingClientRect(),t=tooltip.getBoundingClientRect();
  const x=C.clamp(r.left+r.width/2-t.width/2,12,Math.max(12,window.innerWidth-t.width-12));
  const y=r.top-t.height-12>=12?r.top-t.height-12:Math.min(r.bottom+12,window.innerHeight-t.height-12);
  tooltip.style.left=x+'px';tooltip.style.top=Math.max(12,y)+'px';
}
function deferHideTerm(){clearTimeout(termTimer);termTimer=setTimeout(hideTerm,160);}
document.addEventListener('pointerover',e=>{const term=e.target.closest('.term');if(term&&term!==activeTerm)showTerm(term);else if(e.target.closest('#term-tooltip'))clearTimeout(termTimer);});
document.addEventListener('pointerout',e=>{if(e.target.closest('.term,#term-tooltip')&&!e.relatedTarget?.closest('.term,#term-tooltip'))deferHideTerm();});
document.addEventListener('focusin',e=>{if(e.target.matches('.term'))showTerm(e.target);else hideTerm();});
document.addEventListener('click',e=>{const term=e.target.closest('.term');if(!term)return;if(e.target.closest('#canvas')&&pan.blocksClick(e.detail===0))return;e.preventDefault();e.stopImmediatePropagation();showTerm(term);},true);
document.addEventListener('keydown',e=>{if(e.target.matches('.term')&&(e.key==='Enter'||e.key===' ')){e.preventDefault();showTerm(e.target);}});
document.addEventListener('pointerdown',e=>{if(!e.target.closest('.term,#term-tooltip'))hideTerm();},true);
window.addEventListener('resize',hideTerm);$('inspector').addEventListener('scroll',hideTerm);
document.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;if(b.dataset.nav){navigate(b.dataset.nav,{fit:true});return;}if(b.dataset.open){navigate(b.dataset.open,{details:true});return;}if(b.dataset.result){navigate(b.dataset.result,{details:true});$('search').value='';return;}if(b.dataset.save){toggleSaved(b.dataset.save);return;}if(b.dataset.remove){toggleSaved(b.dataset.remove);return;}if(b.dataset.routeOpen){$('route-dialog').close();navigate(b.dataset.routeOpen,{details:true});return;}});
$('toggle-nav').onclick=()=> $('navigation').hidden?openNav():closeNav();$('close-nav').onclick=closeNav;$('home').onclick=fitAll;$('overview').onclick=fitAll;$('all-tool').onclick=fitAll;$('start-tool').onclick=()=>navigate('projects',{fit:true});$('fit-current').onclick=()=>current==='root'?fitAll():navigate(current,{fit:true});$('zoom-in').onclick=()=>zoom(1.2);$('zoom-out').onclick=()=>zoom(1/1.2);$('close-detail').onclick=closeDetail;$('about-button').onclick=about;
$('search').addEventListener('input',renderSearch);$('search').addEventListener('keydown',e=>{if(e.key==='ArrowDown'){e.preventDefault();$('search-results').querySelector('button')?.focus();}if(e.key==='Enter')$('search-results').querySelector('button')?.click();if(e.key==='Escape')hideSearch();});
$('search-results').addEventListener('keydown',e=>{const a=[...$('search-results').querySelectorAll('button')],i=a.indexOf(document.activeElement);if(e.key==='ArrowDown'||e.key==='ArrowUp'){e.preventDefault();a[C.clamp(i+(e.key==='ArrowDown'?1:-1),0,a.length-1)]?.focus();}});
document.addEventListener('pointerdown',e=>{if(!e.target.closest('.search-box'))hideSearch();});
$('route-button').onclick=()=>{renderRoute();$('route-dialog').showModal();};$('close-route').onclick=()=>$('route-dialog').close();$('download-route').onclick=downloadRoute;
function syncPanCursor(){
  const active=pan.effective==='pan';
  $('canvas').classList.toggle('pan-mode',active);
  $('canvas').classList.toggle('dragging',pan.dragging);
  $('pan-tool').classList.toggle('active',active);
  $('select-tool').classList.toggle('active',!active);
  $('pan-tool').setAttribute('aria-pressed',String(pan.tool==='pan'));
  $('select-tool').setAttribute('aria-pressed',String(pan.tool==='select'));
  $('pan-tool').classList.toggle('temporary',active&&pan.tool!=='pan');
}
function setPan(value){pan.setTool(value?'pan':'select');syncPanCursor();}
$('pan-tool').onclick=()=>setPan(true);$('select-tool').onclick=()=>setPan(false);
const pointers=new Map();let gesture=null;
function rebaseGesture(){const p=[...pointers.values()];if(p.length>1)gesture={view:{...view},x:(p[0].x+p[1].x)/2,y:(p[0].y+p[1].y)/2,d:Math.hypot(p[0].x-p[1].x,p[0].y-p[1].y)};else if(p.length)gesture={view:{...view},x:p[0].x,y:p[0].y};else gesture=null;}
// Capture before card buttons: a pan must never turn into their click action.
document.addEventListener('pointerdown',()=>pan.freshPointer(),true);
$('canvas').addEventListener('click',e=>{if(pan.blocksClick(e.detail===0)){e.preventDefault();e.stopImmediatePropagation();}},true);
$('canvas').addEventListener('pointerdown',e=>{
  const accepted=pan.begin({id:e.pointerId,button:e.button,onControl:!!e.target.closest('button,a'),shiftKey:e.shiftKey});
  syncPanCursor();if(!accepted)return;
  hideTerm();e.preventDefault();e.stopPropagation();
  const r=$('canvas').getBoundingClientRect();pointers.set(e.pointerId,{x:e.clientX-r.left,y:e.clientY-r.top});
  $('canvas').setPointerCapture(e.pointerId);rebaseGesture();
},true);
$('canvas').addEventListener('pointermove',e=>{
  if(!pointers.has(e.pointerId)||!gesture)return;
  const r=$('canvas').getBoundingClientRect();pointers.set(e.pointerId,{x:e.clientX-r.left,y:e.clientY-r.top});const p=[...pointers.values()];
  if(p.length>1&&gesture.d){const x=(p[0].x+p[1].x)/2,y=(p[0].y+p[1].y)/2,d=Math.hypot(p[0].x-p[1].x,p[0].y-p[1].y);view=C.zoomAt(gesture.view,gesture.view.scale*d/Math.max(gesture.d,1),gesture.x,gesture.y);view.x+=x-gesture.x;view.y+=y-gesture.y;}
  else view={...gesture.view,x:gesture.view.x+p[0].x-gesture.x,y:gesture.view.y+p[0].y-gesture.y};updateView();
});
function endPointer(e){pan.end(e.pointerId);if(pointers.delete(e.pointerId))rebaseGesture();syncPanCursor();}
document.addEventListener('pointerup',endPointer,true);document.addEventListener('pointercancel',endPointer,true);
$('canvas').addEventListener('lostpointercapture',endPointer);
$('canvas').addEventListener('wheel',e=>{
  e.preventDefault();hideTerm();
  const r=$('canvas').getBoundingClientRect();
  const unit=e.deltaMode===1?16:e.deltaMode===2?r.height:1;
  const dx=e.deltaX*unit,dy=e.deltaY*unit;
  // Shift and the hand tool apply to scrolling as well as pointer dragging.
  // Keep both axes: vertical scrolling must move vertically, never zoom in Pan.
  if(e.shiftKey||pan.effective==='pan'||Math.abs(dx)>Math.abs(dy)){
    view.x-=dx;view.y-=dy;updateView();
  }else zoom(Math.exp(-dy*.0015),e.clientX-r.left,e.clientY-r.top);
},{passive:false});
const isEditing=target=>target.closest('input,textarea,select,dialog,[contenteditable="true"]');
document.addEventListener('keydown',e=>{
  if(e.key==='Escape'){if(activeTerm){hideTerm();return;}hideSearch();closeNav();if(!$('route-dialog').open)closeDetail();return;}
  if(isEditing(e.target))return;
  if(e.key==='Shift'){pan.modifier(e.code||'Shift',true);syncPanCursor();return;}
  if(e.code==='Space'&&e.target===$('canvas')){e.preventDefault();pan.modifier('Space',true);syncPanCursor();return;}
  if(e.ctrlKey||e.metaKey||e.altKey)return;
  if(e.key==='/'){e.preventDefault();$('search').focus();}if(e.key==='f'||e.key==='F')fitAll();if(e.key==='+'||e.key==='=')zoom(1.2);if(e.key==='-')zoom(1/1.2);
  const moves={ArrowLeft:[70,0],ArrowRight:[-70,0],ArrowUp:[0,70],ArrowDown:[0,-70]};if(e.target===$('canvas')&&moves[e.key]){e.preventDefault();view.x+=moves[e.key][0];view.y+=moves[e.key][1];updateView();}
});
document.addEventListener('keyup',e=>{if(e.key==='Shift')pan.modifier(e.code||'Shift',false);if(e.code==='Space')pan.modifier('Space',false);syncPanCursor();});
function resetInteraction(){pan.reset();for(const id of pointers.keys())if($('canvas').hasPointerCapture(id))$('canvas').releasePointerCapture(id);pointers.clear();gesture=null;syncPanCursor();hideTerm();}
window.addEventListener('blur',resetInteraction);document.addEventListener('visibilitychange',()=>{if(document.hidden)resetInteraction();});
$('nodes').addEventListener('focusin',e=>{const b=e.target.closest('button'),id=b?.dataset.open||b?.dataset.save;if(!id)return;const card=cardById.get(id),r=$('canvas').getBoundingClientRect(),bounds=e.target.getBoundingClientRect();if(bounds.right<r.left||bounds.left>r.right||bounds.bottom<r.top||bounds.top>r.bottom){view=C.fitBox(card,r.width,r.height,40,1);updateView();}});
window.addEventListener('hashchange',()=>{const id=readHash();if(id==='root')fitAll();else if(index.has(id))navigate(id,{details:cardById.has(id),hash:false});});
new ResizeObserver(()=>{const w=$('canvas').clientWidth,h=$('canvas').clientHeight;if(w!==viewportWidth||h!==viewportHeight){view.x+=(w-viewportWidth)/2;view.y+=(h-viewportHeight)/2;updateView();}}).observe($('canvas'));
renderBoard();setPan(false);const initial=readHash();requestAnimationFrame(()=>{if(initial==='root')fitAll();else navigate(index.has(initial)?initial:'neural',{details:cardById.has(initial),hash:false});});
})();
