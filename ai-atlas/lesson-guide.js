(function(scope){
'use strict';
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const editAttr=(path,part)=>path?` data-edit-path="${esc(path)}"${part===undefined?'':` data-edit-part="${part}"`}`:'';
const paragraphs=(value,path)=>String(value??'').split(/\n\s*\n/).map((text,i)=>text.trim()?`<p${editAttr(path,i)}>${esc(text)}</p>`:'').join('');
// The navigation describes situations; full method names remain in the panels.
const situations={
 random:'Могу провести эксперимент',
 standardize:'Есть только готовые данные',
 iv:'Есть порог или внешнее изменение',
 meta:'Нужно понять, кому помогает',
 dag:'Нужно объяснить механизм',
 pc:'Нужно найти причинные связи',
 longitudinal:'Сложные условия и проверка выводов',
 neural:'Продвинутые направления'
};

function render(guide){
 const count=guide.groups.reduce((sum,group)=>sum+group.methods.length,0);
 const methodNames=new Map(guide.groups.flatMap(group=>group.methods.map(method=>[method.id,method.title])));
 const cleanTitle=title=>title.replace(/^\d+\.\s*/, '');
 const caseBody=item=>`<dl class="guide-case-body">${[['question','Вопрос'],['approach','Как устроено сравнение'],['lesson','Вывод и его границы']].filter(([field])=>item[field]).map(([field,label])=>`<dt>${label}</dt><dd>${paragraphs(item[field],'lesson.guide.cases.'+guide.cases.indexOf(item)+'.'+field)}</dd>`).join('')}</dl>`;
 const caseCard=(item,related=false)=>`<details class="guide-case"><summary><span class="guide-case-kind">${esc(item.kind)}</span><strong>${esc(item.title)}</strong><span class="guide-case-open" aria-hidden="true">Развернуть разбор ↓</span><span class="guide-case-close" aria-hidden="true">Свернуть разбор ↑</span></summary>${caseBody(item)}${related?`<div class="guide-related" aria-label="Методы из этого разбора">${item.methods.map(id=>`<button data-guide-jump="causal-method-${esc(id)}">${esc(methodNames.get(id))} →</button>`).join('')}</div>`:''}</details>`;
 const groups=guide.groups.map((group,index)=>`<section class="guide-panel" data-guide-panel="${index}" id="causal-guide-${index}" aria-labelledby="guide-heading-${index}"${index?' hidden':''}>
  <header class="guide-panel-heading"><p class="guide-eyebrow">Группа ${index+1} из ${guide.groups.length}</p><h2 id="guide-heading-${index}">${esc(cleanTitle(group.title))}</h2>${paragraphs(group.purpose,'lesson.guide.groups.'+index+'.purpose')}</header>
  <p class="guide-list-hint">Выбери метод, чтобы разобрать его на примере.</p>
  <div class="guide-methods">${group.methods.map((method,methodIndex)=>{
   const cases=guide.cases.filter(item=>item.methods.includes(method.id));
   return `<details class="guide-method" id="causal-method-${esc(method.id)}"${index===0&&methodIndex===0?' open':''}>
    <summary><span class="guide-method-number" aria-hidden="true">${String(methodIndex+1).padStart(2,'0')}</span><span class="guide-method-heading"><strong>${esc(method.title)}</strong><span${editAttr('lesson.guide.groups.'+index+'.methods.'+methodIndex+'.meaning')}>${esc(method.meaning)}</span></span><span class="guide-disclosure" aria-hidden="true">+</span></summary>
    <div class="guide-method-body"><section class="guide-example"><h3>Пример задачи</h3>${paragraphs(method.example,'lesson.guide.groups.'+index+'.methods.'+methodIndex+'.example')}</section>
     <div class="guide-requirements"><section><h3>Какие данные собрать</h3>${paragraphs(method.data,'lesson.guide.groups.'+index+'.methods.'+methodIndex+'.data')}</section><section class="guide-assumptions"><h3>Что может исказить результат</h3>${paragraphs(method.assumptions,'lesson.guide.groups.'+index+'.methods.'+methodIndex+'.assumptions')}</section></div>
     ${cases.length?`<section class="guide-linked-cases"><h3>Разборы с этим методом</h3>${cases.map(item=>caseCard(item)).join('')}</section>`:''}
    </div></details>`;
  }).join('')}</div></section>`).join('');
 return `<div class="causal-guide" data-guide-current="0">
  <header class="guide-intro"><p class="guide-eyebrow">Методы и семейства: ${count} · Разборы: ${guide.cases.length}</p><h2>Как выбрать причинный метод</h2><p>Начни с условий своей задачи. В каждой группе — методы, примеры и требования к сравнению.</p></header>
  <div class="guide-layout"><nav class="guide-sidebar" aria-label="Выбор ситуации и справочников"><p class="guide-nav-label">Какие у тебя условия?</p>
   ${guide.groups.map((group,index)=>`<button data-guide-view="${index}" aria-controls="causal-guide-${index}" aria-pressed="${index===0}"><span class="guide-nav-number" aria-hidden="true">${String(index+1).padStart(2,'0')}</span><span>${esc(group.situation||situations[group.methods[0]?.id]||cleanTitle(group.title))}</span><span class="guide-nav-arrow" aria-hidden="true">→</span></button>`).join('')}
   <div class="guide-nav-extra"><button data-guide-view="cases" aria-controls="causal-cases" aria-pressed="false">Все случаи из практики <span aria-hidden="true">→</span></button><button data-guide-view="basics" aria-controls="causal-basics" aria-pressed="false">Понятия и порядок изучения <span aria-hidden="true">→</span></button></div>
  </nav><div class="guide-main">${groups}
   <section class="guide-panel" data-guide-panel="cases" id="causal-cases" aria-labelledby="guide-heading-cases" hidden><header class="guide-panel-heading"><p class="guide-eyebrow">Практика</p><h2 id="guide-heading-cases">Как методы работают в задачах</h2><p>У каждого разбора указан его характер: исследование, опыт платформы или учебный сценарий. Из разбора можно перейти прямо к методу.</p></header><div class="guide-cases">${guide.cases.map(item=>caseCard(item,true)).join('')}</div></section>
   <section class="guide-panel" data-guide-panel="basics" id="causal-basics" aria-labelledby="guide-heading-basics" hidden><header class="guide-panel-heading"><p class="guide-eyebrow">Справочник</p><h2 id="guide-heading-basics">Понятия и порядок изучения</h2>${paragraphs(guide.intro,'lesson.guide.intro')}</header><section class="guide-study-route"><h3>С чего начать</h3><ol>${guide.route.map((step,i)=>`<li${editAttr('lesson.guide.route.'+i)}>${esc(step)}</li>`).join('')}</ol></section><dl class="guide-basics">${guide.foundations.map(([term,meaning],i)=>`<div${editAttr('lesson.guide.foundations.'+i)}><dt>${esc(term)}</dt><dd>${esc(meaning)}</dd></div>`).join('')}</dl></section>
  </div></div></div>`;
}

const readingPositions=new WeakMap();
function positionsFor(root){
 if(!readingPositions.has(root))readingPositions.set(root,new Map());
 return readingPositions.get(root);
}
function showView(root,key){
 for(const panel of root.querySelectorAll('[data-guide-panel]'))panel.hidden=panel.dataset.guidePanel!==key;
 for(const button of root.querySelectorAll('[data-guide-view]'))button.setAttribute('aria-pressed',String(button.dataset.guideView===key));
 root.dataset.guideCurrent=key;
}
function panelTop(container,panel){
 return Math.max(0,panel.getBoundingClientRect().top-container.getBoundingClientRect().top+container.scrollTop-container.clientTop);
}
function selectView(container,key){
 const root=container.querySelector('.causal-guide');
 if(!root)return false;
 const panel=[...root.querySelectorAll('[data-guide-panel]')].find(panel=>panel.dataset.guidePanel===key);
 if(!panel)return false;
 if(root.dataset.guideCurrent===key){showView(root,key);return true;}
 const positions=positionsFor(root);
 positions.set(root.dataset.guideCurrent,container.scrollTop);
 showView(root,key);
 container.scrollTop=positions.get(key)??panelTop(container,panel);
 return true;
}
function capture(container){
 const root=container.querySelector('.causal-guide');
 if(!root)return null;
 const positions=positionsFor(root),view=root.dataset.guideCurrent;
 positions.set(view,container.scrollTop);
 return {view,positions:Object.fromEntries(positions),details:[...root.querySelectorAll('details')].map(detail=>detail.open)};
}
function restore(container,snapshot){
 const root=container.querySelector('.causal-guide');
 if(!root||!snapshot)return false;
 const panels=[...root.querySelectorAll('[data-guide-panel]')],panel=panels.find(panel=>panel.dataset.guidePanel===snapshot.view);
 if(!panel)return false;
 const keys=new Set(panels.map(panel=>panel.dataset.guidePanel));
 const positions=new Map(Object.entries(snapshot.positions||{}).filter(([key,value])=>keys.has(key)&&Number.isFinite(value)&&value>=0));
 readingPositions.set(root,positions);
 showView(root,snapshot.view);
 [...root.querySelectorAll('details')].forEach((detail,index)=>{if(typeof snapshot.details?.[index]==='boolean')detail.open=snapshot.details[index];});
 container.scrollTop=positions.get(snapshot.view)??panelTop(container,panel);
 return true;
}

function jump(container,id){
 const target=[...container.querySelectorAll('.guide-method')].find(method=>method.id===id);
 const panel=target?.closest('[data-guide-panel]');
 if(!panel||!selectView(container,panel.dataset.guidePanel))return false;
 target.open=true;
 target.querySelector('summary')?.focus({preventScroll:true});
 target.scrollIntoView({block:'start'});
 return true;
}

const api={render,selectView,jump,capture,restore};
if(typeof module!=='undefined'&&module.exports)module.exports=api;else scope.AtlasLessonGuide=api;
})(typeof window!=='undefined'?window:globalThis);
