(function(scope){
'use strict';
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const paragraphs=value=>String(value??'').split(/\n\s*\n/).map(text=>`<p>${esc(text)}</p>`).join('');
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
 const caseBody=item=>`<dl class="guide-case-body"><dt>Вопрос</dt><dd>${esc(item.question)}</dd><dt>Как устроено сравнение</dt><dd>${esc(item.approach)}</dd><dt>Вывод и его границы</dt><dd>${esc(item.lesson)}</dd></dl>`;
 const caseCard=(item,related=false)=>`<details class="guide-case"><summary><span class="guide-case-kind">${esc(item.kind)}</span><strong>${esc(item.title)}</strong><span class="guide-case-open" aria-hidden="true">Развернуть разбор ↓</span><span class="guide-case-close" aria-hidden="true">Свернуть разбор ↑</span></summary>${caseBody(item)}${related?`<div class="guide-related" aria-label="Методы из этого разбора">${item.methods.map(id=>`<button data-guide-jump="causal-method-${esc(id)}">${esc(methodNames.get(id))} →</button>`).join('')}</div>`:''}</details>`;
 const groups=guide.groups.map((group,index)=>`<section class="guide-panel" data-guide-panel="${index}" id="causal-guide-${index}" aria-labelledby="guide-heading-${index}"${index?' hidden':''}>
  <header class="guide-panel-heading"><p class="guide-eyebrow">Группа ${index+1} из ${guide.groups.length}</p><h2 id="guide-heading-${index}">${esc(cleanTitle(group.title))}</h2><p>${esc(group.purpose)}</p></header>
  <p class="guide-list-hint">Выбери метод, чтобы разобрать его на примере.</p>
  <div class="guide-methods">${group.methods.map((method,methodIndex)=>{
   const cases=guide.cases.filter(item=>item.methods.includes(method.id));
   return `<details class="guide-method" id="causal-method-${esc(method.id)}"${index===0&&methodIndex===0?' open':''}>
    <summary><span class="guide-method-number" aria-hidden="true">${String(methodIndex+1).padStart(2,'0')}</span><span class="guide-method-heading"><strong>${esc(method.title)}</strong><span>${esc(method.meaning)}</span></span><span class="guide-disclosure" aria-hidden="true">+</span></summary>
    <div class="guide-method-body"><section class="guide-example"><h3>Пример задачи</h3>${paragraphs(method.example)}</section>
     <div class="guide-requirements"><section><h3>Какие данные собрать</h3>${paragraphs(method.data)}</section><section class="guide-assumptions"><h3>Что может исказить результат</h3>${paragraphs(method.assumptions)}</section></div>
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
   <section class="guide-panel" data-guide-panel="basics" id="causal-basics" aria-labelledby="guide-heading-basics" hidden><header class="guide-panel-heading"><p class="guide-eyebrow">Справочник</p><h2 id="guide-heading-basics">Понятия и порядок изучения</h2><p>${esc(guide.intro)}</p></header><section class="guide-study-route"><h3>С чего начать</h3><ol>${guide.route.map(step=>`<li>${esc(step)}</li>`).join('')}</ol></section><dl class="guide-basics">${guide.foundations.map(([term,meaning])=>`<div><dt>${esc(term)}</dt><dd>${esc(meaning)}</dd></div>`).join('')}</dl></section>
  </div></div></div>`;
}

function selectView(container,key){
 const root=container.querySelector('.causal-guide');
 if(!root)return false;
 const panels=[...root.querySelectorAll('[data-guide-panel]')];
 if(!panels.some(panel=>panel.dataset.guidePanel===key))return false;
 for(const panel of panels)panel.hidden=panel.dataset.guidePanel!==key;
 for(const button of root.querySelectorAll('[data-guide-view]'))button.setAttribute('aria-pressed',String(button.dataset.guideView===key));
 root.dataset.guideCurrent=key;
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

const api={render,selectView,jump};
if(typeof module!=='undefined'&&module.exports)module.exports=api;else scope.AtlasLessonGuide=api;
})(typeof window!=='undefined'?window:globalThis);
