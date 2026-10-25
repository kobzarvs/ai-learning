(function(scope){
'use strict';

// Reuse tasks, assignment draws and noise when comparing instruction policies.
function observations(count=1000,seed=7){
 let state=seed>>>0;
 const uniform=()=>{state=(Math.imul(1664525,state)+1013904223)>>>0;return(state+1)/4294967297;};
 const difficulty=Array.from({length:count},()=>1+4*uniform());
 const assignmentDraw=difficulty.map(uniform);
 const noise=difficulty.map(()=>2*uniform()-1);
 return {difficulty,assignmentDraw,noise};
}
const sample=observations();
const mean=values=>values.reduce((sum,value)=>sum+value,0)/values.length;
function simulate({minutesSaved=4,assignment='biased',data=sample}={}){
 if(![0,4,8].includes(minutesSaved)||!['biased','random'].includes(assignment))throw Error('Invalid experiment parameter');
 const {difficulty,assignmentDraw,noise}=data;
 const instruction=difficulty.map((d,i)=>Number(assignmentDraw[i]<(assignment==='random'?0.5:0.1+0.2*(d-1))));
 const oldTime=difficulty.map((d,i)=>20+10*d+noise[i]);
 const newTime=oldTime.map(time=>time-minutesSaved);
 const observed=oldTime.map((time,i)=>instruction[i]?newTime[i]:time);
 const oldGroup=observed.filter((_,i)=>!instruction[i]),newGroup=observed.filter((_,i)=>instruction[i]);
 const groups=[0,1].map(group=>{
  const ids=instruction.flatMap((value,i)=>value===group?[i]:[]);
  const counts=[0,0,0,0];
  for(const i of ids)counts[Math.max(0,Math.min(3,Math.floor(difficulty[i]-1)))]++;
  return {count:ids.length,counts,shares:counts.map(count=>100*count/ids.length),
   meanDifficulty:mean(ids.map(i=>difficulty[i])),meanTime:mean(ids.map(i=>observed[i]))};
 });
 const observedDifference=mean(newGroup)-mean(oldGroup);
 const change=mean(newTime.map((time,i)=>time-oldTime[i]));
 const instructionMean=mean(instruction),timeMean=mean(observed);
 let covariance=0,xVariance=0,yVariance=0;
 for(let i=0;i<difficulty.length;i++){
  const dx=instruction[i]-instructionMean,dy=observed[i]-timeMean;
  covariance+=dx*dy;xVariance+=dx*dx;yVariance+=dy*dy;
 }
 return {difficulty,instruction,oldTime,newTime,observed,observedDifference,change,groups,
  correlation:covariance/Math.sqrt(xVariance*yVariance)};
}
const number=value=>(Math.abs(value)<0.0005?0:value).toLocaleString('ru-RU',{maximumFractionDigits:2});
const px=value=>value.toFixed(2);
function axes({xDomain,yDomain,xTicks,yTicks,xLabel,yLabel}){
 const x=value=>78+(value-xDomain[0])/(xDomain[1]-xDomain[0])*552;
 const y=value=>278-(value-yDomain[0])/(yDomain[1]-yDomain[0])*250;
 let html='';
 for(const value of xTicks)html+=`<line class="plot-grid" x1="${px(x(value))}" y1="28" x2="${px(x(value))}" y2="278"/><text x="${px(x(value))}" y="310" text-anchor="middle">${number(value)}</text>`;
 for(const value of yTicks)html+=`<line class="plot-grid" x1="78" y1="${px(y(value))}" x2="630" y2="${px(y(value))}"/><text x="64" y="${px(y(value)+7)}" text-anchor="end">${number(value)}</text>`;
 html+=`<path class="plot-axis" d="M78 28V278H630"/><text x="354" y="352" text-anchor="middle">${xLabel}</text><text transform="translate(23 153) rotate(-90)" text-anchor="middle">${yLabel}</text>`;
 return {x,y,html};
}
function distribution(result){
 const percentage=value=>value.toLocaleString('ru-RU',{maximumFractionDigits:1});
 const labels=['Прежняя инструкция','Новая инструкция'];
 // Both groups use the same difficulty bins and a common percentage scale.
 const maximum=Math.max(50,Math.ceil(Math.max(...result.groups.flatMap(g=>g.shares))/25)*25);
 const panels=result.groups.map((group,index)=>{
  const offset=index*210,top=50+offset,bottom=150+offset;
  const y=value=>bottom-value/maximum*100;
  const grid=[0,maximum/2,maximum].map(value=>`<line class="plot-grid" x1="78" y1="${px(y(value))}" x2="630" y2="${px(y(value))}"/><text x="64" y="${px(y(value)+7)}" text-anchor="end">${number(value)}%</text>`).join('');
  const bars=group.shares.map((share,i)=>{
   const center=78+(i+.5)*138;
   return `<rect class="${index?'plot-new':'plot-old'}" x="${px(center-51)}" y="${px(y(share))}" width="102" height="${px(bottom-y(share))}"><title>Сложность ${i+1}–${i+2}: ${group.counts[i]} заданий, ${percentage(share)}% группы</title></rect><text x="${px(center)}" y="${px(y(share)-7)}" text-anchor="middle">${percentage(share)}%</text><text x="${px(center)}" y="${bottom+28}" text-anchor="middle">${i+1}–${i+2}</text>`;
  }).join('');
  return `<g class="distribution-panel" data-instruction="${index}"><text class="distribution-heading" x="78" y="${top-25}">${labels[index]}</text><text x="630" y="${top-25}" text-anchor="end">Заданий: ${group.count}</text>${grid}${bars}<text x="78" y="${bottom+55}">Средняя сложность: ${number(group.meanDifficulty)}</text></g>`;
 }).join('');
 return `<svg viewBox="0 0 660 460" role="img" aria-labelledby="causal-distribution-title causal-distribution-description"><title id="causal-distribution-title">Распределение сложности отдельно для каждой инструкции</title><desc id="causal-distribution-description">Сложность задания: от 1 до 5. По вертикали — доля заданий внутри группы. Прежняя инструкция: ${result.groups[0].count} заданий, средняя сложность ${number(result.groups[0].meanDifficulty)}. Новая инструкция: ${result.groups[1].count} заданий, средняя сложность ${number(result.groups[1].meanDifficulty)}. Обе группы показаны с одинаковыми шкалами.</desc>${panels}<text x="78" y="450">Проще</text><text x="354" y="450" text-anchor="middle">Сложность задания</text><text x="630" y="450" text-anchor="end">Сложнее</text></svg>`;
}
function comparisonPlot(result){
 const a=axes({xDomain:[0,3],yDomain:[-15,15],xTicks:[],yTicks:[-15,-10,-5,0,5,10,15],xLabel:'Новая инструкция минус прежняя',yLabel:'Разница времени, минуты'});
 const bars=[{x:1,value:result.observedDifference,label:'Разные группы',color:'plot-naive'},{x:2,value:result.change,label:'Те же задания',color:'plot-causal'}].map(({x,value,label,color})=>{
  const top=Math.min(a.y(0),a.y(value)),height=Math.abs(a.y(value)-a.y(0));
  return `<rect class="${color}" x="${px(a.x(x)-46)}" y="${px(top)}" width="92" height="${px(Math.max(height,1))}"/><text x="${px(a.x(x))}" y="${px(value>=0?top-9:top+height+25)}" text-anchor="middle">${number(value)}</text><text x="${px(a.x(x))}" y="310" text-anchor="middle">${label}</text>`;
 }).join('');
 return `<svg viewBox="0 0 660 370" role="img" aria-labelledby="causal-effect-title causal-effect-description"><title id="causal-effect-title">Сравнение разных групп и причинное изменение для тех же заданий</title><desc id="causal-effect-description">Разные группы: ${number(result.observedDifference)} минуты. Те же задания с обеими инструкциями: ${number(result.change)} минуты. Отрицательное число означает ускорение.</desc>${a.html}<line class="plot-baseline" x1="78" y1="${px(a.y(0))}" x2="630" y2="${px(a.y(0))}"/>${bars}</svg>`;
}
function render(id,{minutesSaved=4,assignment='biased'}={}){
 if(id!=='causal')return '';
 const result=simulate({minutesSaved,assignment});
 const savingChoices=[0,4,8].map(value=>`<button data-causal-saving="${value}" aria-pressed="${minutesSaved===value}">${value} мин</button>`).join('');
 const assignmentChoices=[['biased','Чаще на сложных'],['random','Случайно: вероятность 50%']].map(([value,label])=>`<button data-causal-assignment="${value}" aria-pressed="${assignment===value}">${label}</button>`).join('');
 return `<section class="reading-section lesson-visual" aria-labelledby="causal-visual-title"><h2 id="causal-visual-title">Ускоряет ли новая инструкция работу?</h2><p>1000 сотрудников, по одному заданию у каждого. Меняй условия и сравнивай графики. Задания и случайные помехи сохраняются.</p><div class="experiment-controls-row"><fieldset class="experiment-controls"><legend>Экономия по новой инструкции в модели</legend>${savingChoices}</fieldset><fieldset class="experiment-controls"><legend>Кому дают новую инструкцию</legend>${assignmentChoices}</fieldset></div><div class="experiment-summary" aria-live="polite"><p>Разница между наблюдаемыми группами<strong>${number(result.observedDifference)} мин</strong></p><p>Причинное изменение для тех же заданий<strong>${number(result.change)} мин</strong></p></div><div class="lesson-charts" tabindex="0" aria-label="Два графика рядом. На узком экране прокручивай горизонтально."><div class="lesson-chart-row"><figure class="lesson-figure"><h3>1. Как распределены задания</h3><div class="lesson-plot">${distribution(result)}</div><figcaption>Доля заданий в каждом диапазоне сложности; учтены все 1000 заданий. Шкалы одинаковые. ${assignment==='biased'?'У прежней инструкции больше простых заданий, у новой — сложных.':'При случайном назначении распределения близки, но не совпадают точно.'}</figcaption></figure><figure class="lesson-figure"><h3>2. Сравниваем связь и причину</h3><div class="lesson-plot">${comparisonPlot(result)}</div><div class="group-time-summary"><p>Прежняя инструкция<strong>${number(result.groups[0].meanTime)} мин</strong></p><p>Новая инструкция<strong>${number(result.groups[1].meanTime)} мин</strong></p></div><p class="group-time-calculation">Разница средних: <strong>${number(result.groups[1].meanTime)} − ${number(result.groups[0].meanTime)} ≈ ${number(result.observedDifference)} мин</strong></p><figcaption>«Разные группы» сравнивает сотрудников с разными заданиями. «Те же задания» показывает эффект смены инструкции при неизменной работе: ${number(result.change)} мин. Минус — быстрее, плюс — медленнее.${minutesSaved===0?' При нулевой экономии различие групп остаётся из-за состава заданий и случайных помех.':''}</figcaption></figure></div></div><p class="experiment-note">Учебная симуляция рабочей ситуации, не данные предприятия. В жизни нельзя одновременно увидеть оба исхода одного задания: эффект приходится оценивать методами из вкладки «Методы и случаи».</p><details class="experiment-details"><summary>Как повторить выбранные условия в Python</summary><p>Во вкладке Python задай <code>minutes_saved = ${minutesSaved.toFixed(1)}</code> — экономию минут и <code>random_assignment = ${assignment==='random'?'True':'False'}</code> — случайное назначение инструкции. Формулы те же; генераторы случайных чисел различаются, поэтому точки и оценка по группам могут немного отличаться.</p><p>Корреляция применения новой инструкции со временем: ${number(result.correlation)}. Это связь в записях; она сама по себе не измеряет причинный эффект.</p></details></section>`;
}
const api={observations,simulate,render};
if(typeof module!=='undefined'&&module.exports)module.exports=api;else scope.AtlasLessonVisuals=api;
})(typeof window!=='undefined'?window:globalThis);
