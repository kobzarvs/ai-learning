(function(scope){
'use strict';

// Fixed synthetic observations keep comparisons reproducible: changing the
// intervention must not draw new weather or new noise.
function observations(count=1000,seed=7){
 let state=seed>>>0;
 const uniform=()=>{state=(Math.imul(1664525,state)+1013904223)>>>0;return(state+1)/4294967297;};
 const normal=()=>Math.sqrt(-2*Math.log(uniform()))*Math.cos(2*Math.PI*uniform());
 const weather=Array.from({length:count},normal);
 const iceCream=weather.map(w=>w+0.2*normal());
 const noise=weather.map(()=>0.2*normal());
 return {weather,iceCream,noise};
}
const sample=observations();
const mean=values=>values.reduce((sum,value)=>sum+value,0)/values.length;
function simulate({effect=0,intervention=3,data=sample}={}){
 if(!Number.isFinite(effect)||!Number.isFinite(intervention))throw Error('Invalid experiment parameter');
 const {weather,iceCream,noise}=data;
 const swimming=weather.map((w,i)=>2*w+effect*iceCream[i]+noise[i]);
 const after=weather.map((w,i)=>2*w+effect*intervention+noise[i]);
 const beforeMean=mean(swimming),afterMean=mean(after),consumptionMean=mean(iceCream);
 let covariance=0,xVariance=0,yVariance=0;
 for(let i=0;i<weather.length;i++){
  const dx=iceCream[i]-consumptionMean,dy=swimming[i]-beforeMean;
  covariance+=dx*dy;xVariance+=dx*dx;yVariance+=dy*dy;
 }
 const baseMean=mean(weather.map((w,i)=>2*w+noise[i]));
 const curve=Array.from({length:13},(_,i)=>{const x=-3+i*0.5;return{x,y:baseMean+effect*x};});
 return {weather,iceCream,swimming,after,beforeMean,afterMean,consumptionMean,
  change:afterMean-beforeMean,correlation:covariance/Math.sqrt(xVariance*yVariance),curve};
}

const number=value=>(Math.abs(value)<0.0005?0:value).toLocaleString('ru-RU',{maximumFractionDigits:3});
const px=value=>value.toFixed(2);
function axes({xDomain,yDomain,xTicks,yTicks,xLabel,yLabel}){
 const x=value=>84+(value-xDomain[0])/(xDomain[1]-xDomain[0])*606;
 const y=value=>320-(value-yDomain[0])/(yDomain[1]-yDomain[0])*292;
 let html='';
 for(const value of xTicks)html+=`<line class="plot-grid" x1="${px(x(value))}" y1="28" x2="${px(x(value))}" y2="320"/><text x="${px(x(value))}" y="351" text-anchor="middle">${number(value)}</text>`;
 for(const value of yTicks)html+=`<line class="plot-grid" x1="84" y1="${px(y(value))}" x2="690" y2="${px(y(value))}"/><text x="70" y="${px(y(value)+7)}" text-anchor="end">${number(value)}</text>`;
 html+=`<path class="plot-axis" d="M84 28V320H690"/><text x="387" y="390" text-anchor="middle">${xLabel}</text><text transform="translate(23 174) rotate(-90)" text-anchor="middle">${yLabel}</text>`;
 return {x,y,html};
}
function scatter(result){
 const a=axes({xDomain:[-4,4],yDomain:[-11,11],xTicks:[-4,-2,0,2,4],yTicks:[-10,-5,0,5,10],xLabel:'Показатель потребления мороженого',yLabel:'Активность купальщиков'});
 // Display a subset for legibility. Correlation uses all 1000 observations.
 const dots=result.iceCream.slice(0,240).map((value,i)=>`<circle class="plot-observation" cx="${px(a.x(value))}" cy="${px(a.y(result.swimming[i]))}" r="3.5"/>`).join('');
 return `<svg viewBox="0 0 720 408" role="img" aria-labelledby="causal-scatter-title causal-scatter-description"><title id="causal-scatter-title">Мороженое и купание в обычных наблюдениях</title><desc id="causal-scatter-description">Каждая точка — одно наблюдение. Корреляция ${number(result.correlation)}. Общая погода влияет на обе величины.</desc>${a.html}<defs><clipPath id="causal-scatter-clip"><rect x="84" y="28" width="606" height="292"/></clipPath></defs><g clip-path="url(#causal-scatter-clip)">${dots}</g></svg>`;
}
function interventionPlot(result,level){
 const a=axes({xDomain:[-3,3],yDomain:[-2.5,2.5],xTicks:[-3,-2,-1,0,1,2,3],yTicks:[-2,-1,0,1,2],xLabel:'Принудительно заданное потребление',yLabel:'Средняя активность'});
 const path=result.curve.map((p,i)=>(i?'L':'M')+px(a.x(p.x))+' '+px(a.y(p.y))).join(' ');
 const baseline=a.y(result.beforeMean),selected=a.y(result.afterMean),position=a.x(level);
 return `<svg viewBox="0 0 720 408" role="img" aria-labelledby="causal-effect-title causal-effect-description"><title id="causal-effect-title">Результат принудительного изменения потребления</title><desc id="causal-effect-description">По горизонтали заданное потребление, по вертикали средняя активность купальщиков. Изменение относительно обычных наблюдений: ${number(result.change)}.</desc>${a.html}<line class="plot-baseline" x1="84" y1="${px(baseline)}" x2="690" y2="${px(baseline)}"/><path class="plot-effect" d="${path}"/><line class="plot-change" x1="${px(position)}" y1="${px(baseline)}" x2="${px(position)}" y2="${px(selected)}"/><circle class="plot-selected" cx="${px(position)}" cy="${px(selected)}" r="7"/></svg>`;
}
function render(id,{effect=0,intervention=3}={}){
 if(id!=='causal')return '';
 const result=simulate({effect,intervention});
 const effectChoices=[0,0.3,0.6].map(value=>`<button data-causal-effect="${value}" aria-pressed="${effect===value}">${value===0?'Нет прямого влияния':number(value)+' на единицу'}</button>`).join('');
 const levelChoices=[-3,1,3].map(value=>`<button data-causal-intervention="${value}" aria-pressed="${intervention===value}">${number(value)}</button>`).join('');
 return `<section class="reading-section lesson-visual" aria-labelledby="causal-visual-title"><h2 id="causal-visual-title">Наглядный опыт: связь и причина</h2><p>Погода влияет и на мороженое, и на купание. Добавь прямое влияние мороженого на купание и сравни два графика. Во всех переключениях используются те же погода и случайные помехи.</p><fieldset class="experiment-controls"><legend>Прямое влияние мороженого на купание</legend>${effectChoices}</fieldset><fieldset class="experiment-controls"><legend>Какое потребление принудительно задать всем наблюдениям</legend>${levelChoices}</fieldset><p class="experiment-code-values">Для такого же опыта в Python: <code>direct_effect = ${effect.toFixed(1)}</code> и <code>intervention_level = ${intervention.toFixed(1)}</code>.</p><div class="experiment-summary" aria-live="polite"><p>Корреляция в наблюдениях<strong>${number(result.correlation)}</strong></p><p>Изменение среднего при вмешательстве<strong>${number(result.change)}</strong></p></div><figure class="lesson-figure"><h3>1. Просто наблюдаем</h3><div class="lesson-plot">${scatter(result)}</div><figcaption>Точки вытянуты по диагонали: два показателя меняются вместе. Даже без прямого влияния мороженого это происходит из-за общей причины — погоды. Показаны 240 из 1000 наблюдений.</figcaption></figure><figure class="lesson-figure"><h3>2. Меняем только потребление мороженого</h3><div class="lesson-plot">${interventionPlot(result,intervention)}</div><div class="plot-legend"><span><i class="legend-effect"></i>После вмешательства</span><span><i class="legend-baseline"></i>Среднее до вмешательства</span></div><figcaption>${effect===0?'Линия горизонтальна и совпадает с прежним средним: новое потребление ничего не меняет в формуле купания. Сильная корреляция на первом графике при этом сохраняется.':'Линия наклонена: теперь потребление входит в формулу купания. При увеличении заданного потребления на 1 средняя активность увеличивается на '+number(effect)+'. Точка отмечает выбранное вмешательство.'}</figcaption></figure><p class="experiment-note">Это условные показатели, а не количество людей или порций. Отрицательное значение означает уровень ниже среднего. Графики — отдельная симуляция по формулам Python-примера; случайные точки и точные числа могут отличаться от твоего запуска.</p></section>`;
}
const api={observations,simulate,render};
if(typeof module!=='undefined'&&module.exports)module.exports=api;else scope.AtlasLessonVisuals=api;
})(typeof window!=='undefined'?window:globalThis);
