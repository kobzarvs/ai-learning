(function(scope){
'use strict';
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const tabs={article:'Теория',methods:'Методы и случаи',python:'Python',rust:'Rust + Burn',mlx:'Python + MLX',exercise:'Практика'};
function mergeResults(local,semantic){
 const seen=new Set(),items=[];
 for(const item of [...local.filter(x=>x.exact),...semantic,...local]){
  const key=[item.topicId,item.tab,item.section].join(':');if(seen.has(key))continue;seen.add(key);items.push(item);if(items.length===15)break;
 }
 return items;
}
function resultHTML(items,offset=0){return items.map((item,index)=>`<button type="button" data-search-result="${index+offset}"><strong>${esc(item.title)}</strong><small>${esc(tabs[item.tab]||'Тема')}${item.section?' · '+esc(item.section):''}</small><span class="search-snippet">${esc(item.snippet)}</span></button>`).join('');}
function reveal(container,anchor={}){
 if(anchor.methodId)return Boolean(scope.AtlasLessonGuide?.jump(container,'causal-method-'+anchor.methodId));
 if(anchor.guideView)scope.AtlasLessonGuide?.selectView(container,anchor.guideView);
 let target;
 if(anchor.caseTitle){
  target=[...container.querySelectorAll('#causal-cases .guide-case')].find(el=>el.querySelector('summary strong')?.textContent===anchor.caseTitle);
  if(target)target.open=true;
 }else if(anchor.section){target=[...container.querySelectorAll('h2,h3,.code-block>header>span')].find(el=>el.textContent.trim()===anchor.section);}
 if(target){target.scrollIntoView({block:'start'});target.setAttribute('tabindex','-1');target.focus({preventScroll:true});}
 return Boolean(target||anchor.guideView);
}
function create({input,results,localSearch,onSelect,fetch:fetcher=scope.fetch?.bind(scope),delay=ms=>new Promise(resolve=>setTimeout(resolve,ms))}){
 let generation=0,controller=null,timer=null,items=[];
 function cancel(){generation++;controller?.abort();clearTimeout(timer);timer=null;}
 function hide(){cancel();results.hidden=true;input.setAttribute('aria-expanded','false');}
 function paint(query,remote={},pending=false,failed=false){
  const keyword=mergeResults(localSearch(query),remote.keywordResults||[]).slice(0,6);
  const keys=new Set(keyword.map(x=>[x.topicId,x.tab,x.section].join(':')));
  const semantic=(remote.semanticResults||remote.results||[]).filter(x=>!keys.has([x.topicId,x.tab,x.section].join(':'))).slice(0,9);
  items=[...keyword,...semantic];
  const status=query.length<2?'Введите хотя бы два символа для поиска по смыслу.':pending?'Ищу по всем материалам…':failed?'Поиск по смыслу временно недоступен.':!semantic.length?'Других совпадений по смыслу нет.':'';
  results.innerHTML=`<section class="search-group"><h3>По словам</h3>${resultHTML(keyword)}${!keyword.length?'<p class="search-status">Совпадений по словам нет.</p>':''}</section><section class="search-group"><h3>По смыслу</h3>${resultHTML(semantic,keyword.length)}${status?`<p class="search-status" role="status">${esc(status)}</p>`:''}</section>`;
  results.hidden=false;input.setAttribute('aria-expanded','true');
 }
 async function refresh(){
  cancel();const own=generation,query=input.value.trim();
  if(!query){hide();return;}
  paint(query,{},query.length>=2);
  if(query.length<2)return;
  controller=new AbortController();const active=controller,signal=controller.signal,deadline=setTimeout(()=>active.abort(),15000);
  try{
   for(let attempt=0;attempt<16;attempt++){
    const response=await fetcher('/api/search?q='+encodeURIComponent(query),{signal,cache:'no-store'});
    if(own!==generation)return;
    if(response.status===202){await delay(700);if(own!==generation)return;continue;}
    if(!response.ok)throw Error('Search unavailable');
    const data=await response.json();if(own!==generation)return;
    const stamp=rows=>Array.isArray(rows)?rows.map(item=>({...item,revision:data.revision})):[];
    paint(query,{keywordResults:stamp(data.keywordResults),semanticResults:stamp(data.semanticResults||data.results)},false,data.semanticUnavailable===true);return;
   }
   throw Error('Search timeout');
  }catch(error){if(own===generation)paint(query,{},false,true);}
  finally{clearTimeout(deadline);}
 }
 input.addEventListener('input',()=>{cancel();const query=input.value.trim();if(!query){hide();return;}paint(query,{},query.length>=2);timer=setTimeout(refresh,500);});
 input.addEventListener('focus',()=>{if(input.value.trim()&&results.hidden)refresh();});
 results.addEventListener('click',event=>{const button=event.target.closest('[data-search-result]');if(!button)return;const item=items[Number(button.dataset.searchResult)];if(!item)return;hide();input.value='';onSelect(item);});
 return {hide,refresh,invalidate(){if(!results.hidden&&input.value.trim())refresh();}};
}
const api={create,mergeResults,resultHTML,reveal};
if(typeof module!=='undefined'&&module.exports)module.exports=api;else scope.AtlasSearch=api;
})(typeof window!=='undefined'?window:globalThis);
