const {test}=require('node:test'),assert=require('node:assert/strict');
const S=require('../search.js');
const item={topicId:'causal',title:'Причинные модели',tab:'methods',section:'Сравнение',snippet:'Как отличить причину',anchor:{methodId:'random'}};
function elements(){const make=()=>({value:'',hidden:true,innerHTML:'',listeners:{},attrs:{},addEventListener(type,fn){this.listeners[type]=fn;},setAttribute(k,v){this.attrs[k]=v;}});return {input:make(),results:make()};}
test('search results escape content and keep exact names ahead of semantic matches',()=>{
 const merged=S.mergeResults([{...item,title:'<script>',topicId:'cnn',exact:true}], [item]);assert.equal(merged[0].topicId,'cnn');
 const html=S.resultHTML(merged);assert.match(html,/&lt;script&gt;/);assert.doesNotMatch(html,/<script>/);assert.match(html,/Методы и случаи/);
});
test('late semantic response cannot replace newer query or reopen a dismissed menu',async()=>{
 const dom=elements(),requests=[];
 const ui=S.create({...dom,localSearch:()=>[],onSelect:()=>{},fetch:()=>new Promise(resolve=>requests.push(resolve))});
 dom.input.value='первый запрос';const first=ui.refresh();
 dom.input.value='второй запрос';const second=ui.refresh();
 requests[1](Response.json({results:[{...item,title:'Второй'}]}));await second;
 requests[0](Response.json({results:[{...item,title:'Первый'}]}));await first;
 assert.match(dom.results.innerHTML,/Второй/);assert.doesNotMatch(dom.results.innerHTML,/Первый/);
 dom.input.value='третий';const third=ui.refresh();ui.hide();requests[2](Response.json({results:[item]}));await third;
 assert.equal(dom.results.hidden,true);
});
test('offline search retains local results and selecting a row supplies its exact location',async()=>{
 const dom=elements(),selected=[];
 const ui=S.create({...dom,localSearch:()=>[item],onSelect:r=>selected.push(r),fetch:async()=>{throw Error('offline');}});
 dom.input.value='эффект';await ui.refresh();assert.match(dom.results.innerHTML,/Причинные модели/);assert.match(dom.results.innerHTML,/временно недоступен/);
 dom.results.listeners.click({target:{closest:()=>({dataset:{searchResult:'0'}})}});
 assert.equal(selected[0].anchor.methodId,'random');assert.equal(dom.results.hidden,true);
});
test('pending search is polled and eventually replaces loading state',async()=>{
 const dom=elements();let calls=0;
 const ui=S.create({...dom,localSearch:()=>[],onSelect:()=>{},delay:async()=>{},fetch:async()=>++calls===1?Response.json({pending:true},{status:202}):Response.json({results:[item]})});
 dom.input.value='почему';await ui.refresh();assert.equal(calls,2);assert.match(dom.results.innerHTML,/Причинные модели/);assert.doesNotMatch(dom.results.innerHTML,/Ищу по/);
});
test('word matches and semantic matches appear in separate labelled groups',async()=>{
 const dom=elements();
 const ui=S.create({...dom,localSearch:()=>[],onSelect:()=>{},fetch:async()=>Response.json({keywordResults:[{...item,title:'Совпадение слов',topicId:'knn'}],semanticResults:[{...item,title:'Совпадение смысла'}]})});
 dom.input.value='соседи';await ui.refresh();
 const html=dom.results.innerHTML;
 assert.ok(html.indexOf('По словам')<html.indexOf('Совпадение слов'));
 assert.ok(html.indexOf('Совпадение слов')<html.indexOf('По смыслу'));
 assert.ok(html.indexOf('По смыслу')<html.indexOf('Совпадение смысла'));
});
