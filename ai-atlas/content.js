(function(scope){
'use strict';
const object=value=>value&&typeof value==='object'&&!Array.isArray(value);
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const bounded=(value,min,max)=>Number.isFinite(value)?Math.max(min,Math.min(max,value)):null;

function groupSources(sources){
  // A project's documentation and GitHub repository belong to the same group.
  const families=[
    ['PyTorch',/^https?:\/\/(?:(?:docs\.)?pytorch\.org\/|github\.com\/pytorch\/)/i],
    ['EconML',/^https?:\/\/(?:www\.pywhy\.org\/EconML\/|github\.com\/py-why\/EconML\/)/i],
    ['DoWhy',/^https?:\/\/(?:www\.pywhy\.org\/dowhy\/|github\.com\/py-why\/dowhy\/)/i],
    ['causal-learn',/^https?:\/\/(?:causal-learn\.readthedocs\.io\/|github\.com\/py-why\/causal-learn\/)/i],
    ['scikit-learn',/^https?:\/\/(?:scikit-learn\.org\/|github\.com\/scikit-learn\/)/i],
    ['MLX',/^https?:\/\/(?:ml-explore\.github\.io\/|github\.com\/ml-explore\/)/i],
    ['Burn',/^https?:\/\/(?:burn\.dev\/|docs\.rs\/burn(?:\/|$)|github\.com\/tracel-ai\/burn\/)/i],
    ['Hugging Face',/^https?:\/\/huggingface\.co\//i],
    ['Tigramite',/^https?:\/\/github\.com\/jakobrunge\/tigramite(?:\/|$)/i],
    ['CausalImpact',/^https?:\/\/github\.com\/google\/CausalImpact(?:\/|$)/i],
    ['Microsoft Research',/^https?:\/\/www\.microsoft\.com\/en-us\/research\//i],
    ['Apple',/^https?:\/\/developer\.apple\.com\//i],
    ['Книги и научные публикации',/^https?:\/\/(?:arxiv\.org|www\.nber\.org|pubmed\.ncbi\.nlm\.nih\.gov|dspace\.mit\.edu|miguelhernan\.org|ftp\.cs\.ucla\.edu)\//i]
  ];
  const groups=new Map(),seen=new Set();
  for(const entry of sources){
    if(!Array.isArray(entry))continue;
    const [label,url]=entry;
    if(typeof label!=='string'||typeof url!=='string'||!/^https?:\/\//i.test(url)||seen.has(url))continue;
    seen.add(url);
    const group=families.find(([,pattern])=>pattern.test(url))?.[0]||'Другие материалы';
    const prefix=group+':';
    let title=label.toLocaleLowerCase('ru').startsWith(prefix.toLocaleLowerCase('ru'))?label.slice(prefix.length).trim():label;
    title=title?title[0].toLocaleUpperCase('ru')+title.slice(1):label;
    if(!groups.has(group))groups.set(group,[]);
    groups.get(group).push([title,url]);
  }
  return [...families.map(([title])=>title),'Другие материалы'].filter(title=>groups.has(title)).map(title=>({title,links:groups.get(title)}));
}

function apply(records,{data,index,lessons,presentation}){
  if(!Array.isArray(records)||records.some(r=>!object(r)||typeof r.id!=='string'||!object(r.payload)))throw Error('Invalid content records');
  const originals=new Map();
  (function visit(n){originals.set(n.id,n);for(const child of n.children||[])visit(child);})(data.root);
  const replace=(target,source)=>{for(const key of Object.keys(target))delete target[key];for(const [key,value] of Object.entries(source))if(!['__proto__','constructor','prototype'].includes(key))target[key]=value;};
  for(const {id,payload} of records){
    if(id==='_catalog'){
      for(const key of ['sources','glossary'])if(object(payload[key]))replace(data[key],payload[key]);
      if(object(payload.meta))replace(lessons.meta,payload.meta);
      continue;
    }
    if(!index.has(id))continue;
    if(object(payload.node))for(const [key,value] of Object.entries(payload.node)){
      if(['id','children','parent','branch','__proto__','prototype','constructor'].includes(key))continue;
      index.get(id)[key]=value;originals.get(id)[key]=value;
    }
    if(object(payload.lesson))lessons.topics[id]=payload.lesson;
    else if(payload.lesson===null)delete lessons.topics[id];
    presentation[id]={layout:object(payload.layout)?payload.layout:{},sections:object(payload.sections)?payload.sections:{}};
  }
}

function chart(block){
  const palette=['#356957','#8b60a8','#c17734','#427aa3'];
  const series=(Array.isArray(block.series)?block.series:[]).slice(0,8).map(s=>({name:s.name,values:(Array.isArray(s.values)?s.values:[]).filter(p=>Number.isFinite(p.x)&&Number.isFinite(p.y)).slice(0,1000)}));
  const points=series.flatMap(s=>s.values);if(!points.length)return '';
  let minX=Math.min(...points.map(p=>p.x)),maxX=Math.max(...points.map(p=>p.x));
  let minY=Math.min(0,...points.map(p=>p.y)),maxY=Math.max(0,...points.map(p=>p.y));
  if(minX===maxX)maxX=minX+1;if(minY===maxY)maxY=minY+1;
  const x=v=>75+(v-minX)/(maxX-minX)*620,y=v=>315-(v-minY)/(maxY-minY)*255;
  const number=n=>Number(n.toFixed(2)).toLocaleString('ru');
  let svg='';
  for(let i=0;i<=4;i++){
    const vy=minY+(maxY-minY)*i/4,vx=minX+(maxX-minX)*i/4;
    svg+=`<path d="M75 ${y(vy)}H695" stroke="#dce4d6"/><text x="65" y="${y(vy)+7}" text-anchor="end">${number(vy)}</text><text x="${x(vx)}" y="345" text-anchor="middle">${number(vx)}</text>`;
  }
  svg+=series.map((s,i)=>`<g fill="${palette[i%4]}" stroke="${palette[i%4]}">${block.type==='scatter'?'':`<polyline fill="none" stroke-width="3" points="${s.values.map(p=>`${x(p.x)},${y(p.y)}`).join(' ')}"/>`}${s.values.map(p=>`<circle cx="${x(p.x)}" cy="${y(p.y)}" r="4"><title>${esc(s.name)}: ${number(p.x)}, ${number(p.y)}</title></circle>`).join('')}</g>`).join('');
  return `<figure class="content-chart"><figcaption>${esc(block.title)}</figcaption><svg viewBox="0 0 750 390" role="img" aria-label="${esc(block.title)}"><g font-size="20" fill="#34432f">${svg}<text x="385" y="380" text-anchor="middle">${esc(block.xLabel)}</text><text x="75" y="30">${esc(block.yLabel)}</text></g></svg><div class="content-chart-legend">${series.map((s,i)=>`<span style="color:${palette[i%4]}">● ${esc(s.name)}</span>`).join('')}</div></figure>`;
}

function renderSections(blocks,{rich=esc,code,experiment}={}){
  if(!Array.isArray(blocks))return null;
  return blocks.slice(0,100).map(block=>{
    if(!object(block))return '';
    const width=bounded(block.width,320,1600),emphasis=['info','task','warning'].includes(block.emphasis)?block.emphasis:'';
    let body='';
    if(block.kind==='chart')body=chart(block);
    else if(block.kind==='experiment')body=experiment?.(block.experiment)||'';
    else if(block.kind==='code')body=code?code(block):`<pre><code>${esc(block.code)}</code></pre>`;
    else if(block.kind==='list')body=`<ul>${(Array.isArray(block.items)?block.items:[]).map(item=>`<li>${rich(item,true)}</li>`).join('')}</ul>`;
    else body=String(block.body||'').split(/\n\s*\n/).map(text=>`<p>${rich(text,true).replace(/\n/g,'<br>')}</p>`).join('');
    return `<section class="reading-section content-block${emphasis?' content-'+emphasis:''}"${width?` style="max-width:${width}px"`:''}>${block.kind!=='chart'&&block.title?`<h2>${esc(block.title)}</h2>`:''}${body}</section>`;
  }).join('');
}

function layoutValues(layout={}){
  return {readerWidth:bounded(layout.readerWidth,700,1800),textWidth:bounded(layout.textWidth,500,1600),chartsColumns:[1,2].includes(layout.chartsColumns)?layout.chartsColumns:null};
}

function createClient({fetch:fetcher,onChange,onError=()=>{},now=()=>Date.now()}){
  let revision='0',pending=null,lastAttempt=0;
  return {
    get revision(){return revision;},
    refresh(){
      if(pending)return pending;
      if(lastAttempt&&now()-lastAttempt<2000)return Promise.resolve(false);
      lastAttempt=now();
      pending=(async()=>{
        try{
          const response=await fetcher('/api/content?since='+revision,{cache:'no-store',signal:AbortSignal.timeout(12000)});
          if(!response.ok)throw Error('Content unavailable');
          const update=await response.json();
          if(!/^\d+$/.test(update.revision)||!Array.isArray(update.records))throw Error('Invalid content response');
          if(update.records.length)onChange(update.records,revision!=='0');
          revision=update.revision;
          return true;
        }catch(error){onError(error);return false;}
        finally{pending=null;}
      })();
      return pending;
    }
  };
}

function start(options){
  if(!/^https?:$/.test(scope.location?.protocol||''))return null;
  const client=createClient({...options,fetch:scope.fetch.bind(scope)});
  let lastActivity=Date.now();
  const refresh=()=>{if(!document.hidden)client.refresh();};
  for(const event of ['pointerdown','keydown','wheel'])document.addEventListener(event,()=>{lastActivity=Date.now();},{passive:true});
  scope.addEventListener('focus',()=>{lastActivity=Date.now();refresh();});
  document.addEventListener('visibilitychange',()=>{if(!document.hidden){lastActivity=Date.now();refresh();}});
  // Pause after five idle minutes so an abandoned tab does not keep the free database awake.
  scope.setInterval(()=>{if(Date.now()-lastActivity<300000)refresh();},30000);
  refresh();return client;
}

const api={apply,groupSources,renderSections,layoutValues,createClient,start};
if(typeof module!=='undefined'&&module.exports)module.exports=api;else scope.AtlasContent=api;
})(typeof window!=='undefined'?window:globalThis);
