(function(scope){
'use strict';
const object=value=>value&&typeof value==='object'&&!Array.isArray(value);
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const bounded=(value,min,max)=>Number.isFinite(value)?Math.max(min,Math.min(max,value)):null;
const editAttr=(path,part)=>path?` data-edit-path="${esc(path)}"${part===undefined?'':` data-edit-part="${part}"`}`:'';

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
  const order=[...families.slice(0,-1).map(([title])=>title),'Другие материалы',families.at(-1)[0]];
  return order.filter(title=>groups.has(title)).map(title=>({title,links:groups.get(title)}));
}

function apply(records,{data,index,lessons,presentation}){
  if(!Array.isArray(records)||records.some(r=>!object(r)||typeof r.id!=='string'||!object(r.payload)))throw Error('Invalid content records');
  const originals=new Map();
  (function visit(n){originals.set(n.id,n);for(const child of n.children||[])visit(child);})(data.root);
  const replace=(target,source)=>{for(const key of Object.keys(target))delete target[key];for(const [key,value] of Object.entries(source))if(!['__proto__','constructor','prototype'].includes(key))target[key]=value;};
  for(const {id,payload,revision} of records){
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
    presentation[id]={layout:object(payload.layout)?payload.layout:{},sections:object(payload.sections)?payload.sections:{},revision:String(revision||'0')};
  }
}

function categoryOutline(points,radius){
  const cloud=points.flatMap(p=>Array.from({length:8},(_,i)=>({x:p.x+Math.cos(i*Math.PI/4)*radius,y:p.y+Math.sin(i*Math.PI/4)*radius}))).sort((a,b)=>a.x-b.x||a.y-b.y);
  const cross=(a,b,c)=>(b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
  const half=values=>{const result=[];for(const p of values){while(result.length>1&&cross(result.at(-2),result.at(-1),p)<=0)result.pop();result.push(p);}return result;};
  const lower=half(cloud),upper=half([...cloud].reverse()),hull=[...lower.slice(0,-1),...upper.slice(0,-1)];
  if(!hull.length)return '';
  const mid=(a,b)=>`${(a.x+b.x)/2} ${(a.y+b.y)/2}`;
  return `M${mid(hull.at(-1),hull[0])}`+hull.map((p,i)=>`Q ${p.x} ${p.y} ${mid(p,hull[(i+1)%hull.length])}`).join(' ')+'Z';
}

function chart(block){
  const colors={blue:'#2563eb',orange:'#f97316',green:'#16a34a',purple:'#9333ea',gray:'#64748b'},palette=Object.values(colors);
  const series=(Array.isArray(block.series)?block.series:[]).slice(0,8).filter(object).map((s,i)=>({...s,color:colors[s.color]||palette[i%palette.length],values:(Array.isArray(s.values)?s.values:[]).filter(p=>object(p)&&Number.isFinite(p.x)&&Number.isFinite(p.y)).slice(0,1000)}));
  const points=series.flatMap(s=>s.values);if(!points.length)return '';
  const pointSize=bounded(block.pointSize,5,14)??10,outlined=series.some(s=>s.outline);
  const domain=(axis,explicit)=>{
    const values=points.map(p=>p[axis]);let low=Math.min(...values),high=Math.max(...values);
    if(Array.isArray(explicit)&&explicit.length===2&&explicit.every(Number.isFinite)&&explicit[0]<low&&explicit[1]>high){
      const span=explicit[1]-explicit[0],minimum=span*(outlined?.15:.08);
      if(low-explicit[0]>=minimum&&explicit[1]-high>=minimum)return explicit;
      const reserve=span*(outlined?.18:.09);return [Math.min(explicit[0],low-reserve),Math.max(explicit[1],high+reserve)];
    }
    if(low===high){low-=.5;high+=.5;}
    const margin=(high-low)*Math.max(bounded(block.axisPadding,.05,.3)??.12,outlined?.2:.12);return[low-margin,high+margin];
  };
  const [minX,maxX]=domain('x',block.xDomain),[minY,maxY]=domain('y',block.yDomain);
  const x=v=>70+(v-minX)/(maxX-minX)*460,y=v=>285-(v-minY)/(maxY-minY)*235;
  const number=n=>Number(n.toFixed(2)).toLocaleString('ru');
  let svg='';
  for(let i=0;i<=4;i++){
    const vy=minY+(maxY-minY)*i/4,vx=minX+(maxX-minX)*i/4;
    svg+=`<path d="M70 ${y(vy)}H530" stroke="#e2e5e9"/><text x="60" y="${y(vy)+7}" text-anchor="end">${number(vy)}</text><text x="${x(vx)}" y="313" text-anchor="middle">${number(vx)}</text>`;
  }
  svg+='<path d="M70 50V285H530" fill="none" stroke="#6b7280"/>';
  svg+=series.filter(s=>s.outline&&s.values.length).map(s=>`<path class="chart-category-outline" d="${categoryOutline(s.values.map(p=>({x:x(p.x),y:y(p.y)})),pointSize+14)}" fill="${s.color}" fill-opacity="0.07" stroke="${s.color}" stroke-width="2.5"/>`).join('');
  const marker=(s,p)=>{
    const cx=x(p.x),cy=y(p.y),fill=s.hollow?'#fff':s.color,stroke=s.hollow?s.color:'#fff',r=pointSize;
    const title=`<title>${esc(p.label||s.name)}: ${number(p.x)}, ${number(p.y)}</title>`;
    const shape=s.marker==='diamond'?`<path d="M${cx} ${cy-r-2}l${r+2} ${r+2}-${r+2} ${r+2}-${r+2}-${r+2}Z" fill="${fill}" stroke="${stroke}" stroke-width="2.5">${title}</path>`:
      s.marker==='square'?`<rect x="${cx-r}" y="${cy-r}" width="${r*2}" height="${r*2}" fill="${fill}" stroke="${stroke}" stroke-width="2.5">${title}</rect>`:
      `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="2.5">${title}</circle>`;
    const label=String(p.label||''),display=label.length>18?label.slice(0,17)+'…':label,labelWidth=label.length>18?190:Math.max(12,display.length*11),right=cx+r+7+labelWidth>530;
    return shape+(label?`<text x="${cx+(right?-r-7:r+7)}" y="${cy<90?cy+25:cy-10}" text-anchor="${right?'end':'start'}"${label.length>18?' textLength="190" lengthAdjust="spacingAndGlyphs"':''} fill="${s.color}" stroke="none">${esc(display)}</text>`:'');
  };
  svg+=series.map(s=>`${block.type==='scatter'?'':`<polyline fill="none" stroke="${s.color}" stroke-width="3" points="${s.values.map(p=>`${x(p.x)},${y(p.y)}`).join(' ')}"/>`}${s.values.map(p=>marker(s,p)).join('')}`).join('');
  const symbol=s=>s.marker==='diamond'?(s.hollow?'◇':'◆'):s.marker==='square'?(s.hollow?'□':'■'):(s.hollow?'○':'●');
  return `<figure class="content-chart"><figcaption>${esc(block.title)}</figcaption><svg viewBox="0 0 560 360" role="img" aria-label="${esc(block.title)}"><g font-size="20" fill="#374151">${svg}<text x="300" y="350" text-anchor="middle">${esc(block.xLabel)}</text><text x="70" y="25">${esc(block.yLabel)}</text></g></svg><div class="content-chart-legend">${series.map(s=>`<span style="color:${s.color}">${symbol(s)} ${esc(s.name)}</span>`).join('')}</div></figure>`;
}

function mediaUrl(value){
  if(typeof value!=='string'||value.length>8192||/[\s\\\x00-\x1f\x7f]/.test(value))return '';
  try{const url=new URL(value);return /^https?:$/.test(url.protocol)&&url.hostname&&!url.username&&!url.password&&url.port!=='0'?value:'';}catch{return '';}
}

function illustration(block,path=''){
  let src='';
  if(block.kind==='svg'&&typeof block.svg==='string'&&block.svg.length<=200000){
    // Image documents have no access to the page DOM and do not execute scripts.
    // Never insert publication SVG with innerHTML, object or iframe.
    try{src='data:image/svg+xml;charset=utf-8,'+encodeURIComponent(block.svg);}catch{}
  }else if(block.kind==='image'){
    if(typeof block.src==='string'&&/^https:\/\/[a-z0-9-]+\.public\.blob\.vercel-storage\.com\/illustrations\/[a-f0-9]{64}\.webp$/.test(block.src))src=block.src;
  }
  const source=mediaUrl(block.sourceUrl),caption=typeof block.caption==='string'?block.caption:'',alt=typeof block.alt==='string'?block.alt:'';
  const attribution=source?`<a href="${esc(source)}" target="_blank" rel="noopener noreferrer">${esc(block.sourceLabel||'Источник иллюстрации')} ↗</a>`:esc(block.sourceLabel||'');
  return `<figure class="content-illustration">${src?`<img src="${esc(src)}" alt="${esc(alt)}" loading="lazy" decoding="async" referrerpolicy="no-referrer">`:`<p class="content-media-error">Изображение недоступно. ${esc(alt)}</p>`}${caption||attribution?`<figcaption>${caption?`<p${editAttr(path&&path+'.caption')}>${esc(caption)}</p>`:''}${attribution?`<p class="content-media-source">${attribution}</p>`:''}</figcaption>`:''}</figure>`;
}

function renderSections(blocks,{rich=esc,code,experiment,path=''}={},depth=0,budget={count:0}){
  if(!Array.isArray(blocks))return null;
  if(depth>3)return '';
  return blocks.slice(0,100).map((block,index)=>{
    if(!object(block)||++budget.count>100)return '';
    const width=bounded(block.width,320,1600),emphasis=['info','task','warning'].includes(block.emphasis)?block.emphasis:'';
    let body='';
    const location=path?path+'.'+index:'';
    if(block.kind==='group'){
      const columns=[1,2,3].includes(block.columns)?block.columns:1,gap=bounded(block.gap,12,48)??24;
      body=`<div class="content-group-grid" style="--group-columns:${columns};--group-gap:${gap}px">${renderSections(block.blocks,{rich,code,experiment,path:location&&location+'.blocks'},depth+1,budget)||''}</div>`;
    }else if(block.kind==='image'||block.kind==='svg')body=illustration(block,location);
    else if(block.kind==='chart')body=chart(block);
    else if(block.kind==='experiment')body=experiment?.(block.experiment)||'';
    else if(block.kind==='code')body=code?code(block):`<pre><code>${esc(block.code)}</code></pre>`;
    else if(block.kind==='list')body=`<ul>${(Array.isArray(block.items)?block.items:[]).map((item,i)=>`<li${editAttr(location&&location+'.items.'+i)}>${rich(item,true)}</li>`).join('')}</ul>`;
    else body=String(block.body||'').split(/\n\s*\n/).map((text,i)=>`<p${editAttr(location&&location+'.body',i)}>${rich(text,true).replace(/\n/g,'<br>')}</p>`).join('');
    return `<section class="reading-section content-block${block.kind==='group'?' content-group':''}${emphasis?' content-'+emphasis:''}"${location?` data-edit-section="${esc(location)}"`:''}${width?` style="max-width:${width}px"`:''}>${block.kind!=='chart'&&block.title?`<h2>${esc(block.title)}</h2>`:''}${body}</section>`;
  }).join('');
}

function layoutValues(layout={}){
  return {contentPadding:bounded(layout.contentPadding,8,64),sectionGap:bounded(layout.sectionGap,12,64),readerWidth:bounded(layout.readerWidth,700,1800),textWidth:bounded(layout.textWidth,500,1600),chartsColumns:[1,2].includes(layout.chartsColumns)?layout.chartsColumns:null};
}

function createClient({fetch:fetcher,onChange,onError=()=>{},now=()=>Date.now()}){
  let revision='0',pending=null,lastAttempt=0,requiredRevision='0';
  return {
    get revision(){return revision;},
    refresh({minRevision='0'}={}){
      if(/^\d+$/.test(minRevision)&&BigInt(minRevision)>BigInt(requiredRevision))requiredRevision=String(minRevision);
      if(pending)return pending;
      if(BigInt(revision)>=BigInt(requiredRevision)&&lastAttempt&&now()-lastAttempt<2000)return Promise.resolve(false);
      lastAttempt=now();
      pending=(async()=>{
        try{
          // A publication may arrive while an older read is still in flight.
          for(let attempt=0;attempt<3;attempt++){
            const response=await fetcher('/api/content?since='+revision,{cache:'no-store',signal:AbortSignal.timeout(12000)});
            if(!response.ok)throw Error('Content unavailable');
            const update=await response.json();
            if(!/^\d+$/.test(update.revision)||!Array.isArray(update.records)||BigInt(update.revision)<BigInt(revision))throw Error('Invalid content response');
            if(update.records.length)onChange(update.records,revision!=='0');
            revision=String(update.revision);
            if(BigInt(revision)>=BigInt(requiredRevision))return true;
          }
          return false;
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
  let lastActivity=Date.now(),lastCheck=0;
  const refresh=()=>{if(!document.hidden){lastCheck=Date.now();client.refresh();}};
  for(const event of ['pointerdown','keydown','wheel'])document.addEventListener(event,()=>{lastActivity=Date.now();},{passive:true});
  scope.addEventListener('focus',()=>{lastActivity=Date.now();refresh();});
  document.addEventListener('visibilitychange',()=>{if(!document.hidden){lastActivity=Date.now();refresh();}});
  // Follow an open material promptly; pause abandoned tabs after five idle minutes.
  scope.setInterval(()=>{
    if(Date.now()-lastActivity<300000&&(document.getElementById('reader')?.open||Date.now()-lastCheck>=30000))refresh();
  },3000);
  refresh();return client;
}

const api={apply,groupSources,renderSections,layoutValues,createClient,start};
if(typeof module!=='undefined'&&module.exports)module.exports=api;else scope.AtlasContent=api;
})(typeof window!=='undefined'?window:globalThis);
