(function(scope){
'use strict';
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
function indexTree(root){const index=new Map();function visit(n,parent,branch){if(index.has(n.id))throw Error('Duplicate '+n.id);index.set(n.id,{...n,parent,branch:branch||n.id});for(const c of n.children||[])visit(c,n.id,parent?branch:c.id);}visit(root,null,null);return index;}
const countLeaves=n=>n.children?n.children.reduce((a,c)=>a+countLeaves(c),0):1;
const normalize=s=>String(s||'').toLowerCase().replace(/ё/g,'е');
function search(index,q){const terms=normalize(q).trim().split(/\s+/).filter(Boolean);if(!terms.length)return[];return [...index.values()].filter(n=>n.parent).map(n=>{const name=normalize(n.name+' '+(n.full||'')+' '+n.id);const all=normalize([name,n.aliases,n.what,n.how,n.example,n.tasks,n.lab,n.note,n.training,n.input,n.output,n.criteria,...(n.steps||[])].join(' '));return{node:n,score:terms.every(t=>all.includes(t))?terms.reduce((v,t)=>v+(name.includes(t)?4:1),0):0};}).filter(r=>r.score).sort((a,b)=>b.score-a.score).map(r=>r.node);}
function layout(root){
const groups=[],cards=[],edges=[];const cw=360,ch=416,gap=32,header=128;
const clusterWidth=3*(cw+gap)+gap,columnStep=clusterWidth+64,parentWidth=2*clusterWidth+128;
function cluster(n,x,y,branch,nested){const cols=Math.min(3,n.children.length), rows=Math.ceil(n.children.length/cols);const group={id:n.id,branch,x,y,w:cols*(cw+gap)+gap,h:rows*(ch+gap)+header+gap,nested};groups.push(group);n.children.forEach((c,i)=>cards.push({id:c.id,parent:n.id,branch,x:x+gap+(i%cols)*(cw+gap),y:y+header+Math.floor(i/cols)*(ch+gap),w:cw,h:ch}));return group;}
let x=0;
for(const branch of root.children){if(branch.children.every(n=>!n.children)){const g=cluster(branch,x,0,branch.id,false);x+=g.w+200;continue;}
 const parent={id:branch.id,branch:branch.id,x,y:0,w:parentWidth,h:0,nested:false};groups.push(parent);let rowY=160,maxBottom=0;
 branch.children.forEach((n,i)=>{if(i&&i%2===0)rowY=maxBottom+70;const g=cluster(n,x+32+(i%2)*columnStep,rowY,branch.id,true);maxBottom=Math.max(maxBottom,g.y+g.h);});parent.h=maxBottom+32;x+=parent.w+200;
}
const roots=new Map(groups.filter(g=>!g.nested).map(g=>[g.id,g]));
const modelX=Math.max(roots.get('tasks').w,roots.get('small').w)+220,trainingX=modelX+roots.get('models').w+220;
const positions={tasks:[0,0],models:[modelX,0],training:[trainingX,0],small:[0,roots.get('tasks').h+220],practice:[trainingX,roots.get('training').h+220]};
for(const branch of root.children){const g=roots.get(branch.id),[nx,ny]=positions[branch.id],dx=nx-g.x,dy=ny-g.y;for(const item of [...groups,...cards])if(item.branch===branch.id){item.x+=dx;item.y+=dy;}}
return{groups,cards,edges};
}
function fitBox(b,w,h,p=46,max=1){const scale=clamp(Math.min((w-p*2)/b.w,(h-p*2)/b.h,max),.035,1.65);return{scale,x:(w-b.w*scale)/2-b.x*scale,y:(h-b.h*scale)/2-b.y*scale};}
function zoomAt(v,s,x,y){s=clamp(s,.035,1.65);const r=s/v.scale;return{scale:s,x:x-(x-v.x)*r,y:y-(y-v.y)*r};}
function restoreRoute(s,index){try{const x=JSON.parse(s);return new Set(Array.isArray(x)?x.filter(id=>typeof id==='string'&&index.has(id)&&id!=='root'):[]);}catch{return new Set();}}
// Orthogonal routing through free space, including the gaps between cards.
// Group bodies unrelated to either endpoint are obstacles too, so long links
// travel around whole sections instead of threading through their contents.
function routingObstacles(cards, groups, source, target, clearance=10) {
  const allowed=new Set([source.parent,source.branch,target.parent,target.branch]);
  const inflate=r=>({x:r.x-clearance,y:r.y-clearance,w:r.w+2*clearance,h:r.h+2*clearance});
  const blockedGroups=groups.filter(g=>!allowed.has(g.id));
  const visibleGroups=blockedGroups.filter(g=>!blockedGroups.some(p=>p!==g&&g.x>=p.x&&g.y>=p.y&&g.x+g.w<=p.x+p.w&&g.y+g.h<=p.y+p.h));
  const covered=c=>visibleGroups.some(g=>c.x>=g.x&&c.y>=g.y&&c.x+c.w<=g.x+g.w&&c.y+c.h<=g.y+g.h);
  return [
    ...visibleGroups.map(g=>inflate({x:g.x,y:g.y-30,w:g.w,h:g.h+30})),
    ...cards.filter(c=>!covered(c)).map(inflate),
    ...groups.filter(g=>allowed.has(g.id)).map(g=>inflate({x:g.x+20,y:g.y-30,w:g.w-40,h:105}))
  ];
}
function segmentBlocked(a,b,rectangles) {
  const epsilon=1e-7;
  return rectangles.some(r=>{
    if(a.x===b.x) return a.x>r.x+epsilon&&a.x<r.x+r.w-epsilon&&Math.max(a.y,b.y)>r.y+epsilon&&Math.min(a.y,b.y)<r.y+r.h-epsilon;
    if(a.y===b.y) return a.y>r.y+epsilon&&a.y<r.y+r.h-epsilon&&Math.max(a.x,b.x)>r.x+epsilon&&Math.min(a.x,b.x)<r.x+r.w-epsilon;
    throw Error('Routing segments must be orthogonal');
  });
}
function simplifyRoute(points) {
  const result=[];
  for(const point of points){
    if(result.length&&result[result.length-1].x===point.x&&result[result.length-1].y===point.y)continue;
    while(result.length>1){const a=result[result.length-2],b=result[result.length-1];if((a.x===b.x&&b.x===point.x)||(a.y===b.y&&b.y===point.y))result.pop();else break;}
    result.push(point);
  }
  return result;
}
function roundedRoute(points,radius=8) {
  if(points.length<2)return '';
  let d=`M${points[0].x} ${points[0].y}`;
  for(let i=1;i<points.length-1;i++){
    const a=points[i-1],b=points[i],c=points[i+1];
    const incoming=Math.hypot(b.x-a.x,b.y-a.y),outgoing=Math.hypot(c.x-b.x,c.y-b.y);
    const r=Math.min(radius,incoming/2,outgoing/2);
    const entry={x:b.x+(a.x-b.x)*r/incoming,y:b.y+(a.y-b.y)*r/incoming};
    const exit={x:b.x+(c.x-b.x)*r/outgoing,y:b.y+(c.y-b.y)*r/outgoing};
    d+=` L${entry.x} ${entry.y} Q${b.x} ${b.y} ${exit.x} ${exit.y}`;
  }
  const end=points[points.length-1];return d+` L${end.x} ${end.y}`;
}
function routeConnection(cards,groups,source,target) {
  const clearance=10,obstacles=routingObstacles(cards,groups,source,target,clearance);
  function ports(c){return [
    {point:{x:c.x+c.w,y:c.y+c.h/2},x:c.x+c.w+clearance,y:c.y+c.h/2,dir:0},
    {point:{x:c.x,y:c.y+c.h/2},x:c.x-clearance,y:c.y+c.h/2,dir:0},
    {point:{x:c.x+c.w/2,y:c.y},x:c.x+c.w/2,y:c.y-clearance,dir:1},
    {point:{x:c.x+c.w/2,y:c.y+c.h},x:c.x+c.w/2,y:c.y+c.h+clearance,dir:1}
  ].filter(p=>!obstacles.some(r=>p.x>r.x&&p.x<r.x+r.w&&p.y>r.y&&p.y<r.y+r.h));}
  const starts=ports(source),ends=ports(target);if(!starts.length||!ends.length)return null;
  const xs=[...new Set([...obstacles.flatMap(r=>[r.x,r.x+r.w]),...starts.map(p=>p.x),...ends.map(p=>p.x)])].sort((a,b)=>a-b);
  const ys=[...new Set([...obstacles.flatMap(r=>[r.y,r.y+r.h]),...starts.map(p=>p.y),...ends.map(p=>p.y)])].sort((a,b)=>a-b);
  xs.unshift(xs[0]-40);xs.push(xs[xs.length-1]+40);ys.unshift(ys[0]-40);ys.push(ys[ys.length-1]+40);
  const nx=xs.length,ny=ys.length,total=nx*ny*2;
  const dist=new Float64Array(total).fill(Infinity),previous=new Int32Array(total).fill(-1),origin=new Int8Array(total).fill(-1);
  const heap=[];
  function push(item){let i=heap.length;heap.push(item);while(i){const p=(i-1)>>1;if(heap[p].priority<=item.priority)break;heap[i]=heap[p];i=p;}heap[i]=item;}
  function pop(){const first=heap[0],last=heap.pop();if(heap.length){let i=0;while(i*2+1<heap.length){let c=i*2+1;if(c+1<heap.length&&heap[c+1].priority<heap[c].priority)c++;if(heap[c].priority>=last.priority)break;heap[i]=heap[c];i=c;}heap[i]=last;}return first;}
  const nodeId=(x,y)=>ys.indexOf(y)*nx+xs.indexOf(x);
  const goals=new Map(ends.map(p=>[nodeId(p.x,p.y),p]));
  const heuristic=(x,y)=>Math.min(...ends.map(p=>Math.abs(p.x-x)+Math.abs(p.y-y)));
  starts.forEach((p,i)=>{const state=nodeId(p.x,p.y)*2+p.dir;dist[state]=clearance;origin[state]=i;push({state,cost:clearance,priority:clearance+heuristic(p.x,p.y)});});
  const clearEdges=new Map();let finish=-1,best=Infinity,finishPort=null;
  while(heap.length){
    const item=pop();if(item.cost!==dist[item.state])continue;if(item.priority>=best)break;
    const node=item.state>>1,dir=item.state%2,ix=node%nx,iy=Math.floor(node/nx),a={x:xs[ix],y:ys[iy]};
    const end=goals.get(node);
    if(end){const cost=item.cost+clearance+(dir!==end.dir?24:0);if(cost<best){best=cost;finish=item.state;finishPort=end;}continue;}
    for(const [dx,dy,nextDir] of [[1,0,0],[-1,0,0],[0,1,1],[0,-1,1]]){
      const jx=ix+dx,jy=iy+dy;if(jx<0||jx>=nx||jy<0||jy>=ny)continue;
      const nextNode=jy*nx+jx,b={x:xs[jx],y:ys[jy]},key=Math.min(node,nextNode)+':'+Math.max(node,nextNode);
      let clear=clearEdges.get(key);if(clear===undefined){clear=!segmentBlocked(a,b,obstacles);clearEdges.set(key,clear);}if(!clear)continue;
      const state=nextNode*2+nextDir,cost=item.cost+Math.abs(a.x-b.x)+Math.abs(a.y-b.y)+(dir!==nextDir?24:0);
      if(cost>=dist[state])continue;dist[state]=cost;previous[state]=item.state;origin[state]=origin[item.state];push({state,cost,priority:cost+heuristic(b.x,b.y)});
    }
  }
  // An impossible route is omitted; never fall back to a line through content.
  if(finish<0)return null;
  const points=[];for(let s=finish;s>=0;s=previous[s]){const node=s>>1;points.push({x:xs[node%nx],y:ys[Math.floor(node/nx)]});}
  points.reverse();points.unshift(starts[origin[finish]].point);points.push(finishPort.point);
  return simplifyRoute(points);
}

// The tool belongs to the user; modifiers are only overrides. A pointer gesture
// owns its mode until all its pointers end, regardless of key-up ordering.
function createPanController(){
  let tool='select',mode=null,blockClick=false;
  const modifiers=new Set(),pointers=new Set();
  const effective=()=>mode||(modifiers.size?'pan':tool);
  return {
    get tool(){return tool;},get effective(){return effective();},
    get dragging(){return mode==='pan'&&pointers.size>0;},
    setTool(next){tool=next==='pan'?'pan':'select';},
    modifier(key,down){if(down)modifiers.add(key);else modifiers.delete(key);},
    freshPointer(){blockClick=false;},
    begin({id,button=0,onControl=false,shiftKey=false}){
      if(button!==0&&button!==1)return false;
      if(!pointers.size)mode=button===1||shiftKey||effective()==='pan'||!onControl?'pan':'select';
      pointers.add(id);return mode==='pan';
    },
    end(id){if(!pointers.delete(id))return;blockClick=mode==='pan';if(!pointers.size)mode=null;},
    blocksClick(keyboard=false){return !keyboard&&(blockClick||mode==='pan');},
    reset(){blockClick=mode==='pan';modifiers.clear();pointers.clear();mode=null;}
  };
}
function createTermRenderer(glossary={}){
  const escape=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const aliases=new Map();
  for(const [key,term] of Object.entries(glossary))for(const alias of term.aliases||[key])aliases.set(alias.toLowerCase(),key);
  const pattern=[...aliases.keys()].sort((a,b)=>b.length-a.length).map(s=>s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('|');
  if(!pattern)return escape;
  const re=new RegExp('(?<![\\p{L}\\p{N}_])('+pattern+')(?![\\p{L}\\p{N}_])','giu');
  return (text,focusable=false)=>{
    let out='',offset=0;
    for(const match of String(text??'').matchAll(re)){
      const key=aliases.get(match[0].toLowerCase()),term=glossary[key];
      out+=escape(String(text).slice(offset,match.index));
      out+=`<abbr class="term" data-term="${escape(key)}" title="${escape(term.full+' — '+term.description)}"${focusable?' tabindex="0"':''}>${escape(match[0])}</abbr>`;
      offset=match.index+match[0].length;
    }
    return out+escape(String(text??'').slice(offset));
  };
}
const api={createPanController,createTermRenderer,routingObstacles,segmentBlocked,roundedRoute,routeConnection,clamp,indexTree,countLeaves,search,layout,fitBox,zoomAt,restoreRoute};if(typeof module!=='undefined'&&module.exports)module.exports=api;else scope.AtlasCore=api;
})(typeof window!=='undefined'?window:globalThis);
