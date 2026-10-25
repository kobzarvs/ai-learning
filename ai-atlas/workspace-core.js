(function(scope){
'use strict';
const uid=()=>typeof crypto!=='undefined'&&crypto.randomUUID?crypto.randomUUID():'item-'+Date.now().toString(36)+'-'+Math.random().toString(36).slice(2);
const number=(v,fallback=0)=>Number.isFinite(v)?Math.max(-100000,Math.min(100000,v)):fallback;
const text=(v,fallback='')=>typeof v==='string'?v:fallback;
function defaultBoard(node,lessons={}){
 const cards=[];
 const add=(id,type,ref,title,body,x,y)=>cards.push({id,type,ref,title,body,x,y,w:420,h:330,groupId:null});
 add('article:'+node.id,'article',node.id,'Теория',node.what,80,90);
 if(node.children){node.children.forEach((child,i)=>add('topic:'+child.id,'topic',child.id,child.name,child.what,80+(i%3)*460,470+Math.floor(i/3)*380));}
 else{
  add('python:'+node.id,'example',node.id,'Python · практический пример','Открыть код, команду запуска и разбор результата.',540,90);
  if(lessons[node.id]?.rust)add('rust:'+node.id,'rust',node.id,'Rust + Burn · тот же принцип','Пример для актуальной версии Burn с командой запуска.',1000,90);
  add('exercise:'+node.id,'exercise',node.id,'Практика',lessons[node.id]?.exercise||'Запиши ожидаемый результат, измени одно условие и сравни ответы.',80,470);
 }
 return {version:1,layoutVersion:2,cards,groups:[]};
}
function normalizeBoard(raw,fallback){
 if(!raw||!Array.isArray(raw.cards)||!Array.isArray(raw.groups))return structuredClone(fallback);
 const seen=new Set();
 const cards=raw.cards.filter(c=>c&&typeof c.id==='string'&&!seen.has(c.id)&&(seen.add(c.id),true)).slice(0,1000).map(c=>({
  id:c.id,type:['article','topic','example','rust','exercise','note'].includes(c.type)?c.type:'note',ref:text(c.ref),
  title:text(c.title,'Без названия'),body:text(c.body),x:number(c.x),y:number(c.y),w:Math.max(280,number(c.w,360)),h:Math.max(180,number(c.h,230)),groupId:typeof c.groupId==='string'?c.groupId:null
 }));
 const groupIds=new Set();
 const groups=raw.groups.filter(g=>g&&typeof g.id==='string'&&!groupIds.has(g.id)&&(groupIds.add(g.id),true)).slice(0,200).map(g=>({id:g.id,title:text(g.title,'Группа'),x:number(g.x),y:number(g.y),w:Math.max(400,number(g.w,800)),h:Math.max(260,number(g.h,520))}));
 const validGroups=new Set(groups.map(g=>g.id));for(const c of cards)if(!validGroups.has(c.groupId))c.groupId=null;
 return upgradeLayout({version:1,layoutVersion:raw.layoutVersion||1,cards,groups});
}
function ensureLessonCards(board,node,lessons){
 if(node.children||!lessons[node.id]?.rust||board.cards.some(c=>c.type==='rust'&&c.ref===node.id))return false;
 const card=defaultBoard(node,lessons).cards.find(c=>c.type==='rust');
 card.y=Math.max(90,...board.cards.map(c=>c.y+c.h+50),...board.groups.map(g=>g.y+g.h+50));
 card.x=80;board.cards.push(card);return true;
}
// One-time expansion for 20px typography. Relative arrangement and all text
// survive; frame bounds are refitted around their existing member cards.
function upgradeLayout(board){
 if(board.layoutVersion>=2)return board;
 for(const card of board.cards){card.x*=1.22;card.y*=1.45;card.w=Math.max(420,card.w*1.22);card.h=Math.max(330,card.h*1.45);}
 for(const group of board.groups){group.x*=1.22;group.y*=1.45;group.w*=1.22;group.h*=1.45;fitGroup(board,group.id);}
 board.layoutVersion=2;return board;
}
function bounds(items,padding=32,header=56){
 if(!items.length)return{x:80,y:70,w:800,h:520};
 const x=Math.min(...items.map(c=>c.x))-padding,y=Math.min(...items.map(c=>c.y))-header;
 return{x,y,w:Math.max(...items.map(c=>c.x+c.w))-x+padding,h:Math.max(...items.map(c=>c.y+c.h))-y+padding};
}
function selectRect(cards,r){const a={x:Math.min(r.x,r.x+r.w),y:Math.min(r.y,r.y+r.h),w:Math.abs(r.w),h:Math.abs(r.h)};return cards.filter(c=>c.x<a.x+a.w&&c.x+c.w>a.x&&c.y<a.y+a.h&&c.y+c.h>a.y).map(c=>c.id);}
function groupCards(board,ids,title){
 const selected=board.cards.filter(c=>ids.includes(c.id));const group={id:uid(),title:title.trim()||'Новая группа',...bounds(selected)};
 board.groups.push(group);for(const c of selected)c.groupId=group.id;return group;
}
function moveGroup(board,id,dx,dy){const g=board.groups.find(g=>g.id===id);if(!g)return;g.x+=dx;g.y+=dy;for(const c of board.cards)if(c.groupId===id){c.x+=dx;c.y+=dy;}}
function fitGroup(board,id){const g=board.groups.find(g=>g.id===id);if(!g)return;const members=board.cards.filter(c=>c.groupId===id);if(members.length)Object.assign(g,bounds(members));}
function removeGroup(board,id){board.groups=board.groups.filter(g=>g.id!==id);for(const c of board.cards)if(c.groupId===id)c.groupId=null;}
function addNote(board,title,body,point,groupId=null){const g=board.groups.find(g=>g.id===groupId);const members=board.cards.filter(c=>c.groupId===groupId);
 const c={id:uid(),type:'note',ref:'',title:title.trim()||'Без названия',body,x:g?g.x+32:point.x,y:g?(members.length?Math.max(...members.map(c=>c.y+c.h))+32:g.y+70):point.y,w:420,h:330,groupId:g?.id||null};
 board.cards.push(c);if(g)fitGroup(board,g.id);return c;
}
function parseImport(source){const value=JSON.parse(source);if(value?.version!==1||!value.boards||typeof value.boards!=='object'||Array.isArray(value.boards))throw Error('Это не файл рабочих пространств Atlas.');const boards={};for(const [id,board] of Object.entries(value.boards)){if(['__proto__','prototype','constructor'].includes(id))continue;if(!board||!Array.isArray(board.cards)||!Array.isArray(board.groups))throw Error('Повреждена структура холста '+id);if(board.cards.length>1000||board.groups.length>200)throw Error('Слишком много карточек или групп на холсте '+id);boards[id]=normalizeBoard(board,{version:1,cards:[],groups:[]});}return{version:1,boards};}
const api={uid,upgradeLayout,defaultBoard,normalizeBoard,ensureLessonCards,bounds,selectRect,groupCards,moveGroup,fitGroup,removeGroup,addNote,parseImport};
if(typeof module!=='undefined'&&module.exports)module.exports=api;else scope.AtlasWorkspaceCore=api;
})(typeof window!=='undefined'?window:globalThis);
