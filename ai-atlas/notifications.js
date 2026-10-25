(function(scope){
'use strict';
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const notificationLabel=total=>`Нерешённые замечания: ${total}`;
function notificationItems(notes){
 return notes.length?notes.map(note=>`<button type="button" class="notification-item" data-notification-note="${esc(note.id)}"><strong>${esc(note.topic_title||note.topic_id)}</strong><span>${esc(note.summary)}</span><small>${note.status==='accepted'?'Принято · ещё не выполнено':'На рассмотрении'}</small></button>`).join(''):'<p class="notification-empty">Нет нерешённых замечаний.</p>';
}
async function fetchNotificationPages(api,pages=1){
 let next=null,total=0;const items=new Map();
 for(let page=0;page<pages;page++){
  const data=await api('/api/editorial-notifications'+(next?'?before='+encodeURIComponent(next):''));
  total=data.total;for(const note of data.notes)items.set(note.id,note);next=data.next;
  if(!next)break;
 }
 return {total,notes:[...items.values()],next};
}
function create({anchors,api,onSelect,document:doc=scope.document}){
 let data=null,error='',pending=null,again=false,pages=1,lastActivity=Date.now();
 const controls=anchors.map((anchor,index)=>{
  const wrapper=doc.createElement('div'),id='atlas-notifications-'+index;wrapper.className='atlas-notifications';
  wrapper.innerHTML=`<button type="button" class="notification-trigger" popovertarget="${id}" aria-controls="${id}" aria-expanded="false" aria-haspopup="dialog" aria-label="Замечания: загрузка"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M10 21h4"/></svg><span class="notification-badge" aria-hidden="true">…</span></button>
   <div id="${id}" class="notification-menu" popover="auto" role="dialog" aria-label="Нерешённые замечания"><header><strong>Замечания по разделам</strong><span data-notification-total></span></header><div class="notification-list"><p class="notification-empty">Загружаю…</p></div><p class="notification-error" role="status" hidden></p><button type="button" class="notification-more" data-notification-more hidden>Показать ещё</button><button type="button" class="notification-more" data-notification-retry hidden>Повторить загрузку</button></div>`;
  anchor.before(wrapper);
  const button=wrapper.querySelector('.notification-trigger'),popup=wrapper.querySelector('[popover]');
  const position=()=>{const rect=button.getBoundingClientRect(),width=Math.min(440,scope.innerWidth-24);popup.style.width=width+'px';popup.style.left=Math.max(12,Math.min(rect.right-width,scope.innerWidth-width-12))+'px';popup.style.top=Math.max(12,rect.bottom+8)+'px';popup.style.maxHeight=Math.max(100,scope.innerHeight-rect.bottom-20)+'px';};
  popup.addEventListener('beforetoggle',event=>{button.setAttribute('aria-expanded',String(event.newState==='open'));if(event.newState==='open'){lastActivity=Date.now();position();refresh();}});
  wrapper.addEventListener('keydown',event=>{if(event.key==='Escape'&&popup.matches(':popover-open'))event.stopPropagation();});
  wrapper.addEventListener('click',async event=>{
   const item=event.target.closest('[data-notification-note]');
   if(item){popup.hidePopover();await onSelect(item.dataset.notificationNote);return;}
   if(event.target.closest('[data-notification-more]')){pages++;refresh();}
   if(event.target.closest('[data-notification-retry]'))refresh();
  });
  scope.addEventListener('resize',()=>{if(popup.matches(':popover-open'))position();});
  return {button,popup};
 });
 function render(){
  for(const {button,popup} of controls){
   button.setAttribute('aria-label',data?notificationLabel(data.total):error?'Замечания: не удалось загрузить':'Замечания: загрузка');
   button.querySelector('.notification-badge').textContent=data?(data.total>99?'99+':String(data.total)):error?'!':'…';
   button.classList.toggle('has-notifications',Boolean(data?.total));
   popup.querySelector('[data-notification-total]').textContent=data?String(data.total):'';
   const list=popup.querySelector('.notification-list'),scroll=list.scrollTop,focused=doc.activeElement?.dataset?.notificationNote;
   const html=data?notificationItems(data.notes):`<p class="notification-empty">${error?'Список пока недоступен.':'Загружаю…'}</p>`;
   if(list.innerHTML!==html){list.innerHTML=html;list.scrollTop=scroll;if(focused)[...list.querySelectorAll('[data-notification-note]')].find(el=>el.dataset.notificationNote===focused)?.focus({preventScroll:true});}
   const errorElement=popup.querySelector('.notification-error');errorElement.textContent=error;errorElement.hidden=!error;
   const more=popup.querySelector('[data-notification-more]');more.hidden=!data?.next;more.disabled=Boolean(pending);
   popup.querySelector('[data-notification-retry]').hidden=!error;
  }
 }
 function refresh(){
  if(pending){again=true;return pending;}
  pending=(async()=>{
   try{
    do{again=false;try{data=await fetchNotificationPages(api,pages);pages=Math.max(1,Math.ceil(data.notes.length/20));error='';}catch(cause){error=cause.message||'Не удалось загрузить замечания.';}render();}while(again);
   }finally{pending=null;render();}
  })();
  render();return pending;
 }
 for(const event of ['pointerdown','keydown','wheel'])doc.addEventListener(event,()=>{lastActivity=Date.now();},{passive:true});
 doc.addEventListener('visibilitychange',()=>{if(!doc.hidden){lastActivity=Date.now();refresh();}});
 scope.addEventListener('focus',()=>{lastActivity=Date.now();if(!doc.hidden)refresh();});
 scope.setInterval(()=>{if(!doc.hidden&&Date.now()-lastActivity<300000)refresh();},30000);
 refresh();return {refresh};
}
const api={notificationLabel,notificationItems,fetchNotificationPages,create};
if(typeof module!=='undefined'&&module.exports)module.exports=api;else scope.AtlasNotifications=api;
})(typeof window!=='undefined'?window:globalThis);
