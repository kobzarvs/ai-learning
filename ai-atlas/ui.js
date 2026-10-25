(function(scope){
'use strict';
const escape=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function highlight(code,language){
 const prism=scope.Prism,grammar=prism?.languages[language];
 return grammar?prism.highlight(String(code),grammar,language):escape(code);
}
const icons={
 copy:'<rect x="8" y="8" width="12" height="13" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"/>',
 download:'<path d="M12 3v12m-5-5 5 5 5-5M4 16v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4"/>',
 target:'<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><path d="m12 12 9-9m-5 0h5v5"/>',
 steps:'<rect x="5" y="4" width="14" height="17" rx="2"/><path d="M9 4V2h6v2M10 9h6m-6 5h6m-6 4h6M8 9h.01M8 14h.01M8 18h.01"/>',
 mechanism:'<path d="m14 6 4-4a6 6 0 0 1-7 8L4 17a2 2 0 0 0 3 3l7-7a6 6 0 0 0 8-7l-4 4-4-4Z"/>',
 example:'<rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8" cy="8" r="1.5"/><path d="m3 16 5-5 5 5 3-3 5 5"/>',
 terminal:'<rect x="2" y="4" width="20" height="16" rx="3"/><path d="m6 9 3 3-3 3m6 0h5"/>',
 result:'<path d="M4 3v17h17M8 15l4-4 4 2 5-7"/><path d="M17 6h4v4"/>',
 experiment:'<path d="M9 3h6m-5 0v6l-6 10a1.3 1.3 0 0 0 1 2h14a1.3 1.3 0 0 0 1-2L14 9V3M7 15h10"/><path d="M10 18h.01m4-1h.01"/>',
 info:'<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10h.01"/>',
 book:'<path d="M12 5C8 2 5 3 2 4v15c4-1 7-1 10 2 3-3 6-3 10-2V4c-3-1-6-2-10 1Zm0 0v16"/>',
 sources:'<path d="m10 13 4-4m-6 7-2 2a4 4 0 0 1-6-6l5-5a4 4 0 0 1 6 0m2 1 2-2a4 4 0 0 1 6 6l-5 5a4 4 0 0 1-6 0" transform="translate(2 -1) scale(.9)"/>',
 topics:'<rect x="8" y="2" width="8" height="5" rx="1"/><rect x="2" y="17" width="7" height="5" rx="1"/><rect x="15" y="17" width="7" height="5" rx="1"/><path d="M12 7v5H5v5m7-5h7v5"/>'
};
function icon(kind='book'){
 const key=Object.hasOwn(icons,kind)?kind:'book';
 return `<span class="section-icon icon-${key}" aria-hidden="true"><svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" focusable="false">${icons[key]}</svg></span>`;
}
// Require a gesture that began AND ended on the backdrop. Selecting code from
// inside a dialog and releasing outside must not dismiss the reader.
function bindDismiss(dialog,{close=()=>dialog.close(),editable=false}={}){
 let beganOutside=false;
 const outside=e=>{const r=dialog.getBoundingClientRect();return e.target===dialog&&(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom);};
 dialog.setAttribute('closedby',editable?'none':'closerequest');
 if(editable)return;
 dialog.addEventListener('cancel',e=>{e.preventDefault();e.stopPropagation?.();close();});
 dialog.addEventListener('pointerdown',e=>{beganOutside=e.button===0&&outside(e);});
 dialog.addEventListener('pointercancel',()=>{beganOutside=false;});
 dialog.addEventListener('click',e=>{const dismiss=beganOutside&&outside(e);beganOutside=false;if(dialog.open&&dismiss){e.preventDefault();close();}});
 dialog.addEventListener('close',()=>{beganOutside=false;});
}
const api={highlight,bindDismiss,icon};
if(typeof module!=='undefined'&&module.exports)module.exports=api;else scope.AtlasUI=api;
})(typeof window!=='undefined'?window:globalThis);
