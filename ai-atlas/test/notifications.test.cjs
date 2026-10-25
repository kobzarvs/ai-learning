const {test}=require('node:test'),assert=require('node:assert/strict');
const id='22222222-2222-4222-8222-222222222222';
const row={id,topic_id:'knn',topic_title:'Ближайшие соседи',tab:'article',status:'pending',summary:'Не хватает примера'};
function req(query='',method='GET'){return new Request('https://atlas.test/api/editorial-notifications'+query,{method});}
function db(rows=[row],total=rows.length){const calls=[];return{calls,query:async(q,p)=>{calls.push({q,p});return[{total,notes:rows}];}};}
test('public notifications return open-note count and bounded summaries without identities or private agent data',async()=>{
 const {notificationsResponse}=await import('../notifications-api.mjs');const sql=db([{...row,user_id:'private',quote:'secret quote',moderator_note:'private comment',run:{error:'private run'}}],37);
 const response=await notificationsResponse(req(),async()=>sql);assert.equal(response.status,200);
 const data=await response.json();assert.equal(data.total,37);assert.deepEqual(data.notes,[row]);assert.equal(data.next,null);
 assert.ok(!JSON.stringify(data).includes('private'));assert.ok(!JSON.stringify(data).includes('secret'));
 assert.match(sql.calls[0].q,/status IN \('pending','accepted'\)/);assert.match(sql.calls[0].q,/LEFT\(/);assert.equal(response.headers.get('cache-control'),'no-store');
});
test('notification pagination preserves total and supports cursors whose note was resolved',async()=>{
 const {notificationsResponse}=await import('../notifications-api.mjs');const rows=Array.from({length:21},(_,i)=>({...row,id:'note-'+i})),sql=db(rows,53);
 const data=await(await notificationsResponse(req('?before='+id),async()=>sql)).json();
 assert.equal(data.total,53);assert.equal(data.notes.length,20);assert.equal(data.next,'note-19');assert.deepEqual(sql.calls[0].p,[id,'']);
 assert.match(sql.calls[0].q,/FROM atlas_editorial_notes WHERE id=NULLIF\(\$1,''\)::uuid/);
});
test('single public notification lookup is bounded, read-only and handles resolved notes',async()=>{
 const {notificationsResponse}=await import('../notifications-api.mjs');const sql=db();
 const response=await notificationsResponse(req('?note='+id),async()=>sql);assert.deepEqual((await response.json()).note,row);assert.deepEqual(sql.calls[0].p,['',id]);
 assert.equal((await notificationsResponse(req('?note='+id),async()=>db([]))).status,404);
 for(const query of ['?note=bad','?before=bad','?note=','?before='])assert.equal((await notificationsResponse(req(query),()=>{throw Error('must not query');})).status,400);
 for(const method of ['POST','PATCH','DELETE'])assert.equal((await notificationsResponse(req('',method),()=>{throw Error('must not query');})).status,405);
 const failed=await notificationsResponse(req(),()=>{throw Error('password');});assert.equal(failed.status,503);assert.ok(!(await failed.text()).includes('password'));
});
test('notification menu escapes titles and summaries and shows exact total separately from the loaded page',()=>{
 const {notificationItems,notificationLabel}=require('../notifications.js');
 const html=notificationItems([{...row,topic_title:'<img src=x>',summary:'<script>bad</script>'}]);assert.match(html,/&lt;img/);assert.doesNotMatch(html,/<img|<script/);assert.match(html,/data-notification-note="22222222/);
 assert.equal(notificationLabel(53),'Нерешённые замечания: 53');assert.match(notificationItems([]),/Нет нерешённых замечаний/);
});

test('refreshing a paged dropdown includes all loaded pages and deduplicates a moving item',async()=>{
 const {fetchNotificationPages}=require('../notifications.js');const urls=[];
 const result=await fetchNotificationPages(async url=>{urls.push(url);return urls.length===1?{total:40,notes:[row],next:id}:{total:39,notes:[row,{...row,id:'second'}],next:null};},2);
 assert.deepEqual(urls,['/api/editorial-notifications','/api/editorial-notifications?before='+id]);assert.equal(result.total,39);assert.equal(result.notes.length,2);assert.equal(result.next,null);
});
