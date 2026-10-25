const {test}=require('node:test'),assert=require('node:assert/strict');
const origin='https://ai-atlas-plum.vercel.app',owner={id:'1615093',login:'owner',name:'Owner'};
const requestId='11111111-1111-4111-8111-111111111111';
const input={requestId,topicId:'knn',expectedRevision:'8',action:'delete',targets:[{path:'node.what',part:0}]};
const request=(data=input,source=origin,token=true)=>new Request(origin+'/api/content-edit',{method:'POST',headers:{Origin:source,'Content-Type':'application/json',...(token?{Cookie:'__Host-atlas_session='+'a'.repeat(43)}:{})},body:JSON.stringify(data)});
function db(user=owner,respond=()=>[]){const calls=[];return{calls,query:async(q,p=[])=>{calls.push({q,p});if(q.includes('JOIN atlas_users u'))return user?[user]:[];return respond(q,p);}};}
test('direct editing rejects anonymous, non-owner and foreign-origin writes before any publication',async()=>{
 const {editResponse}=await import('../content-edit-api.mjs');
 for(const [user,token,source,status]of[[null,false,origin,401],[{id:'123'},true,origin,403],[owner,true,'https://evil.test',403]]){
  const sql=db(user);assert.equal((await editResponse(request(input,source,token),async()=>sql)).status,status);
  assert.ok(!sql.calls.some(c=>c.q.includes('atlas_edit_content')));
 }
});
test('direct deletion submits only transformed server content with authenticated actor and revision',async()=>{
 const {editResponse}=await import('../content-edit-api.mjs');
 const sql=db(owner,q=>q.includes('SELECT payload,revision')?[{payload:{node:{id:'knn',what:'A\n\nB'},lesson:{code:'keep'}},revision:'8'}]:q.includes('atlas_edit_content')?[{edit_id:requestId,revision:'9'}]:[]);
 const r=await editResponse(request({...input,payload:{node:{id:'hacked'}},actorId:'other'}),async()=>sql);
 assert.equal(r.status,200);assert.deepEqual(await r.json(),{editId:requestId,revision:'9'});
 const saved=sql.calls.find(c=>c.q.includes('atlas_edit_content'));
 assert.equal(saved.p[0],'knn');assert.equal(saved.p[1],'8');assert.equal(saved.p[3],owner.id);
 assert.deepEqual(JSON.parse(saved.p[2]),{node:{id:'knn',what:'B'},lesson:{code:'keep'}});
});
test('conflicts, unsupported paths and repeated requests cannot overwrite newer content',async()=>{
 const {editResponse}=await import('../content-edit-api.mjs');
 const stale=db(owner,q=>q.includes('SELECT payload,revision')?[{payload:{node:{what:'A'}},revision:'10'}]:[]);
 assert.equal((await editResponse(request(),async()=>stale)).status,409);
 const bad=db(owner,q=>q.includes('SELECT payload,revision')?[{payload:{node:{id:'knn',what:'A'}},revision:'8'}]:[]);
 assert.equal((await editResponse(request({...input,targets:[{path:'node.id'}]}),async()=>bad)).status,400);
 assert.ok(!bad.calls.some(c=>c.q.includes('atlas_edit_content')));
 const repeated=db(owner,q=>q.includes('atlas_content_edits')?[{id:requestId,revision:'9',fingerprint:require('node:crypto').createHash('sha256').update(JSON.stringify(input)).digest('hex')}]:[]);
 assert.equal((await editResponse(request(),async()=>repeated)).status,200);
 assert.ok(!repeated.calls.some(c=>c.q.includes('atlas_edit_content')));
});
test('a retry racing with the original commit returns its edit ID instead of a stale conflict',async()=>{
 const {editResponse}=await import('../content-edit-api.mjs');let reads=0;
 const fingerprint=require('node:crypto').createHash('sha256').update(JSON.stringify(input)).digest('hex');
 const sql=db(owner,q=>q.includes('atlas_content_edits')?(++reads===1?[]:[{id:requestId,revision:'9',fingerprint}]):q.includes('SELECT payload,revision')?[{payload:{node:{what:'B'}},revision:'9'}]:[]);
 const response=await editResponse(request(),async()=>sql);
 assert.equal(response.status,200);assert.equal((await response.json()).editId,requestId);
});
