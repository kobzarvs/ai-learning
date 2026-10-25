const {test}=require('node:test'),assert=require('node:assert/strict');
const origin='https://ai-atlas-plum.vercel.app';
const owner={id:'1615093',login:'kobzarvs',name:'Valeriy'},reader={id:'123',login:'reader',name:'Reader'};
const noteId='22222222-2222-4222-8222-222222222222',requestId='11111111-1111-4111-8111-111111111111';
const payload={noteId,requestId,version:2,comment:'Добавь график'};
const run={id:'33333333-3333-4333-8333-333333333333',note_id:noteId,status:'queued',phase:'Ожидает исполнителя',summary:'',error:''};
const endpoint=()=>import('../agent-api.mjs');
function request(method='POST',input=payload,source=origin,token=true){return new Request(origin+'/api/agent-runs',{method,headers:{Origin:source,'Content-Type':'application/json',...(token?{Cookie:'__Host-atlas_session='+'a'.repeat(43)}:{})},...(method==='GET'?{}:{body:JSON.stringify(input)})});}
function db(user=owner,respond=()=>[]){const calls=[];return{calls,query:async(statement,params=[])=>{calls.push({statement,params});if(statement.includes('JOIN atlas_users u ON u.id=s.user_id'))return user?[user]:[];return respond(statement,params);}};}

test('dispatch rejects anonymous users, readers and cross-origin requests before enqueue',async()=>{
 const {agentRunsResponse}=await endpoint();
 for(const [user,token,source,status]of[[null,false,origin,401],[reader,true,origin,403],[owner,true,'https://elsewhere.test',403]]){
  const sql=db(user);assert.equal((await agentRunsResponse(request('POST',payload,source,token),async()=>sql)).status,status);
  assert.ok(!sql.calls.some(c=>c.statement.includes('atlas_enqueue_run')));
 }
});
test('dispatch uses authenticated actor and snapshots the moderator comment',async()=>{
 const {agentRunsResponse}=await endpoint();
 const sql=db(owner,(q,p)=>q.includes('atlas_enqueue_run')?[{...run,snapshot:{secret:'not public'},lease_token:'private'}]:[]);
 const result=await agentRunsResponse(request('POST',{...payload,actorId:reader.id}),async()=>sql);
 assert.equal(result.status,202);const data=await result.json();assert.equal(data.run.status,'queued');
 assert.ok(!JSON.stringify(data).includes('private'));assert.ok(!JSON.stringify(data).includes('secret'));
 const call=sql.calls.find(c=>c.statement.includes('atlas_enqueue_run'));
 assert.deepEqual(call.params,[noteId,2,requestId,owner.id,'Добавь график',10]);
});
test('invalid dispatch inputs cannot create jobs',async()=>{
 const {agentRunsResponse}=await endpoint();
 for(const bad of [null,[],{...payload,noteId:'oops'},{...payload,requestId:''},{...payload,version:0},{...payload,version:1.5},{...payload,comment:'x'.repeat(3001)}]){
  const sql=db();assert.equal((await agentRunsResponse(request('POST',bad),async()=>sql)).status,400);
  assert.ok(!sql.calls.some(c=>c.statement.includes('atlas_enqueue_run')));
 }
});
test('idempotent dispatch returns the same existing run and revision conflicts are actionable',async()=>{
 const {agentRunsResponse}=await endpoint();const sql=db(owner,q=>q.includes('atlas_enqueue_run')?[run]:[]);
 for(let i=0;i<2;i++)assert.equal((await(await agentRunsResponse(request(),async()=>sql)).json()).run.id,run.id);
 for(const [error,status]of[['ATLAS_VERSION_CONFLICT',409],['ATLAS_NOTE_NOT_FOUND',404],['ATLAS_NOTE_CLOSED',409],['postgres://private-password',503]]){
  const broken=db(owner,q=>{if(q.includes('atlas_enqueue_run'))throw Error(error);return[];});
  const response=await agentRunsResponse(request(),async()=>broken);assert.equal(response.status,status);assert.ok(!(await response.text()).includes('private-password'));
 }
});
test('job reads scope results to the authenticated note author and expose worker liveness',async()=>{
 const {agentRunsResponse}=await endpoint();const sql=db(reader,q=>q.includes('atlas_agent_workers')?[{online:true,last_seen:'2026-10-25T00:00:00Z'}]:q.includes('atlas_agent_runs')?[run]:[]);
 const response=await agentRunsResponse(request('GET'),async()=>sql);assert.equal(response.status,200);
 const data=await response.json();assert.equal(data.worker.online,true);assert.equal(data.runs.length,1);
 const query=sql.calls.find(c=>c.statement.includes('atlas_agent_runs'));
 assert.ok(query.params.includes(reader.id));assert.ok(query.params.includes(false));
 assert.equal(response.headers.get('cache-control'),'no-store');
});

test('rework requires refinement and uses the authenticated owner with a separate queue operation',async()=>{
 const {agentRunsResponse}=await endpoint();
 for(const comment of ['', '   '])assert.equal((await agentRunsResponse(request('POST',{...payload,kind:'rework',comment}),async()=>db())).status,400);
 const sql=db(owner,q=>q.includes('atlas_rework_run')?[run]:[]);
 const response=await agentRunsResponse(request('POST',{...payload,kind:'rework',actorId:reader.id}),async()=>sql);
 assert.equal(response.status,202);assert.deepEqual(sql.calls.find(c=>c.statement.includes('atlas_rework_run')).params,[noteId,2,requestId,owner.id,payload.comment,10]);
 assert.equal((await agentRunsResponse(request('POST',{...payload,kind:'other'}),async()=>db())).status,400);
});
test('history remains scoped to the note author and omits private snapshots',async()=>{
 const {agentRunsResponse}=await endpoint();const sql=db(reader,q=>q.includes('atlas_agent_runs')?[{...run,instruction:'Уточнение',snapshot:{secret:true}}]:[]);
 const req=new Request(origin+'/api/agent-runs?history='+noteId,{headers:{Cookie:'__Host-atlas_session='+'a'.repeat(43)}});
 const response=await agentRunsResponse(req,async()=>sql);const data=await response.json();
 assert.equal(response.status,200);assert.equal(data.runs[0].instruction,'Уточнение');assert.ok(!JSON.stringify(data).includes('secret'));
 const query=sql.calls.find(c=>c.statement.includes('atlas_agent_runs'));assert.deepEqual(query.params,[false,reader.id,noteId]);
});
