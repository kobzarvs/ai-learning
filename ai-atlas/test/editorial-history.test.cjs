const {test}=require('node:test'),assert=require('node:assert/strict');
const origin='https://ai-atlas-plum.vercel.app',owner={id:'1615093',login:'owner',name:'Owner'};
const request=(query='',{method='GET',token=true}={})=>new Request(origin+'/api/editorial-history'+query,{method,headers:token?{Cookie:'__Host-atlas_session='+'a'.repeat(43)}:{}});
function db(user=owner,rows=[]){
 const calls=[];
 return{calls,query:async(statement,params=[])=>{
  calls.push({statement,params});
  if(statement.includes('JOIN atlas_users u'))return user?[user]:[];
  if(rows instanceof Error)throw rows;
  return rows;
 }};
}
const row={topic_id:'knn',topic_title:'Ближайшие соседи',revision:'9007199254740993',current_revision:'9007199254740994',created_at:'2026-10-25T10:00:00.000Z',kind:'agent',summary:'Уточнён пример',instruction:'Добавить пояснение',tab:'methods',note_id:'11111111-1111-4111-8111-111111111111'};

test('history rejects unsupported methods and enforces owner access before reading publications',async()=>{
 const {historyResponse}=await import('../editorial-history-api.mjs');
 for(const method of ['POST','PUT','PATCH','DELETE','OPTIONS']){
  const result=await historyResponse(request('',{method}),()=>{throw Error('must not connect');});
  assert.equal(result.status,405);assert.equal(result.headers.get('Allow'),'GET');
 }
 for(const [user,token,status]of[[null,false,401],[null,true,401],[{id:'123',login:'reader'},true,403]]){
  const sql=db(user,[row]),result=await historyResponse(request('',{token}),async()=>sql);
  assert.equal(result.status,status);
  assert.ok(sql.calls.every(call=>call.statement.includes('JOIN atlas_users u')));
 }
});

test('history validates every query parameter without issuing a history query',async()=>{
 const {historyResponse}=await import('../editorial-history-api.mjs');
 for(const query of ['?q='+encodeURIComponent('я'.repeat(201)),'?before=','?before=-1','?before=1.5','?before=1e3','?before=1234567890123456789','?before=1%20','?before=1&before=2','?q=a&q=b','?limit=100','?topic=knn','?q=x&unexpected=']){
  const sql=db(),result=await historyResponse(request(query),async()=>sql);
  assert.equal(result.status,400,query);
  assert.ok(sql.calls.every(call=>call.statement.includes('JOIN atlas_users u')),query);
 }
});

test('history parameterizes literal searches and preserves bigint cursors',async()=>{
 const {historyResponse,historyQuery}=await import('../editorial-history-api.mjs');
 const sql=db(),query="  50%_ O'Reilly \\ пример  ";
 const result=await historyResponse(request('?q='+encodeURIComponent(query)+'&before=9007199254740993'),async()=>sql);
 assert.equal(result.status,200);assert.deepEqual(await result.json(),{entries:[],next:null});
 const call=sql.calls.find(c=>c.statement===historyQuery);assert.ok(call);
 assert.deepEqual(call.params,['9007199254740993',"50%_ O'Reilly \\ пример"]);
 assert.ok(!call.statement.includes("O'Reilly"));
 const empty=db();await historyResponse(request('?q=%20%20'),async()=>empty);
 assert.deepEqual(empty.calls.find(c=>c.statement===historyQuery).params,[null,'']);
 const maximum=db();assert.equal((await historyResponse(request('?q='+encodeURIComponent('я'.repeat(200))+'&before=999999999999999999'),async()=>maximum)).status,200);
});

test('history exposes only the public history entry contract and prevents caching',async()=>{
 const {historyResponse}=await import('../editorial-history-api.mjs');
 const sql=db(owner,[{...row,snapshot:{secret:'private'},lease_token:'secret-token',payload:{private:'document'},actor_id:'private-actor',error:'private-error'}]);
 const result=await historyResponse(request(),async()=>sql);
 assert.equal(result.status,200);assert.deepEqual(await result.json(),{entries:[row],next:null});
 assert.equal(result.headers.get('Cache-Control'),'no-store');assert.equal(result.headers.get('X-Content-Type-Options'),'nosniff');
});

test('history returns at most twenty entries and advances using the last displayed revision',async()=>{
 const {historyResponse}=await import('../editorial-history-api.mjs');
 const rows=Array.from({length:21},(_,index)=>({...row,revision:String(100-index)}));
 const page=await (await historyResponse(request(),async()=>db(owner,rows))).json();
 assert.equal(page.entries.length,20);assert.equal(page.entries[0].revision,'100');assert.equal(page.entries.at(-1).revision,'81');assert.equal(page.next,'81');
 const last=await(await historyResponse(request('?before=81'),async()=>db(owner,[rows[20]]))).json();
 assert.equal(last.entries[0].revision,'80');assert.equal(last.next,null);
 const fullLast=await(await historyResponse(request(),async()=>db(owner,rows.slice(0,20)))).json();
 assert.equal(fullLast.entries.length,20);assert.equal(fullLast.next,null);
});

test('history service errors never expose database details',async()=>{
 const {historyResponse}=await import('../editorial-history-api.mjs');
 for(const connect of [async()=>{throw Error('postgres://private-credentials');},async()=>db(owner,Error('SELECT secret_snapshot FROM private_table'))]){
  const result=await historyResponse(request(),connect);
  assert.equal(result.status,503);const body=await result.json();assert.equal(typeof body.error,'string');assert.ok(!/postgres|credentials|snapshot|private_table|SELECT/.test(body.error));
 }
});

test('both deployment routes expose the readonly history handler',async()=>{
 for(const path of ['../api/editorial-history.mjs','../../api/editorial-history.mjs']){
  const {default:route}=await import(path),result=await route.fetch(request('',{method:'POST'}));
  assert.equal(result.status,405);assert.equal(result.headers.get('Allow'),'GET');
 }
});
