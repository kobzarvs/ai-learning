const {test}=require('node:test'),assert=require('node:assert/strict');
const {createHash}=require('node:crypto');
const {textBlock,snapshot,topicSnapshot,accountMarkup}=require('../editorial.js');
const origin='https://ai-atlas-plum.vercel.app';
const token='a'.repeat(43),requestId='11111111-1111-4111-8111-111111111111';
const noteId='22222222-2222-4222-8222-222222222222';
const owner={id:'1615093',login:'kobzarvs',name:'Valeriy'};
const visitor={id:'12345',login:'reader',name:'Reader'};
const valid={requestId,topicId:'causal',tab:'methods',quote:'Что может исказить результат',
 contentRevision:'138',action:'rewrite',instruction:'',anchor:{methodId:'rct',panel:'methods',section:'Проверка',block:'4'}};
function request(method='GET',payload,query='',authenticated=true,source=origin){
 return new Request(origin+'/api/editorial-notes'+query,{method,headers:{Origin:source,
  ...(authenticated?{Cookie:'__Host-atlas_session='+token}:{}),'Content-Type':'application/json'},
  ...(payload!==undefined?{body:JSON.stringify(payload)}:{})});
}
function database(user=visitor,respond=()=>[]){
 const calls=[];
 const sql={calls,query(statement,params=[]){calls.push({statement,params});return Promise.resolve(
  statement.includes('JOIN atlas_users u ON u.id=s.user_id')?(user?[user]:[]):respond(statement,params));},
  transaction:queries=>Promise.all(queries)};
 return sql;
}
const endpoints=import('../editorial-db.mjs');

test('account identity safely renders GitHub names and limits avatar requests to numeric user IDs',()=>{
 const html=accountMarkup({id:'123',login:'reader',name:'<img src=x onerror=alert(1)>',isModerator:false},'account-test');
 assert.match(html,/&lt;img/);assert.doesNotMatch(html,/<img src=x/);
 assert.match(html,/https:\/\/avatars\.githubusercontent\.com\/u\/123\?s=80/);
 assert.match(html,/Мои замечания/);assert.doesNotMatch(html,/Модерация/);
 assert.doesNotMatch(accountMarkup({...visitor,id:'https://other.test/avatar'},'account-test'),/<img/);
 assert.match(accountMarkup(null,'account-test'),/>Войти<\/button>/);
});

test('anonymous readers cannot read or submit editorial notes; cross-origin writes stop before SQL',async()=>{
 const {notesResponse}=await endpoints;const sql=database();
 for(const method of ['GET','POST','PATCH'])assert.equal((await notesResponse(request(method,method==='GET'?undefined:valid,'',false),async()=>sql)).status,401);
 assert.equal(sql.calls.length,0);
 assert.equal((await notesResponse(request('POST',valid,'',true,'https://evil.test'),()=>{throw Error('must not connect');})).status,403);
});

test('ordinary users cannot list all notes or moderate, even when their name imitates the owner',async()=>{
 const {notesResponse,currentUser}=await endpoints;const sql=database({...visitor,login:owner.login,name:owner.name,email:'kobzarvs@gmail.com',isModerator:true});
 const user=await currentUser(request(),sql);assert.equal(user.isModerator,false);
 assert.equal((await notesResponse(request('GET',undefined,'?scope=all'),async()=>sql)).status,403);
 assert.equal((await notesResponse(request('PATCH',{id:noteId,status:'accepted',version:1}),async()=>sql)).status,403);
 assert.ok(!sql.calls.some(c=>c.statement.includes('UPDATE atlas_editorial_notes')));
});

test('own-note listing binds the session identity and restricts pagination cursors to that identity',async()=>{
 const {notesResponse}=await endpoints;const sql=database(visitor,()=>Array.from({length:51},(_,i)=>({id:String(i)})));
 const result=await notesResponse(request('GET',undefined,'?status=pending&before='+noteId),async()=>sql);
 assert.equal(result.status,200);const body=await result.json();assert.equal(body.notes.length,50);assert.equal(body.next,'49');
 const query=sql.calls.at(-1);assert.deepEqual(query.params,[false,visitor.id,'pending','',noteId]);
 assert.match(query.statement,/WHERE id=NULLIF\(\$5,''\)::uuid AND \(\$1::boolean OR user_id=\$2\)/);
});

test('only the verified owner can create an agent command; forged client permissions are ignored',async()=>{
 const {notesResponse}=await endpoints;
 for(const user of [visitor,owner]){
  const sql=database(user,(statement,params)=>{
   if(statement.startsWith('SELECT id FROM atlas_content'))return[{id:'causal'}];
   if(statement.includes('INSERT INTO atlas_editorial_limits'))return[{requests:1}];
   if(statement.includes('INSERT INTO atlas_editorial_notes'))return[{id:params[0],user_id:params[1],execution_mode:params[10],status:'pending'}];
   return[];
  });
  const result=await notesResponse(request('POST',{...valid,user_id:owner.id,isModerator:true,execution_mode:'owner_request',status:'done'}),async()=>sql);
  assert.equal(result.status,201);const {note}=await result.json();
  assert.equal(note.user_id,user.id);assert.equal(note.status,'pending');
  assert.equal(note.execution_mode,user===owner?'owner_request':'manual');
  assert.equal(result.headers.get('cache-control'),'no-store');
 }
});

test('retrying a request returns the existing note without consuming quota or duplicating it',async()=>{
 const {notesResponse}=await endpoints;const note={id:noteId,user_id:visitor.id};
 const sql=database(visitor,statement=>statement.startsWith('SELECT * FROM atlas_editorial_notes')?[note]:[]);
 const result=await notesResponse(request('POST',valid),async()=>sql);assert.equal(result.status,200);
 assert.deepEqual((await result.json()).note,note);assert.equal(sql.calls.length,2);
});

test('whole-topic feedback keeps its topic and revision and uses the same authenticated moderation flow',async()=>{
 const {notesResponse}=await endpoints;
 const context=topicSnapshot({topicId:'knn',title:'kNN — ближайшие соседи',tab:'python',revision:'42'});
 assert.equal(context.tab,'topic');assert.equal(context.contentRevision,'42');assert.equal(context.anchor.block,'');
 for(const user of [visitor,owner]){
  const sql=database(user,(statement,params)=>{
   if(statement.startsWith('SELECT id FROM atlas_content'))return[{id:'knn'}];
   if(statement.includes('INSERT INTO atlas_editorial_limits'))return[{requests:1}];
   if(statement.includes('INSERT INTO atlas_editorial_notes'))return[{topic_id:params[3],tab:params[4],anchor:JSON.parse(params[5]),quote:params[6],instruction:params[9],execution_mode:params[10]}];
   return[];
  });
  const payload={...context,requestId,action:'custom',instruction:'Добавить график выбора ближайших соседей.'};
  const result=await notesResponse(request('POST',payload),async()=>sql);
  assert.equal(result.status,201);const {note}=await result.json();
  assert.equal(note.topic_id,'knn');assert.equal(note.tab,'topic');assert.equal(note.quote,'kNN — ближайшие соседи');
  assert.equal(note.instruction,payload.instruction);assert.equal(note.execution_mode,user===owner?'owner_request':'manual');
  assert.equal((await notesResponse(request('POST',payload,'',false),async()=>sql)).status,401);
  assert.equal((await notesResponse(request('POST',{...payload,action:'rewrite'}),async()=>sql)).status,400);
 }
});

test('invalid, oversized and missing-topic submissions do not create a note',async()=>{
 const {notesResponse}=await endpoints;
 for(const body of [null,[],{...valid,quote:''},{...valid,quote:'x'.repeat(6001)},{...valid,action:'publish'},
  {...valid,action:'custom',instruction:''},{...valid,anchor:{sql:'DROP TABLE'}},{...valid,tab:'arbitrary'},
  {...valid,contentRevision:'1;SELECT'},{...valid,instruction:'x'.repeat(25000)}]){
  const sql=database();assert.equal((await notesResponse(request('POST',body),async()=>sql)).status,400);
  assert.ok(!sql.calls.some(c=>c.statement.includes('INSERT')));
 }
 assert.equal((await notesResponse(request('POST',valid),async()=>database())).status,404);
});

test('rate limits reject excessive notes and SQL errors never disclose credentials',async()=>{
 const {notesResponse}=await endpoints;
 const sql=database(visitor,statement=>statement.startsWith('SELECT id FROM atlas_content')?[{id:'causal'}]:[]);
 assert.equal((await notesResponse(request('POST',valid),async()=>sql)).status,429);
 const result=await notesResponse(request(),()=>{throw Error('postgres://secret-password');});
 assert.equal(result.status,503);assert.ok(!(await result.text()).includes('secret-password'));
});

test('moderation records an audited decision, detects stale versions and never requests automatic execution',async()=>{
 const {notesResponse}=await endpoints;const payload={id:noteId,version:2,status:'accepted',comment:'Объяснить подробнее',execution_mode:'owner_request'};
 const sql=database(owner,statement=>statement.startsWith('WITH previous')?[{id:noteId,status:'accepted',version:3,execution_mode:'manual'}]:[]);
 const result=await notesResponse(request('PATCH',payload),async()=>sql);assert.equal(result.status,200);
 assert.equal((await result.json()).note.execution_mode,'manual');const query=sql.calls.at(-1);
 assert.deepEqual(query.params,[noteId,2,'accepted','Объяснить подробнее',owner.id]);
 assert.match(query.statement,/INSERT INTO atlas_editorial_events/);assert.match(query.statement,/n.version=\$2/);
 assert.doesNotMatch(query.statement,/execution_mode\s*=|atlas_publish|UPDATE atlas_content/);
 assert.equal((await notesResponse(request('PATCH',payload),async()=>database(owner))).status,409);
});

test('expired sessions cannot create notes or invoke owner privileges',async()=>{
 const {notesResponse,currentUser}=await endpoints;const sql=database(null);
 assert.equal(await currentUser(request(),sql),null);assert.match(sql.calls[0].statement,/s.expires_at>now\(\)/);
 assert.equal((await notesResponse(request('POST',valid),async()=>sql)).status,401);
 assert.equal(sql.calls[0].params[0],createHash('sha256').update(token).digest('hex'));
});

test('OAuth uses a single-use cookie-bound state, PKCE and a local return path',async()=>{
 const {authResponse}=await endpoints;const saved={id:process.env.GITHUB_CLIENT_ID,secret:process.env.GITHUB_CLIENT_SECRET};
 process.env.GITHUB_CLIENT_ID='test-client';process.env.GITHUB_CLIENT_SECRET='test-secret';
 try{
  const sql=database();const result=await authResponse(new Request(origin+'/api/auth?action=login&returnTo=https://evil.test'),async()=>sql);
  assert.equal(result.status,303);const location=new URL(result.headers.get('location')),cookie=result.headers.get('set-cookie');
  assert.equal(location.origin,'https://github.com');assert.equal(location.searchParams.get('scope'),'');
  assert.equal(location.searchParams.get('code_challenge_method'),'S256');assert.match(cookie,/HttpOnly; Secure; SameSite=Lax/);
  const insert=sql.calls.find(c=>c.statement.startsWith('INSERT INTO atlas_oauth_states'));
  assert.equal(insert.params[2],'/');assert.equal(location.searchParams.get('code_challenge'),createHash('sha256').update(insert.params[1]).digest('base64url'));
  const state=location.searchParams.get('state');
  const callback=()=>new Request(origin+'/api/auth?code=test-code&state='+state,{headers:{Cookie:'__Host-atlas_oauth='+state}});
  assert.equal((await authResponse(callback(),async()=>database())).status,400);
  const invalid=new Request(origin+'/api/auth?code=test-code&state='+state,{headers:{Cookie:'__Host-atlas_oauth='+'z'.repeat(43)}});
  assert.equal((await authResponse(invalid,()=>{throw Error('must not connect');})).status,400);
  let used=false,fetches=0;
  const sql2=database(null,statement=>{if(statement.startsWith('DELETE FROM atlas_oauth_states')&&!used){used=true;return[{verifier:insert.params[1],return_to:'/#causal'}];}return[];});
  const fetcher=async(url,options)=>{fetches++;if(url.includes('access_token')){assert.equal(JSON.parse(options.body).code_verifier,insert.params[1]);return Response.json({access_token:'private-github-token'});}assert.equal(options.headers.Authorization,'Bearer private-github-token');return Response.json(ownerProfile());};
  const done=await authResponse(callback(),async()=>sql2,fetcher);assert.equal(done.status,303);assert.equal(done.headers.get('location'),'/#causal');
  assert.ok(done.headers.getSetCookie().some(c=>c.startsWith('__Host-atlas_session=')&&c.includes('HttpOnly; Secure')));
  assert.ok(!JSON.stringify([...done.headers]).includes('private-github-token'));
  assert.equal(sql2.calls.find(c=>c.statement.includes('INSERT INTO atlas_users')).params[0],owner.id);
  assert.ok(!JSON.stringify(sql2.calls).includes('private-github-token'));
  assert.equal((await authResponse(callback(),async()=>sql2,fetcher)).status,400);assert.equal(fetches,2);
 }finally{
  for(const [key,value] of [['GITHUB_CLIENT_ID',saved.id],['GITHUB_CLIENT_SECRET',saved.secret]])if(value===undefined)delete process.env[key];else process.env[key]=value;
 }
});
function ownerProfile(){return{id:Number(owner.id),login:owner.login,name:owner.name};}

test('unconfigured sign-in stays unavailable without breaking public reading',async()=>{
 const {authResponse}=await endpoints;const saved=process.env.GITHUB_CLIENT_SECRET;delete process.env.GITHUB_CLIENT_SECRET;
 try{const result=await authResponse(new Request(origin+'/api/auth'),()=>{throw Error('must not connect');});assert.deepEqual(await result.json(),{user:null,configured:false});}
 finally{if(saved!==undefined)process.env.GITHUB_CLIENT_SECRET=saved;}
});

test('text selection ignores code and controls and captures an exact topic/method/quote context',()=>{
 const block={textContent:' Первый  абзац\nс пояснением. ',matches:()=>false,closest(selector){return selector==='section'?{querySelector:()=>({textContent:'Заголовок'})}:selector==='.guide-method'?{id:'causal-method-rct'}:{dataset:{guidePanel:'methods'}};}};
 const content={contains:()=>true,querySelectorAll:()=>[{},block]};
 const target={closest:selector=>selector.includes('button,')?null:block};
 assert.equal(textBlock(target,content),block);
 assert.equal(textBlock({closest:()=>({})},content),null);
 assert.deepEqual(snapshot(block,content,{topicId:'causal',tab:'methods',revision:'138'}),{topicId:'causal',tab:'methods',contentRevision:'138',quote:'Первый абзац с пояснением.',anchor:{methodId:'rct',panel:'methods',section:'Заголовок',block:'1'}});
});
