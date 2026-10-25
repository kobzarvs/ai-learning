import { createHash, randomBytes, randomUUID, timingSafeEqual } from 'node:crypto';
import { database } from './content-db.mjs';
import { enqueueRun,publicRun,workerStatus } from './agent-db.mjs';

export const editorialSchema = [
 `CREATE TABLE IF NOT EXISTS atlas_users (
   id text PRIMARY KEY, login text NOT NULL, name text NOT NULL, updated_at timestamptz NOT NULL DEFAULT now()
 )`,
 `CREATE TABLE IF NOT EXISTS atlas_sessions (
   token_hash text PRIMARY KEY, user_id text NOT NULL REFERENCES atlas_users(id),
   expires_at timestamptz NOT NULL
 )`,
 `CREATE TABLE IF NOT EXISTS atlas_oauth_states (
   token_hash text PRIMARY KEY, verifier text NOT NULL, return_to text NOT NULL,
   expires_at timestamptz NOT NULL DEFAULT now() + interval '10 minutes'
 )`,
 `CREATE TABLE IF NOT EXISTS atlas_editorial_notes (
   id uuid PRIMARY KEY, user_id text NOT NULL REFERENCES atlas_users(id),
   request_id uuid NOT NULL, topic_id text NOT NULL REFERENCES atlas_content(id),
   tab text NOT NULL, anchor jsonb NOT NULL, quote text NOT NULL,
   content_revision bigint NOT NULL, action text NOT NULL CHECK (action IN ('clarify','rewrite','custom')),
   instruction text NOT NULL DEFAULT '', execution_mode text NOT NULL DEFAULT 'manual'
     CHECK (execution_mode IN ('manual','owner_request')), status text NOT NULL DEFAULT 'pending'
     CHECK (status IN ('pending','accepted','rejected','done')),
   moderator_note text NOT NULL DEFAULT '', moderated_by text REFERENCES atlas_users(id),
   version integer NOT NULL DEFAULT 1, created_at timestamptz NOT NULL DEFAULT now(),
   updated_at timestamptz NOT NULL DEFAULT now(), UNIQUE(user_id, request_id)
 )`,
 `ALTER TABLE atlas_editorial_notes ADD COLUMN IF NOT EXISTS execution_mode text NOT NULL DEFAULT 'manual'
   CHECK (execution_mode IN ('manual','owner_request'))`,
 `CREATE INDEX IF NOT EXISTS atlas_editorial_queue ON atlas_editorial_notes(status,created_at,id)`,
 `CREATE INDEX IF NOT EXISTS atlas_editorial_author ON atlas_editorial_notes(user_id,created_at,id)`,
 `CREATE TABLE IF NOT EXISTS atlas_editorial_events (
   id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
   note_id uuid NOT NULL REFERENCES atlas_editorial_notes(id), actor_id text NOT NULL REFERENCES atlas_users(id),
   from_status text NOT NULL, to_status text NOT NULL, comment text NOT NULL,
   created_at timestamptz NOT NULL DEFAULT now()
 )`,
 `CREATE TABLE IF NOT EXISTS atlas_editorial_limits (
   user_id text NOT NULL REFERENCES atlas_users(id), hour timestamptz NOT NULL,
   requests integer NOT NULL, PRIMARY KEY(user_id,hour)
 )`
];

const digest=value=>createHash('sha256').update(value).digest('hex');
const secret=()=>randomBytes(32).toString('base64url');
const uuid=/^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const statuses=['pending','accepted','rejected','done'];
const tabs=['topic','article','methods','python','rust','mlx','exercise'];
const responseHeaders={'Cache-Control':'no-store','X-Content-Type-Options':'nosniff','Referrer-Policy':'no-referrer'};
const json=(value,status=200,headers={})=>Response.json(value,{status,headers:{...responseHeaders,...headers}});
const cookie=(name,value,maxAge)=>`${name}=${value}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=${maxAge}`;
const cookieValue=(request,name)=>(request.headers.get('cookie')||'').split(';').map(v=>v.trim()).find(v=>v.startsWith(name+'='))?.slice(name.length+1)||'';
const validToken=value=>/^[A-Za-z0-9_-]{43}$/.test(value);
const moderator=id=>String(id)===(process.env.ATLAS_MODERATOR_GITHUB_ID||'1615093');
const origin=()=>process.env.ATLAS_ORIGIN||'https://ai-atlas-plum.vercel.app';
export const sameOrigin=request=>request.headers.get('origin')===origin();
const redirect=(url,headers={})=>new Response(null,{status:303,headers:{...responseHeaders,Location:url,...headers}});

export async function currentUser(request,sql){
 const token=cookieValue(request,'__Host-atlas_session');
 if(!validToken(token))return null;
 const [user]=await sql.query(`SELECT u.id,u.login,u.name FROM atlas_sessions s JOIN atlas_users u ON u.id=s.user_id
   WHERE s.token_hash=$1 AND s.expires_at>now()`,[digest(token)]);
 return user?{...user,isModerator:moderator(user.id)}:null;
}

export async function authResponse(request,connect=database,fetcher=fetch){
 const url=new URL(request.url),action=url.searchParams.get('action');
 const configured=Boolean(process.env.GITHUB_CLIENT_ID&&process.env.GITHUB_CLIENT_SECRET);
 if(!['GET','POST'].includes(request.method))return json({error:'Method not allowed'},405);
 try{
  if(request.method==='POST'){
   if(!sameOrigin(request))return json({error:'Недопустимый источник запроса'},403);
   const sql=await connect(),token=cookieValue(request,'__Host-atlas_session');
   if(validToken(token))await sql.query('DELETE FROM atlas_sessions WHERE token_hash=$1',[digest(token)]);
   return json({user:null},200,{'Set-Cookie':cookie('__Host-atlas_session','',0)});
  }
  if(action==='login'){
   if(!configured)return json({error:'Вход ещё настраивается. Замечания пока отправлять нельзя.'},503);
   const back=url.searchParams.get('returnTo')||'/';
   const returnTo=/^\/(?:\?tab=(?:article|methods|python|rust|mlx|exercise))?(?:#[a-zA-Z0-9_-]+)?$/.test(back)?back:'/';
   const state=secret(),verifier=secret(),sql=await connect();
   await sql.transaction([
    sql.query('DELETE FROM atlas_oauth_states WHERE expires_at<now()'),
    sql.query('DELETE FROM atlas_sessions WHERE expires_at<now()'),
    sql.query('INSERT INTO atlas_oauth_states(token_hash,verifier,return_to) VALUES($1,$2,$3)',[digest(state),verifier,returnTo])
   ]);
   const authorize=new URL('https://github.com/login/oauth/authorize');
   authorize.search=new URLSearchParams({client_id:process.env.GITHUB_CLIENT_ID,redirect_uri:origin()+'/api/auth',state,
    scope:'',code_challenge:createHash('sha256').update(verifier).digest('base64url'),code_challenge_method:'S256'}).toString();
   return redirect(authorize.href,{'Set-Cookie':cookie('__Host-atlas_oauth',state,600)});
  }
  if(url.searchParams.has('code')||url.searchParams.has('error')){
   const state=url.searchParams.get('state')||'',saved=cookieValue(request,'__Host-atlas_oauth');
   const clear={'Set-Cookie':cookie('__Host-atlas_oauth','',0)};
   if(!configured||!validToken(state)||!validToken(saved)||!timingSafeEqual(Buffer.from(state),Buffer.from(saved)))return json({error:'Сеанс входа истёк. Начни вход заново.'},400,clear);
   const sql=await connect();
   const [flow]=await sql.query('DELETE FROM atlas_oauth_states WHERE token_hash=$1 AND expires_at>now() RETURNING verifier,return_to',[digest(state)]);
   if(!flow)return json({error:'Этот запрос входа уже использован или истёк.'},400,clear);
   if(url.searchParams.has('error'))return redirect(flow.return_to,clear);
   const tokenResponse=await fetcher('https://github.com/login/oauth/access_token',{method:'POST',headers:{Accept:'application/json','Content-Type':'application/json'},signal:AbortSignal.timeout(10000),body:JSON.stringify({
    client_id:process.env.GITHUB_CLIENT_ID,client_secret:process.env.GITHUB_CLIENT_SECRET,code:url.searchParams.get('code'),redirect_uri:origin()+'/api/auth',code_verifier:flow.verifier
   })});
   const token=await tokenResponse.json();
   if(!tokenResponse.ok||!token.access_token)throw Error('OAuth exchange failed');
   const profileResponse=await fetcher('https://api.github.com/user',{headers:{Accept:'application/vnd.github+json',Authorization:'Bearer '+token.access_token,'User-Agent':'AI-Atlas'},signal:AbortSignal.timeout(10000)});
   const profile=await profileResponse.json();
   if(!profileResponse.ok||!Number.isSafeInteger(profile.id)||profile.id<=0||typeof profile.login!=='string')throw Error('Identity unavailable');
   const session=secret(),id=String(profile.id),previous=cookieValue(request,'__Host-atlas_session');
   await sql.transaction([
    sql.query(`INSERT INTO atlas_users(id,login,name) VALUES($1,$2,$3) ON CONFLICT(id)
      DO UPDATE SET login=EXCLUDED.login,name=EXCLUDED.name,updated_at=now()`,[id,profile.login,profile.name||profile.login]),
    sql.query('DELETE FROM atlas_sessions WHERE token_hash=$1',[digest(previous)]),
    sql.query(`INSERT INTO atlas_sessions(token_hash,user_id,expires_at) VALUES($1,$2,now()+interval '7 days')`,[digest(session),id])
   ]);
   // The GitHub token is used only to identify the user; it is never stored or sent to the browser.
   const headers=new Headers({...responseHeaders,Location:flow.return_to});
   headers.append('Set-Cookie',cookie('__Host-atlas_oauth','',0));
   headers.append('Set-Cookie',cookie('__Host-atlas_session',session,604800));
   return new Response(null,{status:303,headers});
  }
  if(!configured)return json({user:null,configured:false});
  return json({user:await currentUser(request,await connect()),configured:true});
 }catch{return json({error:'Не удалось выполнить вход. Попробуй ещё раз.'},503);}
}

export async function readJSON(request){
 if(!request.headers.get('content-type')?.startsWith('application/json'))throw Error('format');
 const reader=request.body?.getReader();if(!reader)throw Error('body');
 const chunks=[];let size=0;
 while(true){const {value,done}=await reader.read();if(done)break;size+=value.length;if(size>24000){await reader.cancel();throw Error('size');}chunks.push(value);}
 return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}
const clean=(value,max)=>typeof value==='string'&&value.trim().length<=max?value.trim():null;

export const notesQuery=`SELECT n.*,u.login,u.name,c.payload->'node'->>'name' AS topic_title,
 to_jsonb(latest)-'snapshot'-'lease_token' AS run
 FROM atlas_editorial_notes n JOIN atlas_users u ON u.id=n.user_id JOIN atlas_content c ON c.id=n.topic_id
 LEFT JOIN LATERAL (SELECT * FROM atlas_agent_runs r WHERE r.note_id=n.id ORDER BY r.created_at DESC,r.id DESC LIMIT 1) latest ON true
 WHERE ($1::boolean OR n.user_id=$2)
 AND ($3='all' OR n.status=$3 OR ($3='inbox' AND (n.status='pending' OR
   (n.status NOT IN ('done','rejected') AND (latest.status='failed' OR (latest.status='running' AND latest.lease_until<now()))))))
 AND ($4='' OR n.topic_id=$4)
 AND ($5='' OR (n.created_at,n.id)<(SELECT created_at,id FROM atlas_editorial_notes WHERE id=NULLIF($5,'')::uuid AND ($1::boolean OR user_id=$2)))
 AND ($6='' OR n.id=NULLIF($6,'')::uuid)
 AND ($7='' OR position(lower($7) IN lower(concat_ws(' ',c.payload->'node'->>'name',n.quote,n.instruction,n.moderator_note)))>0
  OR EXISTS(SELECT 1 FROM atlas_agent_runs searched WHERE searched.note_id=n.id
   AND position(lower($7) IN lower(concat_ws(' ',searched.summary,searched.error,searched.snapshot->>'moderatorComment')))>0))
 ORDER BY n.created_at DESC,n.id DESC LIMIT ($8::int+1)`;

export async function notesResponse(request,connect=database){
 if(!['GET','POST','PATCH'].includes(request.method))return json({error:'Method not allowed'},405);
 if(request.method!=='GET'&&!sameOrigin(request))return json({error:'Недопустимый источник запроса'},403);
 try{
  const sql=await connect(),user=await currentUser(request,sql);
  if(!user)return json({error:'Войди, чтобы работать с замечаниями.'},401);
  const url=new URL(request.url);
  if(request.method==='GET'){
   const all=url.searchParams.get('scope')==='all';
   if(all&&!user.isModerator)return json({error:'Доступно только модератору.'},403);
   const status=url.searchParams.get('status')||'all',topic=url.searchParams.get('topic')||'',before=url.searchParams.get('before')||'',id=url.searchParams.get('id')||'';
   const q=(url.searchParams.get('q')||'').trim(),limit=url.searchParams.get('limit')||'50';
   if(!['all','inbox',...statuses].includes(status)||(!/^[a-zA-Z0-9_-]{0,80}$/.test(topic))||(before&&!uuid.test(before))||(id&&!uuid.test(id))||q.length>200||!/^\d{1,2}$/.test(limit)||Number(limit)<1||Number(limit)>50)return json({error:'Неверный фильтр'},400);
   const worker=await workerStatus(sql),pageSize=Number(limit);
   const rows=await sql.query(notesQuery,[all,user.id,status,topic,before,id,q,pageSize]);
   const hasMore=rows.length>pageSize,notes=rows.slice(0,pageSize).map(n=>({...n,run:publicRun(n.run)}));
   return json({notes,next:hasMore?notes.at(-1).id:null,worker});
  }
  let input;try{input=await readJSON(request);if(!input||typeof input!=='object'||Array.isArray(input))throw Error('object');}catch{return json({error:'Некорректное или слишком длинное замечание.'},400);}
  if(request.method==='POST'){
   const quote=clean(input.quote,6000),instruction=clean(input.instruction||'',3000),topic=clean(input.topicId,80);
   const anchor=input.anchor;
   if(!uuid.test(input.requestId)||!topic||!/^[a-zA-Z0-9_-]+$/.test(topic)||!tabs.includes(input.tab)||!quote||instruction===null||
     !['clarify','rewrite','custom'].includes(input.action)||(input.action==='custom'&&!instruction)||(input.tab==='topic'&&input.action!=='custom')||
     !anchor||typeof anchor!=='object'||Array.isArray(anchor)||!Object.values(anchor).every(v=>typeof v==='string'&&v.length<=300)||Object.keys(anchor).some(k=>!['methodId','panel','section','block'].includes(k))||
     !/^\d{1,18}$/.test(String(input.contentRevision||'0')))return json({error:'Проверь текст и инструкцию.'},400);
   const [existing]=await sql.query('SELECT * FROM atlas_editorial_notes WHERE user_id=$1 AND request_id=$2',[user.id,input.requestId]);
   if(existing)return submissionResponse(sql,existing,user,input.requestId,200);
   const [topicRow]=await sql.query('SELECT id FROM atlas_content WHERE id=$1',[topic]);
   if(!topicRow)return json({error:'Тема не найдена.'},404);
   const [quota]=await sql.query(`INSERT INTO atlas_editorial_limits(user_id,hour,requests) VALUES($1,date_trunc('hour',now()),1)
     ON CONFLICT(user_id,hour) DO UPDATE SET requests=atlas_editorial_limits.requests+1
     WHERE atlas_editorial_limits.requests<60 RETURNING requests`,[user.id]);
   if(!quota)return json({error:'За час отправлено слишком много замечаний. Попробуй позже.'},429);
   // Only the authenticated owner's own submission can become an agent command.
   // Moderating somebody else's note never changes its execution mode.
   const executionMode=user.isModerator?'owner_request':'manual';
   const [note]=await sql.query(`INSERT INTO atlas_editorial_notes(id,user_id,request_id,topic_id,tab,anchor,quote,content_revision,action,instruction,execution_mode)
     VALUES($1,$2,$3,$4,$5,$6::jsonb,$7,$8::bigint,$9,$10,$11)
     ON CONFLICT(user_id,request_id) DO UPDATE SET request_id=EXCLUDED.request_id RETURNING *`,
     [randomUUID(),user.id,input.requestId,topic,input.tab,JSON.stringify(anchor),quote,String(input.contentRevision||'0'),input.action,instruction,executionMode]);
   return submissionResponse(sql,note,user,input.requestId,201);
  }
  if(!user.isModerator)return json({error:'Доступно только модератору.'},403);
  const comment=clean(input.comment||'',3000);
  if(!uuid.test(input.id)||!statuses.includes(input.status)||!Number.isSafeInteger(input.version)||input.version<1||comment===null)return json({error:'Некорректное решение модератора.'},400);
  const [result]=await sql.query(`WITH previous AS MATERIALIZED (
     SELECT id,status FROM atlas_editorial_notes WHERE id=$1::uuid AND version=$2
     AND NOT EXISTS(SELECT 1 FROM atlas_agent_runs WHERE note_id=$1::uuid AND (status='queued' OR (status='running' AND lease_until>now()))) FOR UPDATE
   ), changed AS (
     UPDATE atlas_editorial_notes n SET status=$3,moderator_note=$4,moderated_by=$5,version=version+1,updated_at=now()
     FROM previous p WHERE n.id=p.id AND n.version=$2 RETURNING n.*,p.status AS previous_status
   ), history AS (
     INSERT INTO atlas_editorial_events(note_id,actor_id,from_status,to_status,comment)
       SELECT id,$5,previous_status,status,$4 FROM changed
   ) SELECT * FROM changed`,[input.id,input.version,input.status,comment,user.id]);
  return result?json({note:result}):json({error:'Замечание изменилось или уже выполняется агентом. Обнови список.'},409);
 }catch{return json({error:'Не удалось сохранить или загрузить замечания. Попробуй ещё раз.'},503);}
}

async function submissionResponse(sql,note,user,requestId,status){
 if(!user.isModerator||note.execution_mode!=='owner_request')return json({note},status);
 try{
  const run=await enqueueRun(sql,{noteId:note.id,version:note.version||1,requestId,actorId:user.id,comment:note.moderator_note||'',priority:0});
  return json({note,run,worker:await workerStatus(sql)},status);
 }catch{return json({note,dispatchError:'Команда сохранена, но запуск не подтверждён. Открой модерацию и нажми «Выполнить сейчас».'},status);}
}
