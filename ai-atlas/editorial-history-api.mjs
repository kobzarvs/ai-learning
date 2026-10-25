import { database } from './content-db.mjs';
import { currentUser } from './editorial-db.mjs';

// Every publication has a global revision; its first revision is the seed.
// A lateral join also keeps one publication per row if a run is retried.
export const historyQuery=`SELECT h.id AS topic_id,
 COALESCE(NULLIF(h.payload->'node'->>'name',''),NULLIF(c.payload->'node'->>'name',''),h.id) AS topic_title,
 h.revision::text AS revision,c.revision::text AS current_revision,h.updated_at AS created_at,
 CASE WHEN e.id IS NOT NULL THEN 'manual' WHEN r.id IS NOT NULL THEN 'agent' ELSE 'publication' END AS kind,
 COALESCE(NULLIF(r.summary,''),h.note,'') AS summary,
 COALESCE(NULLIF(r.instruction,''),r.note_instruction,'') AS instruction,
 COALESCE(r.tab,'article') AS tab,r.note_id::text AS note_id
 FROM atlas_history h JOIN atlas_content c ON c.id=h.id
 LEFT JOIN LATERAL (
  SELECT a.id,a.summary,a.snapshot->>'moderatorComment' AS instruction,
   n.instruction AS note_instruction,n.quote,n.tab,n.id AS note_id
  FROM atlas_agent_runs a JOIN atlas_editorial_notes n ON n.id=a.note_id
  WHERE a.result_revision=h.revision AND n.topic_id=h.id AND a.status='completed'
  ORDER BY a.finished_at DESC NULLS LAST,a.created_at DESC,a.id DESC LIMIT 1
 ) r ON true
 LEFT JOIN atlas_content_edits e ON e.topic_id=h.id AND e.revision=h.revision
 WHERE h.id<>'_catalog'
 AND EXISTS(SELECT 1 FROM atlas_history older WHERE older.id=h.id AND older.revision<h.revision)
 AND ($1::bigint IS NULL OR h.revision<$1::bigint)
 AND ($2::text='' OR position(lower($2::text) IN lower(concat_ws(E'\\n',
  h.payload->'node'->>'name',c.payload->'node'->>'name',h.note,
  r.summary,r.instruction,r.note_instruction,r.quote)))>0)
 ORDER BY h.revision DESC LIMIT 21`;

const headers={'Cache-Control':'no-store','X-Content-Type-Options':'nosniff','Referrer-Policy':'no-referrer'};
const json=(value,status=200,extra={})=>Response.json(value,{status,headers:{...headers,...extra}});
const fields=['topic_id','topic_title','revision','current_revision','created_at','kind','summary','instruction','tab','note_id'];

export async function historyResponse(request,connect=database){
 if(request.method!=='GET')return json({error:'Method not allowed'},405,{Allow:'GET'});
 try{
  const sql=await connect(),user=await currentUser(request,sql);
  if(!user)return json({error:'Войди, чтобы просмотреть историю изменений.'},401);
  if(!user.isModerator)return json({error:'История изменений доступна только владельцу.'},403);
  const params=new URL(request.url).searchParams,q=(params.get('q')||'').trim(),before=params.get('before');
  if([...params.keys()].some(key=>!['q','before'].includes(key)||params.getAll(key).length!==1)||
   q.length>200||(before!==null&&!/^\d{1,18}$/.test(before)))return json({error:'Некорректные параметры истории.'},400);
  const rows=await sql.query(historyQuery,[before,q]);
  const entries=rows.slice(0,20).map(row=>Object.fromEntries(fields.map(key=>[key,row[key]??null])));
  return json({entries,next:rows.length>20?entries.at(-1).revision:null});
 }catch{
  return json({error:'История изменений временно недоступна. Попробуй ещё раз.'},503);
 }
}
