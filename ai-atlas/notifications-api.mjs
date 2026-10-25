import { database } from './content-db.mjs';
const uuid=/^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const json=(value,status=200)=>Response.json(value,{status,headers:{'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'}});
const fields=['id','topic_id','topic_title','tab','status','summary'];
const publicNote=note=>Object.fromEntries(fields.map(key=>[key,note[key]??'']));
// Count and page share a snapshot. The cursor can still refer to a resolved note.
export const notificationsQuery=`WITH unresolved AS MATERIALIZED (
 SELECT n.id,n.topic_id,c.payload->'node'->>'name' AS topic_title,n.tab,n.status,n.created_at,
 LEFT(regexp_replace(CASE WHEN btrim(n.instruction)<>'' THEN n.instruction
   WHEN n.action='clarify' THEN 'Нужно пояснить: '||n.quote
   WHEN n.action='rewrite' THEN 'Нужно переписать понятнее: '||n.quote ELSE n.quote END,'[[:space:]]+',' ','g'),240) AS summary
 FROM atlas_editorial_notes n JOIN atlas_content c ON c.id=n.topic_id WHERE n.status IN ('pending','accepted')
), page AS (
 SELECT * FROM unresolved WHERE ($1='' OR (created_at,id)<(SELECT created_at,id FROM atlas_editorial_notes WHERE id=NULLIF($1,'')::uuid))
 AND ($2='' OR id=NULLIF($2,'')::uuid) ORDER BY created_at DESC,id DESC LIMIT 21
)
SELECT (SELECT count(*)::integer FROM unresolved) AS total,
 COALESCE(jsonb_agg(to_jsonb(page) ORDER BY created_at DESC,id DESC),'[]'::jsonb) AS notes FROM page`;
export async function notificationsResponse(request,connect=database){
 if(request.method!=='GET')return json({error:'Method not allowed'},405);
 const params=new URL(request.url).searchParams,before=params.get('before'),note=params.get('note');
 if((before!==null&&!uuid.test(before))||(note!==null&&!uuid.test(note))||(before&&note))return json({error:'Неверный фильтр замечаний.'},400);
 try{
  const sql=await connect(),[result]=await sql.query(notificationsQuery,[before||'',note||'']);
  const rows=result?.notes||[];
  if(note)return rows.length?json({note:publicNote(rows[0])}):json({error:'Замечание уже обработано или недоступно.'},404);
  const notes=rows.slice(0,20).map(publicNote);
  return json({total:Number(result?.total||0),notes,next:rows.length>20?notes.at(-1).id:null});
 }catch{return json({error:'Не удалось загрузить замечания. Попробуй ещё раз.'},503);}
}
