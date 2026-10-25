import { createHash } from 'node:crypto';
import { database } from './content-db.mjs';
import { currentUser,readJSON,sameOrigin } from './editorial-db.mjs';
import editing from './content-edit.js';

export const editSchema=[
 `CREATE TABLE IF NOT EXISTS atlas_content_edits (
   id uuid PRIMARY KEY, actor_id text NOT NULL REFERENCES atlas_users(id),
   topic_id text NOT NULL REFERENCES atlas_content(id), fingerprint text NOT NULL,
   action text NOT NULL CHECK(action IN ('delete','undo')), undo_of uuid REFERENCES atlas_content_edits(id),
   before_revision bigint NOT NULL, revision bigint NOT NULL, created_at timestamptz NOT NULL DEFAULT now()
 )`,
 `CREATE OR REPLACE FUNCTION atlas_edit_content(p_topic text,p_expected bigint,p_payload jsonb,p_actor text,p_request uuid,p_fingerprint text,p_action text,p_undo uuid)
 RETURNS TABLE(edit_id uuid,revision text) LANGUAGE plpgsql AS $$
 DECLARE existing atlas_content_edits%ROWTYPE; original atlas_content_edits%ROWTYPE;
 current_content atlas_content%ROWTYPE; saved_payload jsonb; after_payload jsonb; published bigint;
 BEGIN
   SELECT * INTO existing FROM atlas_content_edits e WHERE e.id=p_request;
   IF FOUND THEN
     IF existing.actor_id<>p_actor OR existing.fingerprint<>p_fingerprint THEN RAISE EXCEPTION 'ATLAS_EDIT_REUSED'; END IF;
     RETURN QUERY SELECT existing.id,existing.revision::text; RETURN;
   END IF;
   SELECT * INTO current_content FROM atlas_content c WHERE c.id=p_topic FOR UPDATE;
   IF NOT FOUND THEN RAISE EXCEPTION 'ATLAS_EDIT_NOT_FOUND'; END IF;
   SELECT * INTO existing FROM atlas_content_edits e WHERE e.id=p_request;
   IF FOUND THEN
     IF existing.actor_id<>p_actor OR existing.fingerprint<>p_fingerprint THEN RAISE EXCEPTION 'ATLAS_EDIT_REUSED'; END IF;
     RETURN QUERY SELECT existing.id,existing.revision::text; RETURN;
   END IF;
   IF current_content.revision<>p_expected THEN RAISE EXCEPTION 'ATLAS_EDIT_CONFLICT'; END IF;
   saved_payload:=p_payload;
   IF p_action='undo' THEN
     SELECT * INTO original FROM atlas_content_edits e WHERE e.id=p_undo AND e.actor_id=p_actor AND e.topic_id=p_topic AND e.action='delete';
     IF NOT FOUND THEN RAISE EXCEPTION 'ATLAS_EDIT_NOT_FOUND'; END IF;
     SELECT h.payload INTO after_payload FROM atlas_history h WHERE h.id=p_topic AND h.revision=original.revision;
     IF after_payload IS NULL OR current_content.payload<>after_payload THEN RAISE EXCEPTION 'ATLAS_EDIT_CONFLICT'; END IF;
     SELECT h.payload INTO saved_payload FROM atlas_history h WHERE h.id=p_topic AND h.revision=original.before_revision;
   ELSIF p_action<>'delete' THEN RAISE EXCEPTION 'ATLAS_EDIT_INVALID';
   END IF;
   IF saved_payload IS NULL OR saved_payload=current_content.payload THEN RAISE EXCEPTION 'ATLAS_EDIT_INVALID'; END IF;
   published:=atlas_publish(p_topic,p_expected,saved_payload,CASE WHEN p_action='undo' THEN 'Отмена удаления владельцем' ELSE 'Прямое удаление владельцем' END);
   INSERT INTO atlas_content_edits(id,actor_id,topic_id,fingerprint,action,undo_of,before_revision,revision)
     VALUES(p_request,p_actor,p_topic,p_fingerprint,p_action,p_undo,p_expected,published);
   RETURN QUERY SELECT p_request,published::text;
 END $$`
];
const json=(data,status=200)=>Response.json(data,{status,headers:{'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'}});
const uuid=/^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
export async function editResponse(request,connect=database){
 if(request.method!=='POST')return json({error:'Method not allowed'},405);
 if(!sameOrigin(request))return json({error:'Недопустимый источник запроса.'},403);
 try{
  const sql=await connect(),user=await currentUser(request,sql);
  if(!user)return json({error:'Войди, чтобы редактировать материал.'},401);
  if(!user.isModerator)return json({error:'Прямое редактирование доступно только владельцу.'},403);
  let input;try{input=await readJSON(request);}catch{return json({error:'Некорректная правка.'},400);}
  if(!input||!uuid.test(input.requestId)||!/^\d{1,18}$/.test(input.expectedRevision)||!/^[a-zA-Z0-9-][a-zA-Z0-9_-]{0,79}$/.test(input.topicId)||
    !['delete','undo'].includes(input.action)||(input.action==='undo'&&!uuid.test(input.editId))||
    (input.action==='delete'&&(!Array.isArray(input.targets)||input.targets.length<1||input.targets.length>80)))return json({error:'Проверь выделение и версию материала.'},400);
  const canonical={requestId:input.requestId,topicId:input.topicId,expectedRevision:input.expectedRevision,action:input.action,
   ...(input.action==='delete'?{targets:input.targets}:{editId:input.editId})};
  const fingerprint=createHash('sha256').update(JSON.stringify(canonical)).digest('hex');
  const previous=async()=>{
   const [existing]=await sql.query('SELECT id,revision::text,fingerprint FROM atlas_content_edits WHERE id=$1 AND actor_id=$2',[input.requestId,user.id]);
   return existing?(existing.fingerprint===fingerprint?json({editId:existing.id,revision:existing.revision}):json({error:'Запрос уже использован для другой правки.'},409)):null;
  };
  const existing=await previous();if(existing)return existing;
  let payload=null;
  if(input.action==='delete'){
   const [current]=await sql.query('SELECT payload,revision::text FROM atlas_content WHERE id=$1',[input.topicId]);
   if(!current)return json({error:'Материал не найден.'},404);
   if(current.revision!==input.expectedRevision)return await previous()||json({error:'Материал изменился. Обнови его и выдели фрагменты снова.'},409);
   try{payload=editing.remove(current.payload,input.targets);}catch(error){if(error instanceof editing.EditError)return json({error:error.message},400);throw error;}
  }
  const [saved]=await sql.query('SELECT * FROM atlas_edit_content($1,$2,$3::jsonb,$4,$5,$6,$7,$8)',[
   input.topicId,input.expectedRevision,payload?JSON.stringify(payload):null,user.id,input.requestId,fingerprint,input.action,input.editId||null]);
  return json({editId:saved.edit_id,revision:saved.revision});
 }catch(error){
  if(/ATLAS_EDIT_CONFLICT|ATLAS_EDIT_REUSED|duplicate key/.test(error.message))return json({error:'Материал изменился. Обнови его перед удалением или отменой.'},409);
  if(/ATLAS_EDIT_NOT_FOUND/.test(error.message))return json({error:'Правка или материал не найдены.'},404);
  if(/ATLAS_EDIT_INVALID/.test(error.message))return json({error:'Пустая или недопустимая правка.'},400);
  return json({error:'Не удалось сохранить правку. Повтор безопасен: повторного удаления не будет.'},503);
 }
}
