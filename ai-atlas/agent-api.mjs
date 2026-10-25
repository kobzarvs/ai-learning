import { database } from './content-db.mjs';
import { currentUser,readJSON,sameOrigin } from './editorial-db.mjs';
import { enqueueRun,listRuns,runHistory,workerStatus } from './agent-db.mjs';
const json=(data,status=200)=>Response.json(data,{status,headers:{'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'}});
const uuid=/^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
export async function agentRunsResponse(request,connect=database){
 if(!['GET','POST'].includes(request.method))return json({error:'Method not allowed'},405);
 if(request.method==='POST'&&!sameOrigin(request))return json({error:'Недопустимый источник запроса'},403);
 try{
  const sql=await connect(),user=await currentUser(request,sql);
  if(!user)return json({error:'Войди, чтобы видеть задания.'},401);
  if(request.method==='GET'){
   const params=new URL(request.url).searchParams,history=params.get('history');
   if(history!==null){
    if(!uuid.test(history))return json({error:'Неверное замечание.'},400);
    return json({runs:await runHistory(sql,user,history),worker:await workerStatus(sql)});
   }
   const noteIds=(params.get('notes')||'').split(',').filter(Boolean);
   if(noteIds.length>50||noteIds.some(id=>!uuid.test(id)))return json({error:'Неверный список замечаний.'},400);
   return json({runs:await listRuns(sql,user,noteIds),worker:await workerStatus(sql)});
  }
  if(!user.isModerator)return json({error:'Запуск доступен только владельцу.'},403);
  let input;try{input=await readJSON(request);}catch{return json({error:'Некорректное задание.'},400);}
  if(!input||Array.isArray(input)||!uuid.test(input.noteId)||!uuid.test(input.requestId)||!Number.isSafeInteger(input.version)||input.version<1||
    typeof input.comment!=='string'||input.comment.length>3000||
    !['execute','rework'].includes(input.kind??'execute')||(input.kind==='rework'&&!input.comment.trim()))return json({error:'Проверь замечание и инструкцию.'},400);
  const run=await enqueueRun(sql,{...input,actorId:user.id,comment:input.comment.trim(),priority:10});
  return json({run,worker:await workerStatus(sql)},202);
 }catch(error){
  for(const [code,status,message]of[
   ['ATLAS_NOTE_NOT_FOUND',404,'Замечание не найдено.'],
   ['ATLAS_VERSION_CONFLICT',409,'Замечание изменилось. Обнови список перед запуском.'],
   ['ATLAS_REWORK_UNAVAILABLE',409,'Доработка доступна после завершения агента.'],
   ['ATLAS_REWORK_EMPTY',400,'Напиши, что нужно доработать.'],
   ['ATLAS_NOTE_CLOSED',409,'Сначала верни замечание на рассмотрение.'],
   ['ATLAS_REQUEST_REUSED',409,'Этот запрос уже использован для другого замечания.']
  ])if(String(error.message).includes(code))return json({error:message},status);
  return json({error:'Не удалось отправить задание. Повтори попытку — повторного запуска не будет.'},503);
 }
}
