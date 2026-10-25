import { database } from './content-db.mjs';
const json=(data,status=200)=>Response.json(data,{status,headers:{'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'}});
const tabs=new Set(['article','methods','python','rust','mlx','exercise']);
function result(item){
 if(!item||!/^[a-zA-Z0-9_-]+$/.test(item.topicId)||!tabs.has(item.tab))return null;
 const anchor={};for(const key of ['section','methodId','guideView','caseTitle','exerciseLanguage'])if(typeof item.anchor?.[key]==='string')anchor[key]=item.anchor[key].slice(0,300);
 return {topicId:item.topicId,title:String(item.title||'').slice(0,300),tab:item.tab,section:String(item.section||'').slice(0,200),snippet:String(item.snippet||'').slice(0,400),anchor};
}
export async function searchResponse(request,connect=database){
 if(request.method!=='GET')return json({error:'Method not allowed'},405);
 const raw=new URL(request.url).searchParams.get('q')||'';
 const query=raw.normalize('NFKC').replace(/[\u0000-\u001f\u007f]/g,' ').replace(/\s+/g,' ').trim().toLocaleLowerCase('ru');
 if(query.length<2||query.length>300||raw.length>1000)return json({error:'Введи запрос от 2 до 300 символов.'},400);
 try{
  const sql=await connect(),[row]=await sql.query('SELECT * FROM atlas_request_search($1)',[query]);
  if(row?.status==='completed'){
   const groups=Array.isArray(row.results)?{semanticResults:row.results}:row.results||{};
   const clean=items=>(Array.isArray(items)?items:[]).slice(0,15).map(result).filter(Boolean);
   return json({keywordResults:clean(groups.keywordResults),semanticResults:clean(groups.semanticResults),semanticUnavailable:groups.semanticUnavailable===true,revision:String(row.revision)});
  }
  if(['queued','running'].includes(row?.status))return json({pending:true},202);
  return json({error:'Поиск по смыслу временно недоступен.'},503);
 }catch(error){return json({error:'Поиск по смыслу временно недоступен.'},String(error.message).includes('ATLAS_SEARCH_BUSY')?429:503);}
}
