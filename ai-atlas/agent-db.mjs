export const agentSchema = [
 `CREATE TABLE IF NOT EXISTS atlas_agent_runs (
   id uuid PRIMARY KEY, note_id uuid NOT NULL REFERENCES atlas_editorial_notes(id),
   request_id uuid NOT NULL, requested_by text NOT NULL REFERENCES atlas_users(id),
   snapshot jsonb NOT NULL, priority integer NOT NULL DEFAULT 0,
   status text NOT NULL DEFAULT 'queued' CHECK(status IN ('queued','running','completed','failed')),
   phase text NOT NULL DEFAULT 'Ожидает исполнителя', summary text NOT NULL DEFAULT '', error text NOT NULL DEFAULT '',
   worker_id text, lease_token uuid, lease_until timestamptz, heartbeat_at timestamptz,
   result_revision bigint, created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
   started_at timestamptz, finished_at timestamptz, UNIQUE(requested_by,request_id)
 )`,
 `CREATE UNIQUE INDEX IF NOT EXISTS atlas_agent_one_active ON atlas_agent_runs(note_id) WHERE status IN ('queued','running')`,
 `CREATE INDEX IF NOT EXISTS atlas_agent_queue ON atlas_agent_runs(priority DESC,created_at) WHERE status='queued'`,
 `CREATE TABLE IF NOT EXISTS atlas_agent_workers (id text PRIMARY KEY, last_seen timestamptz NOT NULL DEFAULT now())`,
 `CREATE OR REPLACE FUNCTION atlas_enqueue_run(target uuid, expected_version integer, request_key uuid,
   actor text, comment_text text, run_priority integer) RETURNS atlas_agent_runs LANGUAGE plpgsql AS $$
 DECLARE n atlas_editorial_notes; job atlas_agent_runs; previous jsonb;
 BEGIN
   SELECT * INTO n FROM atlas_editorial_notes WHERE id=target FOR UPDATE;
   IF NOT FOUND THEN RAISE EXCEPTION 'ATLAS_NOTE_NOT_FOUND'; END IF;
   UPDATE atlas_agent_runs SET status='failed',phase='Выполнение прервано',
     error='Исполнитель перестал отвечать. Можно запустить задание повторно.',finished_at=now(),updated_at=now()
     WHERE note_id=target AND status='running' AND lease_until<now();
   SELECT * INTO job FROM atlas_agent_runs WHERE requested_by=actor AND request_id=request_key;
   IF FOUND THEN
     IF job.note_id<>target THEN RAISE EXCEPTION 'ATLAS_REQUEST_REUSED'; END IF;
     RETURN job;
   END IF;
   SELECT * INTO job FROM atlas_agent_runs WHERE note_id=target AND status IN ('queued','running');
   IF FOUND THEN RETURN job; END IF;
   IF n.version<>expected_version THEN RAISE EXCEPTION 'ATLAS_VERSION_CONFLICT'; END IF;
   IF n.status IN ('done','rejected') THEN RAISE EXCEPTION 'ATLAS_NOTE_CLOSED'; END IF;
   SELECT jsonb_agg(item ORDER BY created_at) INTO previous FROM (
     SELECT created_at,jsonb_build_object('status',status,'summary',summary,'error',error,
       'moderatorComment',snapshot->>'moderatorComment','resultRevision',result_revision::text) AS item
     FROM atlas_agent_runs WHERE note_id=target ORDER BY created_at DESC,id DESC LIMIT 5
   ) history;
   INSERT INTO atlas_agent_runs(id,note_id,request_id,requested_by,snapshot,priority)
     VALUES(gen_random_uuid(),target,request_key,actor,
       jsonb_build_object('topicId',n.topic_id,'tab',n.tab,'anchor',n.anchor,'quote',n.quote,
         'contentRevision',n.content_revision::text,'action',n.action,'instruction',n.instruction,
         'moderatorComment',comment_text,'requestedBy',actor,'previousRuns',COALESCE(previous,'[]'::jsonb)),run_priority)
     RETURNING * INTO job;
   UPDATE atlas_editorial_notes SET status='accepted',moderator_note=comment_text,moderated_by=actor,
     version=version+1,updated_at=now() WHERE id=target;
   INSERT INTO atlas_editorial_events(note_id,actor_id,from_status,to_status,comment)
     VALUES(target,actor,n.status,'accepted','Запущено агентом: '||comment_text);
   PERFORM pg_notify('atlas_agent_jobs',job.id::text);
   RETURN job;
 END $$`,
 `CREATE OR REPLACE FUNCTION atlas_rework_run(target uuid, expected_version integer, request_key uuid,
   actor text, comment_text text, run_priority integer) RETURNS atlas_agent_runs LANGUAGE plpgsql AS $$
 DECLARE n atlas_editorial_notes; job atlas_agent_runs;
 BEGIN
   SELECT * INTO n FROM atlas_editorial_notes WHERE id=target FOR UPDATE;
   IF NOT FOUND THEN RAISE EXCEPTION 'ATLAS_NOTE_NOT_FOUND'; END IF;
   SELECT * INTO job FROM atlas_agent_runs WHERE requested_by=actor AND request_id=request_key;
   IF FOUND THEN
     IF job.note_id<>target THEN RAISE EXCEPTION 'ATLAS_REQUEST_REUSED'; END IF;
     RETURN job;
   END IF;
   SELECT * INTO job FROM atlas_agent_runs WHERE note_id=target AND status IN ('queued','running');
   IF FOUND THEN RETURN job; END IF;
   IF n.version<>expected_version THEN RAISE EXCEPTION 'ATLAS_VERSION_CONFLICT'; END IF;
   IF n.status<>'done' OR NOT EXISTS(SELECT 1 FROM atlas_agent_runs WHERE note_id=target AND status='completed')
     THEN RAISE EXCEPTION 'ATLAS_REWORK_UNAVAILABLE'; END IF;
   IF btrim(comment_text)='' THEN RAISE EXCEPTION 'ATLAS_REWORK_EMPTY'; END IF;
   UPDATE atlas_editorial_notes SET status='pending' WHERE id=target;
   INSERT INTO atlas_editorial_events(note_id,actor_id,from_status,to_status,comment)
     VALUES(target,actor,'done','pending','Доработка: '||comment_text);
   SELECT * INTO job FROM atlas_enqueue_run(target,expected_version,request_key,actor,comment_text,run_priority);
   UPDATE atlas_agent_runs SET snapshot=snapshot||jsonb_build_object('requestKind','rework')
     WHERE id=job.id RETURNING * INTO job;
   RETURN job;
 END $$`
];

export const publicRunFields=['id','note_id','status','phase','summary','error','created_at','updated_at','started_at','finished_at','heartbeat_at','result_revision','note_version','note_status','instruction'];
export function publicRun(run){
 if(!run)return null;
 const result=Object.fromEntries(publicRunFields.map(key=>[key,run[key]??null]));
 if(run.status==='running'&&run.lease_until&&Date.parse(run.lease_until)<Date.now()){
  result.status='failed';result.phase='Выполнение прервано';result.error='Исполнитель перестал отвечать. Можно запустить задание повторно.';
 }
 return result;
}
export async function workerStatus(sql){
 const [worker]=await sql.query(`SELECT max(last_seen) AS last_seen,
   COALESCE(max(last_seen)>now()-interval '75 seconds',false) AS online FROM atlas_agent_workers`);
 return {online:worker?.online===true,lastSeen:worker?.last_seen||null};
}
export async function enqueueRun(sql,{noteId,version,requestId,actorId,comment='',priority=10,kind='execute'}){
 const operation=kind==='rework'?'atlas_rework_run':'atlas_enqueue_run';
 const [run]=await sql.query(`SELECT * FROM ${operation}($1::uuid,$2,$3::uuid,$4,$5,$6)`,[noteId,version,requestId,actorId,comment,priority]);
 if(!run?.id)throw Error('Agent queue unavailable');
 return publicRun(run);
}

export async function listRuns(sql,user,noteIds=[]){
 const rows=await sql.query(`SELECT r.*,r.snapshot->>'moderatorComment' AS instruction,n.version AS note_version,n.status AS note_status FROM atlas_agent_runs r JOIN atlas_editorial_notes n ON n.id=r.note_id
   WHERE ($1::boolean OR n.user_id=$2)
   AND (cardinality($3::uuid[])=0 OR r.note_id=ANY($3::uuid[]))
   AND r.id=(SELECT latest.id FROM atlas_agent_runs latest WHERE latest.note_id=r.note_id ORDER BY latest.created_at DESC,latest.id DESC LIMIT 1)
   ORDER BY CASE WHEN r.status IN ('queued','running') THEN 0 ELSE 1 END,r.updated_at DESC LIMIT 100`,[user.isModerator,user.id,noteIds]);
 return rows.map(publicRun);
}

export async function runHistory(sql,user,noteId){
 const rows=await sql.query(`SELECT r.*,r.snapshot->>'moderatorComment' AS instruction FROM atlas_agent_runs r
   JOIN atlas_editorial_notes n ON n.id=r.note_id WHERE ($1::boolean OR n.user_id=$2) AND r.note_id=$3::uuid
   ORDER BY r.created_at DESC,r.id DESC LIMIT 20`,[user.isModerator,user.id,noteId]);
 return rows.map(publicRun);
}
