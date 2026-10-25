export const searchSchema=[
 `CREATE TABLE IF NOT EXISTS atlas_search_worker (id boolean PRIMARY KEY DEFAULT true CHECK(id),last_seen timestamptz NOT NULL DEFAULT now(),ready boolean NOT NULL DEFAULT false,indexed_revision bigint NOT NULL DEFAULT 0,topics integer NOT NULL DEFAULT 0,chunks integer NOT NULL DEFAULT 0)`,
 `ALTER TABLE atlas_search_worker ADD COLUMN IF NOT EXISTS index_generation bigint NOT NULL DEFAULT 0`,
 `CREATE TABLE IF NOT EXISTS atlas_search_topics (topic_id text PRIMARY KEY,revision bigint NOT NULL,point_ids jsonb NOT NULL,updated_at timestamptz NOT NULL DEFAULT now())`,
 `ALTER TABLE atlas_search_topics ADD COLUMN IF NOT EXISTS extractor_version text NOT NULL DEFAULT ''`,
 `CREATE TABLE IF NOT EXISTS atlas_search_chunks (id uuid PRIMARY KEY,topic_id text NOT NULL,revision bigint NOT NULL,payload jsonb NOT NULL)`,
 `CREATE INDEX IF NOT EXISTS atlas_search_chunks_topic ON atlas_search_chunks(topic_id)`,
 `CREATE INDEX IF NOT EXISTS atlas_search_chunks_text ON atlas_search_chunks USING gin(to_tsvector('russian',payload->>'text'))`,
 `CREATE TABLE IF NOT EXISTS atlas_search_requests (key text PRIMARY KEY,query text NOT NULL,revision bigint NOT NULL,status text NOT NULL DEFAULT 'queued' CHECK(status IN ('queued','running','completed','failed')),results jsonb NOT NULL DEFAULT '[]',created_at timestamptz NOT NULL DEFAULT now(),updated_at timestamptz NOT NULL DEFAULT now(),expires_at timestamptz NOT NULL DEFAULT now()+interval '5 minutes',lease_token uuid,lease_until timestamptz,attempts integer NOT NULL DEFAULT 0)`,
 `CREATE INDEX IF NOT EXISTS atlas_search_requests_queue ON atlas_search_requests(created_at) WHERE status='queued'`,
 `CREATE TABLE IF NOT EXISTS atlas_search_budget (minute timestamptz PRIMARY KEY,requests integer NOT NULL)`,
 `CREATE OR REPLACE FUNCTION atlas_request_search(input_query text) RETURNS SETOF atlas_search_requests LANGUAGE plpgsql AS $$
 DECLARE current_revision bigint; request_key text; cached atlas_search_requests; budget integer; corpus_signature text; generation bigint;
 BEGIN
  IF char_length(input_query)<2 OR char_length(input_query)>300 THEN RAISE EXCEPTION 'ATLAS_SEARCH_INVALID'; END IF;
  SELECT revision INTO current_revision FROM atlas_clock WHERE id=true;
  SELECT md5(COALESCE(string_agg(id || ':' || revision::text,',' ORDER BY id),'')) INTO corpus_signature FROM atlas_content;
  SELECT index_generation INTO generation FROM atlas_search_worker WHERE id=true;
  request_key:=md5(input_query || ':' || current_revision::text || ':' || corpus_signature || ':' || COALESCE(generation,0)::text);
  PERFORM pg_advisory_xact_lock(hashtext('ai_atlas_search'));
  SELECT * INTO cached FROM atlas_search_requests WHERE key=request_key AND expires_at>now();
  IF FOUND AND (cached.status='completed' OR cached.status='queued' OR (cached.status='running' AND cached.lease_until>now()) OR (cached.status='failed' AND cached.updated_at>now()-interval '10 seconds')) THEN
   RETURN NEXT cached; RETURN;
  END IF;
  IF NOT EXISTS(SELECT 1 FROM atlas_search_worker WHERE id=true AND ready AND last_seen>now()-interval '75 seconds') THEN RAISE EXCEPTION 'ATLAS_SEARCH_OFFLINE'; END IF;
  IF (SELECT count(*) FROM atlas_search_requests WHERE status IN ('queued','running') AND expires_at>now())>=30 THEN RAISE EXCEPTION 'ATLAS_SEARCH_BUSY'; END IF;
  INSERT INTO atlas_search_budget(minute,requests) VALUES(date_trunc('minute',now()),1) ON CONFLICT(minute) DO UPDATE SET requests=atlas_search_budget.requests+1 RETURNING requests INTO budget;
  IF budget>90 THEN RAISE EXCEPTION 'ATLAS_SEARCH_BUSY'; END IF;
  INSERT INTO atlas_search_requests(key,query,revision) VALUES(request_key,input_query,current_revision)
  ON CONFLICT(key) DO UPDATE SET status='queued',results='[]',updated_at=now(),expires_at=now()+interval '5 minutes',lease_token=NULL,lease_until=NULL,attempts=0
  RETURNING * INTO cached;
  PERFORM pg_notify('atlas_search_jobs',request_key);
  RETURN NEXT cached;
 END $$`
];
