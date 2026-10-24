// Server only. No database credentials or write endpoint are sent to the browser.
export async function database() {
  const url = process.env.DATABASE_URL;
  if (!url) throw new Error('Content database is not configured');
  const { neon } = await import('@neondatabase/serverless');
  return neon(url, { fetchOptions: { signal: AbortSignal.timeout(10000) } });
}

export const schema = [
  `CREATE TABLE IF NOT EXISTS atlas_clock (
    id boolean PRIMARY KEY DEFAULT true CHECK (id), revision bigint NOT NULL DEFAULT 0
  )`,
  `INSERT INTO atlas_clock (id) VALUES (true) ON CONFLICT DO NOTHING`,
  `CREATE TABLE IF NOT EXISTS atlas_content (
    id text PRIMARY KEY, payload jsonb NOT NULL CHECK (jsonb_typeof(payload) = 'object'),
    revision bigint NOT NULL DEFAULT 0, note text NOT NULL DEFAULT '',
    updated_at timestamptz NOT NULL DEFAULT now()
  )`,
  `CREATE TABLE IF NOT EXISTS atlas_history (
    id text NOT NULL, revision bigint NOT NULL, payload jsonb NOT NULL,
    note text NOT NULL, updated_at timestamptz NOT NULL,
    PRIMARY KEY (id, revision)
  )`,
  // Locking the clock also orders concurrent commits, so incremental reads cannot miss one.
  `CREATE OR REPLACE FUNCTION atlas_record_change() RETURNS trigger LANGUAGE plpgsql AS $$
  BEGIN
    IF TG_OP = 'UPDATE' AND NEW.payload = OLD.payload THEN RETURN OLD; END IF;
    UPDATE atlas_clock SET revision = revision + 1 WHERE id = true RETURNING revision INTO NEW.revision;
    NEW.updated_at := clock_timestamp();
    INSERT INTO atlas_history (id, revision, payload, note, updated_at)
      VALUES (NEW.id, NEW.revision, NEW.payload, NEW.note, NEW.updated_at);
    RETURN NEW;
  END $$`,
  `CREATE OR REPLACE TRIGGER atlas_content_revision BEFORE INSERT OR UPDATE ON atlas_content
    FOR EACH ROW EXECUTE FUNCTION atlas_record_change()`,
  `CREATE OR REPLACE FUNCTION atlas_publish(topic text, expected_revision bigint, content jsonb, change_note text)
    RETURNS bigint LANGUAGE plpgsql AS $$
  DECLARE published_revision bigint;
  BEGIN
    UPDATE atlas_content SET payload = content, note = change_note
      WHERE id = topic AND revision = expected_revision RETURNING revision INTO published_revision;
    IF NOT FOUND THEN RAISE EXCEPTION 'Content changed; read the current revision before publishing'; END IF;
    RETURN published_revision;
  END $$`
];

export function seedRecords(map, lessons) {
  const records = [{ id: '_catalog', payload: { sources: map.sources, glossary: map.glossary, meta: lessons.meta } }];
  function visit(node) {
    const { children, ...fields } = node;
    records.push({ id: node.id, payload: { node: fields, lesson: lessons.topics[node.id] || null, layout: {}, sections: {} } });
    for (const child of children || []) visit(child);
  }
  visit(map.root);
  return records;
}

export async function seedDatabase(sql, records) {
  return sql.transaction([
    ...schema.map(statement => sql.query(statement)),
    sql.query(`INSERT INTO atlas_content (id, payload, note)
      SELECT item.id, item.payload, 'Первоначальный перенос материалов'
      FROM jsonb_to_recordset($1::jsonb) AS item(id text, payload jsonb)
      WHERE NOT EXISTS (SELECT 1 FROM atlas_content existing WHERE existing.id = item.id)
      ON CONFLICT DO NOTHING`, [JSON.stringify(records)])
  ]);
}

export const changesQuery = `SELECT
  (SELECT revision::text FROM atlas_clock WHERE id = true) AS revision,
  COALESCE(jsonb_agg(jsonb_build_object('id', id, 'payload', payload, 'revision', revision::text)
    ORDER BY revision) FILTER (WHERE id IS NOT NULL), '[]'::jsonb) AS records
  FROM atlas_content WHERE revision > $1::bigint`;

export async function contentResponse(request, connect = database) {
  const headers = { 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' };
  if (request.method !== 'GET') return Response.json({ error: 'Method not allowed' }, { status: 405, headers: { ...headers, Allow: 'GET' } });
  const since = new URL(request.url).searchParams.get('since') || '0';
  if (!/^\d{1,18}$/.test(since)) return Response.json({ error: 'Invalid revision' }, { status: 400, headers });
  try {
    const sql = await connect();
    const [result] = await sql.query(changesQuery, [since]);
    return Response.json(result, { headers });
  } catch {
    // Do not expose connection strings, SQL errors, or unpublished history.
    return Response.json({ error: 'Материалы временно недоступны' }, { status: 503, headers });
  }
}
