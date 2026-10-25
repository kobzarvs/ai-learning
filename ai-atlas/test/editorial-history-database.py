"""Check the exported history and notes queries in an isolated PostgreSQL schema.

Requires ATLAS_TEST_DATABASE_URL, ATLAS_TEST_SCHEMA_FILE (content, editorial,
agent and edit schemas), and ATLAS_TEST_HISTORY_QUERY_FILE (historyQuery).
ATLAS_TEST_NOTES_QUERY_FILE additionally enables notesQuery integration tests.
"""
import json
import os
from pathlib import Path
import unittest
import uuid


class HistoryDatabaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get('ATLAS_TEST_DATABASE_URL'):
            raise unittest.SkipTest('Requires an isolated test database connection')
        import psycopg
        from psycopg.rows import dict_row
        cls.schema = 'atlas_history_test_' + uuid.uuid4().hex
        cls.conn = psycopg.connect(os.environ['ATLAS_TEST_DATABASE_URL'], autocommit=True, row_factory=dict_row)
        cls.conn.execute('CREATE SCHEMA ' + cls.schema)
        try:
            cls.conn.execute('SET search_path TO ' + cls.schema)
            cls.conn.execute(Path(os.environ['ATLAS_TEST_SCHEMA_FILE']).read_text())
            cls.query = Path(os.environ['ATLAS_TEST_HISTORY_QUERY_FILE']).read_text().replace('$1', '%(before)s').replace('$2', '%(q)s')
            cls.notes_query = None
            if os.environ.get('ATLAS_TEST_NOTES_QUERY_FILE'):
                cls.notes_query = Path(os.environ['ATLAS_TEST_NOTES_QUERY_FILE']).read_text()
                for index, name in enumerate(['all', 'user', 'status', 'topic', 'before', 'id', 'q', 'limit'], 1):
                    cls.notes_query = cls.notes_query.replace('$' + str(index), '%(' + name + ')s')
            cls.conn.execute("INSERT INTO atlas_users(id,login,name) VALUES('1615093','owner','Owner'),('reader','reader','Reader')")
        except Exception:
            cls.conn.execute('DROP SCHEMA ' + cls.schema + ' CASCADE')
            cls.conn.close()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.conn.execute('DROP SCHEMA ' + cls.schema + ' CASCADE')
        cls.conn.close()

    def setUp(self):
        self.conn.execute('TRUNCATE atlas_content CASCADE')
        self.conn.execute('TRUNCATE atlas_history')
        self.conn.execute('UPDATE atlas_clock SET revision=0')

    def seed(self, topic, title='Initial title'):
        return self.conn.execute('INSERT INTO atlas_content(id,payload,note) VALUES(%s,%s::jsonb,%s) RETURNING revision',
            (topic, json.dumps({'node': {'id': topic, 'name': title, 'what': 'Original'}}), 'Initial import')).fetchone()['revision']

    def publish(self, topic, title='Published title', note='Published change', text=None):
        revision = self.conn.execute('SELECT revision FROM atlas_content WHERE id=%s', (topic,)).fetchone()['revision']
        payload = {'node': {'id': topic, 'name': title, 'what': text or str(uuid.uuid4())}}
        return self.conn.execute('SELECT atlas_publish(%s,%s,%s::jsonb,%s) AS revision',
            (topic, revision, json.dumps(payload), note)).fetchone()['revision']

    def note(self, topic, instruction='Original instruction', quote='Original quote', tab='methods', *, status='pending', user='1615093', note_id=None, created_at=None):
        note = note_id or str(uuid.uuid4())
        self.conn.execute("INSERT INTO atlas_editorial_notes(id,user_id,request_id,topic_id,tab,anchor,quote,content_revision,action,instruction,status,created_at) VALUES(%s,%s,%s,%s,%s,'{}',%s,0,'custom',%s,%s,COALESCE(%s::timestamptz,clock_timestamp()))",
            (note, user, str(uuid.uuid4()), topic, tab, quote, instruction, status, created_at))
        return note

    def agent_run(self, note, revision, summary='Agent summary', instruction='Moderator instruction', status='completed', *, lease_offset=None, error='', created_at=None):
        run_id = str(uuid.uuid4())
        self.conn.execute("INSERT INTO atlas_agent_runs(id,note_id,request_id,requested_by,snapshot,status,summary,result_revision,finished_at,lease_until,error,created_at) VALUES(%s,%s,%s,'1615093',%s::jsonb,%s,%s,%s,clock_timestamp(),clock_timestamp()+%s::interval,%s,COALESCE(%s::timestamptz,clock_timestamp()))",
            (run_id, note, str(uuid.uuid4()), json.dumps({'moderatorComment': instruction, 'secret': 'Must stay private'}), status, summary, revision, lease_offset, error, created_at))
        return run_id

    def history(self, q='', before=None):
        return self.conn.execute(self.query, {'before': before, 'q': q}).fetchall()

    def notes(self, *, all=True, user='1615093', status='all', topic='', before='', id='', q='', limit=20):
        if self.notes_query is None:
            self.skipTest('Requires exported notesQuery via ATLAS_TEST_NOTES_QUERY_FILE')
        return self.conn.execute(self.notes_query, {'all': all, 'user': user, 'status': status, 'topic': topic, 'before': before, 'id': id, 'q': q, 'limit': limit}).fetchall()

    def test_seed_catalog_and_unpublished_notes_are_excluded(self):
        self.seed('seed-only')
        self.note('seed-only')
        self.seed('_catalog')
        self.publish('_catalog')
        self.seed('published')
        revision = self.publish('published', title='Real publication')
        before = self.conn.execute('SELECT count(*) AS count FROM atlas_history').fetchone()['count']
        rows = self.history()
        self.assertEqual([(row['topic_id'], row['revision'], row['kind']) for row in rows], [('published', str(revision), 'publication')])
        self.assertEqual(rows[0]['tab'], 'article')
        self.assertIsNone(rows[0]['note_id'])
        self.assertEqual(self.conn.execute('SELECT count(*) AS count FROM atlas_history').fetchone()['count'], before)

    def test_agent_attribution_requires_completed_run_and_matching_topic(self):
        self.seed('target')
        revision = self.publish('target', note='Publication fallback')
        self.seed('other')
        self.agent_run(self.note('other'), revision, summary='Wrong topic')
        note = self.note('target')
        self.agent_run(note, revision, summary='Failed run', status='failed')
        row = self.history()[0]
        self.assertEqual((row['kind'], row['summary'], row['note_id']), ('publication', 'Publication fallback', None))
        self.agent_run(note, revision, summary='Prior completion')
        self.agent_run(note, revision, summary='Actual completion', instruction='')
        rows = self.history()
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]['kind'], rows[0]['summary'], rows[0]['instruction'], rows[0]['tab'], rows[0]['note_id']),
            ('agent', 'Actual completion', 'Original instruction', 'methods', note))
        self.assertEqual(set(rows[0]), {'topic_id', 'topic_title', 'revision', 'current_revision', 'created_at', 'kind', 'summary', 'instruction', 'tab', 'note_id'})

    def test_manual_deletion_and_undo_are_real_history_entries(self):
        original = self.seed('manual')
        request = str(uuid.uuid4())
        changed = json.dumps({'node': {'id': 'manual', 'name': 'Manual title'}})
        deletion = self.conn.execute("SELECT * FROM atlas_edit_content(%s,%s,%s::jsonb,'1615093',%s,%s,'delete',NULL)",
            ('manual', original, changed, request, request)).fetchone()
        undo_id = str(uuid.uuid4())
        undo = self.conn.execute("SELECT * FROM atlas_edit_content(%s,%s,NULL,'1615093',%s,%s,'undo',%s)",
            ('manual', deletion['revision'], undo_id, undo_id, request)).fetchone()
        rows = self.history()
        self.assertEqual([row['revision'] for row in rows], [undo['revision'], deletion['revision']])
        self.assertEqual([row['summary'] for row in rows], ['Отмена удаления владельцем', 'Прямое удаление владельцем'])
        self.assertTrue(all(row['kind'] == 'manual' and row['tab'] == 'article' and row['note_id'] is None for row in rows))
        self.assertTrue(all(row['current_revision'] == undo['revision'] for row in rows))

    def test_search_matches_all_public_context_and_treats_sql_wildcards_literally(self):
        self.seed('searchable')
        historical = self.publish('searchable', title='Historic Alias', note='History description')
        note = self.note('searchable', instruction='Original command', quote='Quoted sentence')
        self.agent_run(note, historical, summary='Agent explanation 50%_ growth', instruction='Moderator refinement')
        current = self.publish('searchable', title='Current Alias', note='Unrelated later change')
        self.seed('decoy')
        self.publish('decoy', note='Agent explanation 50ABC growth')
        for phrase in ['HISTORIC alias', 'history description', 'agent EXPLANATION 50%_', 'original command', 'QUOTED sentence', 'moderator refinement', '%_', '50%_']:
            with self.subTest(phrase=phrase):
                self.assertEqual([row['revision'] for row in self.history(phrase)], [str(historical)])
        self.assertEqual([row['revision'] for row in self.history('current alias')], [str(current), str(historical)])
        self.assertEqual(self.history("x' OR true --"), [])
        self.assertEqual(self.history('Must stay private'), [])
        self.assertEqual(self.history('historic alias')[0]['topic_title'], 'Historic Alias')

    def test_revision_cursor_pages_every_publication_once(self):
        self.seed('pages')
        revisions = [self.publish('pages') for _ in range(22)]
        first = self.history()
        self.assertEqual(len(first), 21)
        displayed = first[:20]
        second = self.history(before=displayed[-1]['revision'])
        self.assertEqual(len(second), 2)
        self.assertEqual([row['revision'] for row in displayed + second], [str(revision) for revision in reversed(revisions)])

    def test_inbox_contains_pending_and_latest_failed_or_expired_accepted_notes(self):
        self.seed('inbox')
        pending = self.note('inbox')
        failed = self.note('inbox', status='accepted')
        self.agent_run(failed, None, status='failed')
        expired = self.note('inbox', status='accepted')
        self.agent_run(expired, None, status='running', lease_offset='-1 minute')
        for note_status, run_status, lease in [
            ('done', 'failed', None), ('rejected', 'failed', None),
            ('done', 'running', '-1 minute'), ('rejected', 'running', '-1 minute'),
            ('accepted', 'running', '1 hour'), ('accepted', 'completed', None),
            ('accepted', 'queued', None), ('accepted', None, None),
        ]:
            note = self.note('inbox', status=note_status)
            if run_status:
                self.agent_run(note, None, status=run_status, lease_offset=lease)
        # A failure in an older attempt cannot resurrect a currently healthy job.
        for latest_status, lease in [('completed', None), ('queued', None), ('running', '1 hour')]:
            note = self.note('inbox', status='accepted')
            self.agent_run(note, None, status='failed', created_at='2026-01-01T00:00:00Z')
            self.agent_run(note, None, status=latest_status, lease_offset=lease, created_at='2026-01-02T00:00:00Z')
        rows = self.notes(status='inbox')
        self.assertEqual({str(row['id']) for row in rows}, {pending, failed, expired})
        self.assertEqual(len(rows), 3)
        self.assertEqual(next(row for row in rows if str(row['id']) == expired)['run']['status'], 'running')

    def test_notes_search_finds_older_run_context_and_literal_percent_underscore(self):
        self.seed('search-notes', title='Searchable topic title')
        note = self.note('search-notes', instruction='Original command', quote='Selected quote')
        self.conn.execute('UPDATE atlas_editorial_notes SET moderator_note=%s WHERE id=%s', ('Moderator note', note))
        self.agent_run(note, None, summary='Historical summary 50%_ growth', instruction='Earlier refinement', error='Historical error', status='failed', created_at='2026-01-01T00:00:00Z')
        latest = self.agent_run(note, None, summary='Latest completion', created_at='2026-01-02T00:00:00Z')
        other = self.note('search-notes')
        self.agent_run(other, None, summary='Historical summary 50ABC growth')
        for phrase in ['historical SUMMARY 50%_', '50%_', '%_', 'earlier REFINEMENT', 'historical error', 'original command', 'SELECTED quote', 'moderator note']:
            with self.subTest(phrase=phrase):
                rows = self.notes(q=phrase)
                self.assertEqual([str(row['id']) for row in rows], [note])
                self.assertEqual(rows[0]['run']['id'], latest)
                self.assertEqual(rows[0]['run']['summary'], 'Latest completion')
        self.assertEqual({str(row['id']) for row in self.notes(q='searchable TOPIC title')}, {note, other})
        self.assertEqual(self.notes(q="x' OR true --"), [])
        self.assertEqual(self.notes(q='Must stay private'), [])

    def test_notes_acl_and_cursor_keep_private_notes_out_of_every_page(self):
        self.seed('acl-notes')
        self.seed('another-topic')
        ids = {}
        # Identical timestamps make the UUID tie-breaker observable and stable.
        for index in range(1, 8):
            note_id = '00000000-0000-4000-8000-' + str(index).zfill(12)
            ids[index] = self.note('acl-notes', instruction='Shared match' if index % 2 else 'Private match', user='reader' if index % 2 else '1615093',
                note_id=note_id, created_at='2026-01-01T00:00:00Z')
        another = self.note('another-topic', user='reader', created_at='2025-01-01T00:00:00Z')
        first = self.notes(all=False, user='reader', topic='acl-notes', limit=2)
        self.assertEqual([str(row['id']) for row in first], [ids[7], ids[5], ids[3]])
        second = self.notes(all=False, user='reader', topic='acl-notes', before=ids[5], limit=2)
        self.assertEqual([str(row['id']) for row in second], [ids[3], ids[1]])
        self.assertEqual([str(row['id']) for row in first[:2] + second], [ids[7], ids[5], ids[3], ids[1]])
        self.assertEqual(self.notes(all=False, user='reader', q='Private match'), [])
        self.assertEqual(self.notes(all=False, user='reader', id=ids[6]), [])
        self.assertEqual(self.notes(all=False, user='reader', before=ids[6]), [])
        self.assertEqual(self.notes(all=False, user='reader', before=str(uuid.uuid4())), [])
        self.assertEqual([str(row['id']) for row in self.notes(all=False, user='reader', topic='another-topic')], [another])
        self.assertEqual([str(row['id']) for row in self.notes(all=False, user='reader', id=ids[3])], [ids[3]])
        self.assertEqual({str(row['id']) for row in self.notes(all=True, before=ids[6], topic='acl-notes')}, {ids[index] for index in range(1, 6)})


if __name__ == '__main__':
    unittest.main()
