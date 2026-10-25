"""Run against an isolated PostgreSQL schema; never mutate real publications."""
import concurrent.futures
import json
import os
from pathlib import Path
import unittest
import uuid


class EditDatabaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get('ATLAS_TEST_DATABASE_URL'):
            raise unittest.SkipTest('Requires an isolated test database connection')
        import psycopg
        cls.pg=psycopg;cls.dsn=os.environ['ATLAS_TEST_DATABASE_URL'];cls.schema='atlas_edit_test_'+uuid.uuid4().hex
        cls.conn=psycopg.connect(cls.dsn,autocommit=True)
        cls.conn.execute('CREATE SCHEMA '+cls.schema)
        cls.conn.execute('SET search_path TO '+cls.schema)
        cls.conn.execute(Path(os.environ['ATLAS_TEST_SCHEMA_FILE']).read_text())
        cls.conn.execute("INSERT INTO atlas_users(id,login,name) VALUES('1615093','owner','Owner')")

    @classmethod
    def tearDownClass(cls):
        cls.conn.execute('DROP SCHEMA '+cls.schema+' CASCADE');cls.conn.close()

    def setUp(self):
        self.topic='test-'+uuid.uuid4().hex
        self.original={'node':{'id':self.topic,'what':'A\n\nB\n\nC'},'sections':{}}
        self.revision=self.conn.execute('INSERT INTO atlas_content(id,payload) VALUES(%s,%s::jsonb) RETURNING revision',(self.topic,json.dumps(self.original))).fetchone()[0]

    def publish(self,revision,payload,request=None,action='delete',undo=None,connection=None):
        request=request or str(uuid.uuid4())
        return (connection or self.conn).execute('SELECT * FROM atlas_edit_content(%s,%s,%s::jsonb,%s,%s,%s,%s,%s)',
            (self.topic,revision,json.dumps(payload) if payload is not None else None,'1615093',request,request,action,undo)).fetchone()

    def test_concurrent_retry_creates_one_publication_and_preserves_history(self):
        request=str(uuid.uuid4());changed={'node':{'id':self.topic,'what':'B'},'sections':{}}
        def call(_):
            with self.pg.connect(self.dsn,autocommit=True,options='-c search_path='+self.schema) as c:
                return self.publish(self.revision,changed,request,connection=c)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(call,range(3)))
        self.assertEqual(len(set(results)),1)
        self.assertEqual(self.conn.execute('SELECT count(*) FROM atlas_history WHERE id=%s',(self.topic,)).fetchone()[0],2)
        self.assertEqual(self.conn.execute('SELECT payload FROM atlas_content WHERE id=%s',(self.topic,)).fetchone()[0],changed)
        self.assertEqual(self.publish(self.revision,changed,request),results[0])

    def test_stale_write_cannot_clobber_and_undo_restores_exact_payload(self):
        changed={'node':{'id':self.topic,'what':'B'},'sections':{}}
        edit,revision=self.publish(self.revision,changed)
        with self.assertRaises(self.pg.Error):self.publish(self.revision,{'node':{'what':'WRONG'}})
        undo,new_revision=self.publish(revision,None,action='undo',undo=edit)
        self.assertGreater(int(new_revision),int(revision))
        self.assertEqual(self.conn.execute('SELECT payload FROM atlas_content WHERE id=%s',(self.topic,)).fetchone()[0],self.original)

    def test_undo_does_not_overwrite_intervening_agent_edits(self):
        edit,revision=self.publish(self.revision,{'node':{'what':'B'}})
        current=self.conn.execute('SELECT atlas_publish(%s,%s,%s::jsonb,%s)',(self.topic,revision,json.dumps({'node':{'what':'agent update'}}),'Agent')).fetchone()[0]
        with self.assertRaises(self.pg.Error):self.publish(current,None,action='undo',undo=edit)
        self.assertEqual(self.conn.execute('SELECT payload FROM atlas_content WHERE id=%s',(self.topic,)).fetchone()[0]['node']['what'],'agent update')

    def test_multiple_deletions_can_be_undone_in_reverse_order(self):
        first,revision=self.publish(self.revision,{'node':{'what':'B\n\nC'}})
        second,revision=self.publish(revision,{'node':{'what':'C'}})
        _,revision=self.publish(revision,None,action='undo',undo=second)
        _,revision=self.publish(revision,None,action='undo',undo=first)
        self.assertEqual(self.conn.execute('SELECT payload FROM atlas_content WHERE id=%s',(self.topic,)).fetchone()[0],self.original)

if __name__=='__main__':unittest.main()
