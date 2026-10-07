"""SQLite source documents, immutable extractions and auditable record revisions."""
from pathlib import Path
import json
import sqlite3
import uuid
import hashlib
from contextlib import contextmanager

from ..training import now


class Conflict(ValueError): pass


class Store:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, sha TEXT UNIQUE NOT NULL, payload TEXT NOT NULL, created TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS extractions(id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id), payload TEXT NOT NULL, created TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS reviews(extraction_id TEXT NOT NULL REFERENCES extractions(id), record_id TEXT NOT NULL,
                    revision INTEGER NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL, reviewer TEXT NOT NULL, created TEXT NOT NULL,
                    PRIMARY KEY(extraction_id,record_id,revision));
                CREATE VIRTUAL TABLE IF NOT EXISTS evidence_search USING fts5(extraction_id UNINDEXED, record_id UNINDEXED, text, tokenize='trigram');
            ''')

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=20);db.row_factory=sqlite3.Row;db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:yield db
        finally:db.close()

    def add_document(self,document):
        with self.connect() as db:
            existing=db.execute('SELECT id FROM documents WHERE sha=?',(document['sha256'],)).fetchone()
            if existing:return self.document(existing['id'])
            identifier=uuid.uuid4().hex;document={**document,'id':identifier}
            db.execute('INSERT INTO documents VALUES(?,?,?,?)',(identifier,document['sha256'],json.dumps(document,ensure_ascii=False),now()))
        return document

    def document(self,identifier):
        with self.connect() as db:row=db.execute('SELECT payload FROM documents WHERE id=?',(identifier,)).fetchone()
        if not row:raise KeyError('Document not found')
        return json.loads(row['payload'])

    def documents(self):
        with self.connect() as db:rows=db.execute('SELECT payload,created FROM documents ORDER BY created DESC').fetchall()
        return [{'id':(d:=json.loads(r['payload']))['id'],'metadata':d['metadata'],'characters':len(d['text']),
                 'blocks':len(d['blocks']),'created':r['created'],'sha256':d['sha256']} for r in rows]

    def add_extraction(self,document_id,payload):
        identifier=uuid.uuid4().hex;payload={**payload,'id':identifier,'document_id':document_id,'created_at':now()}
        with self.connect() as db:
            db.execute('INSERT INTO extractions VALUES(?,?,?,?)',(identifier,document_id,json.dumps(payload,ensure_ascii=False),payload['created_at']))
            db.executemany('INSERT INTO evidence_search VALUES(?,?,?)',[(identifier,r['id'],r['evidence']['text']) for r in payload['records']])
        return self.extraction(identifier)

    def extraction(self,identifier):
        with self.connect() as db:
            row=db.execute('SELECT payload FROM extractions WHERE id=?',(identifier,)).fetchone()
            reviews=db.execute('SELECT * FROM reviews WHERE extraction_id=? ORDER BY revision',(identifier,)).fetchall()
        if not row:raise KeyError('Extraction not found')
        payload=json.loads(row['payload']);latest={r['record_id']:r for r in reviews}
        for record in [*payload['records'],*payload.get('entities',[])]:
            record['revision']=0
            record.setdefault('status','pending')
            if record['id'].startswith('e'):
                record['model_type']=record['type'];record['model_canonical']=record.get('canonical',record['text'])
            if record['id'] in latest:
                r=latest[record['id']];record.update(json.loads(r['payload']));record.update(status=r['status'],revision=r['revision'],reviewer=r['reviewer'],reviewed_at=r['created'])
        if payload.get('entities'):
            concepts={}
            from .data import TYPE_NAMES
            for e in payload['entities']:
                e['type_name']=TYPE_NAMES.get(e['type'],e['type'])
                canonical=e.get('canonical',e['text']);cid='local:'+hashlib.sha256((e['type']+'\0'+canonical).encode()).hexdigest()[:16]
                e['concept_id']=cid
                if e['status']=='rejected':continue
                concepts.setdefault(cid,{'id':cid,'canonical':canonical,'type':e['type'],'mentions':[]})['mentions'].append(e['id'])
            payload['concepts']=list(concepts.values())
        payload['review_counts']={s:sum(r['status']==s for r in payload['records']) for s in ('pending','approved','rejected')}
        payload['entity_review_counts']={s:sum(r['status']==s for r in payload.get('entities',[])) for s in ('pending','approved','rejected')}
        return payload

    def extractions(self,document_id):
        with self.connect() as db:rows=db.execute('SELECT id,created FROM extractions WHERE document_id=? ORDER BY created DESC',(document_id,)).fetchall()
        return [dict(r) for r in rows]

    def review(self,identifier,record_id,expected_revision,status,reviewer,correction=None):
        if status not in ('approved','rejected','pending') or not reviewer.strip():raise ValueError('Valid status and reviewer required')
        extraction=self.extraction(identifier);records={r['id']:r for r in extraction['records']}
        if record_id not in records:raise KeyError('Record not found')
        correction=correction or {}
        if set(correction)-{'arm','endpoint','normalized_value','note','evidence'}:raise ValueError('Unknown review field')
        if 'evidence' in correction:
            evidence=correction['evidence'];document=self.document(extraction['document_id']);a,b=evidence['start'],evidence['end']
            if not isinstance(a,int) or not isinstance(b,int) or not 0<=a<b<=len(document['text']):raise ValueError('Invalid evidence offsets')
            if document['text'][a:b]!=evidence['text']:raise ValueError('Evidence must match exact source text')
            block=next((x for x in document['blocks'] if x['start']<=a<b<=x['end']),None)
            if not block:raise ValueError('Reviewed evidence must lie within one source block')
            correction['evidence']={**records[record_id]['evidence'],**evidence,'block':block['id'],'section':block['section'],'locator':block['locator']}
        correction={**{k:records[record_id].get(k) for k in ('arm','endpoint','normalized_value','note','evidence')},**correction}
        # Acquire write transaction before checking current revision: concurrent reviewers cannot silently overwrite.
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT MAX(revision) AS revision FROM reviews WHERE extraction_id=? AND record_id=?',(identifier,record_id)).fetchone()
            current=row['revision'] or 0
            if current!=expected_revision:raise Conflict(f'Review revision changed: expected {expected_revision}, current {current}')
            db.execute('INSERT INTO reviews VALUES(?,?,?,?,?,?,?)',(identifier,record_id,current+1,status,json.dumps(correction,ensure_ascii=False),reviewer,now()))
        return self.extraction(identifier)

    def history(self,identifier,record_id):
        with self.connect() as db:rows=db.execute('SELECT revision,status,payload,reviewer,created FROM reviews WHERE extraction_id=? AND record_id=? ORDER BY revision',(identifier,record_id)).fetchall()
        return [{**dict(r),'payload':json.loads(r['payload'])} for r in rows]

    def review_entity(self,identifier,entity_id,expected_revision,status,reviewer,type_,canonical,note=None):
        from .data import TYPES
        if type_ not in [*TYPES,'other']:raise ValueError('Choose one of the nine medical types or other')
        if status not in ('approved','rejected','pending') or not reviewer.strip() or not canonical.strip():raise ValueError('Status, reviewer and canonical name required')
        extraction=self.extraction(identifier);entities={e['id']:e for e in extraction.get('entities',[])}
        if entity_id not in entities:raise KeyError('Entity not found')
        correction={'type':type_,'canonical':canonical.strip(),'note':note,'normalization':'human_review'}
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT MAX(revision) AS revision FROM reviews WHERE extraction_id=? AND record_id=?',(identifier,entity_id)).fetchone()
            current=row['revision'] or 0
            if current!=expected_revision:raise Conflict(f'Entity revision changed: expected {expected_revision}, current {current}')
            db.execute('INSERT INTO reviews VALUES(?,?,?,?,?,?,?)',(identifier,entity_id,current+1,status,json.dumps(correction,ensure_ascii=False),reviewer,now()))
        return self.extraction(identifier)

    def search(self,query):
        # Quoted terms prevent caller-supplied FTS syntax; SQL always uses parameters.
        terms=query.strip().split()
        if not terms or len(query)>200:raise ValueError('Search requires 1–200 characters')
        expression=' AND '.join('"'+term.replace('"','""')+'"' for term in terms)
        with self.connect() as db:
            if all(len(t)>=3 for t in terms):
                rows=db.execute('SELECT extraction_id,record_id,text FROM evidence_search WHERE evidence_search MATCH ? LIMIT 50',(expression,)).fetchall()
            else:
                condition=' AND '.join('instr(lower(text),lower(?))>0' for _ in terms)
                rows=db.execute('SELECT extraction_id,record_id,text FROM evidence_search WHERE '+condition+' LIMIT 50',terms).fetchall()
        return [dict(r) for r in rows]
