from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Literal
import uuid
import xml.etree.ElementTree as ET

from fastapi import APIRouter,HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel,Field

from ..data import read_json,save_json
from ..training import now
from .documents import parse_document
from .store import Store,Conflict
from .exporting import export_payload


class DocumentRequest(BaseModel):
    content:str=Field(min_length=1,max_length=500_000)
    format:Literal['text','jats']='text'
    title:str=Field(default='',max_length=500)
    source:str=Field(default='user-provided',max_length=2000)


class ReviewRequest(BaseModel):
    expected_revision:int=Field(ge=0)
    status:Literal['approved','rejected','pending']
    reviewer:str=Field(min_length=1,max_length=100)
    arm:str|None=Field(default=None,max_length=1000)
    endpoint:str|None=Field(default=None,max_length=1000)
    normalized_value:str|None=Field(default=None,max_length=1000)
    note:str|None=Field(default=None,max_length=2000)


class EntityReviewRequest(BaseModel):
    expected_revision:int=Field(ge=0)
    status:Literal['approved','rejected','pending']
    reviewer:str=Field(min_length=1,max_length=100)
    type:str=Field(min_length=1,max_length=20)
    canonical:str=Field(min_length=1,max_length=1000)
    note:str|None=Field(default=None,max_length=2000)


def router(workspace):
    root=Path(workspace)/'medical';store=Store(root/'literature.sqlite3');routes=APIRouter(prefix='/api/medical')
    executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='medical-extract');busy=Lock();cached=[]
    jobs=root/'jobs';jobs.mkdir(parents=True,exist_ok=True)
    for path in jobs.glob('*.json'):
        old=read_json(path)
        if old['status'] in ('queued','running'):
            old.update(status='interrupted',error='Server restarted; resubmit extraction');save_json(path,old)
    def doc(identifier):
        try:return store.document(identifier)
        except KeyError as e:raise HTTPException(404,str(e)) from e
    def extract(identifier):
        try:return store.extraction(identifier)
        except KeyError as e:raise HTTPException(404,str(e)) from e
    @routes.get('')
    def overview():
        from .runtime import relation_artifact,relation_training_status,latest_comparison
        active=relation_artifact(root) if (root/'report.json').exists() else None
        return {'status':read_json(root/'status.json') if (root/'status.json').exists() else {'phase':'not_trained'},
                'report':read_json(root/'report.json') if (root/'report.json').exists() else None,'documents':store.documents(),
                'decoding_experiment':read_json(root/'decoding-v2/report.json') if (root/'decoding-v2/report.json').exists() else None,
                'decoder_policy':read_json(root/'decoder_policy.json') if (root/'decoder_policy.json').exists() else {'enabled':False},
                'active_relation':{k:v for k,v in active.items() if k!='run'} if active else None,
                'relation_comparison':latest_comparison(root),
                'relation_training':relation_training_status(root)}
    @routes.post('/documents',status_code=201)
    def ingest(body:DocumentRequest):
        try:return store.add_document(parse_document(body.content,body.format,body.title,body.source))
        except (ValueError,TypeError) as e:raise HTTPException(400,str(e)) from e
        except ET.ParseError as e:raise HTTPException(400,'Invalid JATS XML') from e
    @routes.get('/documents/{identifier}')
    def document(identifier:str):return {**doc(identifier),'extractions':store.extractions(identifier)}
    @routes.post('/documents/{identifier}/extract',status_code=202)
    def enqueue(identifier:str):
        document=doc(identifier)
        if not (root/'status.json').exists() or read_json(root/'status.json')['phase']!='completed':
            raise HTTPException(409,'医学模型仍在训练；完成后可提交文档抽取')
        if not busy.acquire(blocking=False):raise HTTPException(409,'已有文档正在抽取，请等待完成')
        jid=uuid.uuid4().hex;path=jobs/(jid+'.json');job={'id':jid,'document_id':identifier,'status':'queued','created_at':now()}
        save_json(path,job)
        def work():
            try:
                job.update(status='running',started_at=now());save_json(path,job)
                from .runtime import relation_artifact
                active=relation_artifact(root)
                if not cached or cached[0].fingerprints['relations']!=active['sha256']:
                    from .pipeline import LiteratureExtractor
                    cached[:]=[LiteratureExtractor(workspace)]
                result=store.add_extraction(identifier,cached[0].extract(document))
                job.update(status='completed',extraction_id=result['id'],finished_at=now())
            except Exception as e:job.update(status='failed',error=f'{type(e).__name__}: {e}')
            finally:save_json(path,job);busy.release()
        try:executor.submit(work)
        except Exception:busy.release();raise
        return job
    @routes.get('/jobs/{identifier}')
    def job(identifier:str):
        if len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier):raise HTTPException(400,'Invalid job ID')
        path=jobs/(identifier+'.json')
        if not path.exists():raise HTTPException(404,'Job not found')
        return read_json(path)
    @routes.get('/extractions/{identifier}')
    def result(identifier:str):return extract(identifier)
    @routes.post('/extractions/{identifier}/records/{record_id}/review')
    def review(identifier:str,record_id:str,body:ReviewRequest,request:Request):
        reviewer=getattr(request.state,'medical_user',None) or body.reviewer
        try:return store.review(identifier,record_id,body.expected_revision,body.status,reviewer,
                                 body.model_dump(exclude={'expected_revision','status','reviewer'},exclude_unset=True))
        except Conflict as e:raise HTTPException(409,str(e)) from e
        except KeyError as e:raise HTTPException(404,str(e)) from e
        except (ValueError,TypeError) as e:raise HTTPException(400,str(e)) from e
    @routes.get('/extractions/{identifier}/records/{record_id}/history')
    @routes.get('/extractions/{identifier}/entities/{record_id}/history')
    def history(identifier:str,record_id:str):
        extract(identifier);return store.history(identifier,record_id)
    @routes.post('/extractions/{identifier}/entities/{entity_id}/review')
    def entity_review(identifier:str,entity_id:str,body:EntityReviewRequest,request:Request):
        reviewer=getattr(request.state,'medical_user',None) or body.reviewer
        try:return store.review_entity(identifier,entity_id,body.expected_revision,body.status,reviewer,body.type,body.canonical,body.note)
        except Conflict as e:raise HTTPException(409,str(e)) from e
        except KeyError as e:raise HTTPException(404,str(e)) from e
        except ValueError as e:raise HTTPException(400,str(e)) from e
    @routes.get('/extractions/{identifier}/export')
    def export(identifier:str,mode:Literal['approved','all']='approved'):
        extract(identifier);payload=export_payload(store,identifier,mode)
        return JSONResponse(payload,headers={'Content-Disposition':f'attachment; filename="medical-evidence-{mode}.json"','Cache-Control':'no-store'})
    @routes.get('/search')
    def search(q:str):
        try:return store.search(q)
        except ValueError as e:raise HTTPException(400,str(e)) from e
    return routes
