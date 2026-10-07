"""Local document processing CLI, using the same persisted records as the service."""
from pathlib import Path
import argparse
import json

from .documents import parse_document
from .store import Store


def main():
    parser=argparse.ArgumentParser(description='中文医学文献：原文、医学模型、证据核验与结构化导出')
    parser.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[2]/'workspace')
    commands=parser.add_subparsers(dest='command',required=True)
    ingest=commands.add_parser('ingest');ingest.add_argument('file',type=Path);ingest.add_argument('--format',choices=['text','jats'],default='text')
    ingest.add_argument('--title',default='');ingest.add_argument('--source',required=True)
    commands.add_parser('list')
    extract=commands.add_parser('extract');extract.add_argument('document_id');extract.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    review=commands.add_parser('review');review.add_argument('extraction_id');review.add_argument('record_id')
    review.add_argument('--revision',type=int,required=True);review.add_argument('--status',choices=['approved','rejected','pending'],required=True)
    review.add_argument('--reviewer',required=True);review.add_argument('--arm');review.add_argument('--endpoint');review.add_argument('--value');review.add_argument('--note')
    export=commands.add_parser('export');export.add_argument('extraction_id');export.add_argument('output',type=Path);export.add_argument('--all-candidates',action='store_true')
    search=commands.add_parser('search');search.add_argument('query')
    args=parser.parse_args();store=Store(args.workspace/'medical/literature.sqlite3')
    if args.command=='ingest':
        result=store.add_document(parse_document(args.file.read_text(encoding='utf-8-sig'),args.format,args.title,args.source))
        print(json.dumps({'id':result['id'],'metadata':result['metadata'],'characters':len(result['text'])},ensure_ascii=False,indent=2));return
    if args.command=='list':result=store.documents()
    elif args.command=='extract':
        from .pipeline import LiteratureExtractor
        result=store.add_extraction(args.document_id,LiteratureExtractor(args.workspace,args.device).extract(store.document(args.document_id)))
    elif args.command=='review':
        correction={k:v for k,v in {'arm':args.arm,'endpoint':args.endpoint,'normalized_value':args.value,'note':args.note}.items() if v is not None}
        result=store.review(args.extraction_id,args.record_id,args.revision,args.status,args.reviewer,correction)
    elif args.command=='search':result=store.search(args.query)
    else:
        result=store.extraction(args.extraction_id);document=store.document(result['document_id'])
        if not args.all_candidates:
            result['records']=[r for r in result['records'] if r['status']=='approved'];allowed={i for r in result['records'] for i in r['mentions']}
            allowed.update(e['id'] for e in result['entities'] if e['status']=='approved')
            result['entities']=[e for e in result['entities'] if e['id'] in allowed]
            result['concepts']=[c for c in result['concepts'] if any(i in allowed for i in c['mentions'])];result['relations']=[]
        result.update(source_metadata=document['metadata'],source_text=document['text'],export_mode='all' if args.all_candidates else 'approved')
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(args.output.resolve());return
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
