"""Local document processing CLI, using the same persisted records as the service."""
from pathlib import Path
import argparse
import json

from .documents import parse_document
from .store import Store
from .exporting import export_payload
from .backup import create_snapshot


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
    backup=commands.add_parser('backup',help='使用 SQLite 在线备份 API 创建一致性文献与核验记录快照')
    backup.add_argument('--output',type=Path,required=True,help='备份目标；必须不存在并位于加密的独立备份卷')
    restore=commands.add_parser('restore',help='校验备份并生成新的、未覆盖任何现有文件的恢复副本')
    restore.add_argument('--input',type=Path,required=True,help='待恢复的 SQLite 备份')
    restore.add_argument('--output',type=Path,required=True,help='新恢复副本路径；不得是当前工作数据库')
    args=parser.parse_args()
    database=args.workspace/'medical/literature.sqlite3'
    if args.command=='backup':
        result=create_snapshot(database,args.output)
        print(json.dumps({'backup':result['path'],**{k:v for k,v in result.items() if k!='path'}},ensure_ascii=False,indent=2));return
    if args.command=='restore':
        result=create_snapshot(args.input,args.output)
        print(json.dumps({'restored_copy':result['path'],**{k:v for k,v in result.items() if k!='path'}},ensure_ascii=False,indent=2));return
    store=Store(database)
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
        result=export_payload(store,args.extraction_id,'all' if args.all_candidates else 'approved')
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(args.output.resolve());return
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
