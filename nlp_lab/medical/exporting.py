"""One export contract shared by HTTP and CLI; no dangling concept references."""
from copy import deepcopy

from ..training import now


def export_payload(store, identifier, mode='approved'):
    if mode not in ('approved','all'): raise ValueError('Export mode must be approved or all')
    payload=deepcopy(store.extraction(identifier));document=store.document(payload['document_id'])
    if mode=='approved':
        payload['records']=[r for r in payload['records'] if r['status']=='approved']
        allowed={e['id'] for e in payload['entities'] if e['status']=='approved'}
        allowed.update(i for r in payload['records'] for i in r['mentions'])
        payload['entities']=[e for e in payload['entities'] if e['id'] in allowed and e['status']!='rejected']
        allowed={e['id'] for e in payload['entities']}
        for r in payload['records']:
            original=r['mentions'];r['mentions']=[i for i in original if i in allowed]
            r['omitted_rejected_mentions']=len(original)-len(r['mentions'])
        payload['concepts']=[{**c,'mentions':[i for i in c['mentions'] if i in allowed]}
                             for c in payload['concepts'] if any(i in allowed for i in c['mentions'])]
        payload['relations']=[]
    else:
        payload['review_history']={r['id']:history for r in [*payload['records'],*payload['entities']]
                                   if (history:=store.history(identifier,r['id']))}
    payload.update(export_mode=mode,export_contract='medical_evidence_v2',
                   review_scope='Approved records may reference pending entities and pending rule binding suggestions; each status remains explicit.',
                   source_metadata=document['metadata'],source_text=document['text'],
                   offset_reference='Unicode code points in linearized document text; half-open intervals',exported_at=now())
    return payload
