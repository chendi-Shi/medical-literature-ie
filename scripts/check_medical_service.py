"""Real HTTP/source/review/export checks, without converting examples into benchmark scores."""
from pathlib import Path
import json
import time
import urllib.request


def call(path,body=None):
    req=urllib.request.Request('http://127.0.0.1:8778'+path,
        data=json.dumps(body,ensure_ascii=False).encode() if body is not None else None,
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=30) as response:return json.load(response)


def main():
    overview=call('/api/medical');assert overview['status']['phase']=='completed'
    expected_models={**overview['report']['weights_sha256'],'relations':overview['active_relation']['sha256']}
    checks=[]
    for source in overview['documents']:
        document=call('/api/medical/documents/'+source['id'])
        # Prefer an already completed extraction using the current pipeline source recipe.
        from nlp_lab.data import digest
        fingerprints={name:digest(Path('nlp_lab/medical')/name) for name in ('documents.py','evidence.py','bindings.py','context.py','runtime.py','pipeline.py','ner.py')}
        found=None
        for item in document['extractions']:
            extraction=call('/api/medical/extractions/'+item['id'])
            if extraction.get('pipeline_source_sha256')==fingerprints and extraction.get('models')==expected_models:found=extraction;break
        if found is None:
            job=call('/api/medical/documents/'+source['id']+'/extract',{})
            deadline=time.monotonic()+50
            while job['status'] in ('queued','running') and time.monotonic()<deadline:
                time.sleep(.5);job=call('/api/medical/jobs/'+job['id'])
            if job['status']!='completed':raise ValueError('Document job incomplete: '+str(job))
            found=call('/api/medical/extractions/'+job['extraction_id'])
        text=document['text'];entities={e['id']:e for e in found['entities']}
        assert found['document_sha256']==document['sha256']
        for entity in entities.values():
            assert text[entity['start']:entity['end']]==entity['text']
            for modifier in entity.get('context',{}).get('modifiers',[])+entity.get('context',{}).get('blocked_modifiers',[]):
                for key in ('cue','scope'):
                    ref=modifier[key];assert text[ref['start']:ref['end']]==ref['text']
        for definition in found['alias_definitions']:
            assert definition['alias'] not in {'HR','OR','RR','CI','P'}
            assert text[definition['start']:definition['end']]==definition['evidence']
        for r in found['records']:
            evidence=r['evidence'];assert text[evidence['start']:evidence['end']]==evidence['text']
            assert all(e in entities for e in r['mentions'])
            for cue in r.get('provenance_context',{}).get('cues',[]):assert text[cue['start']:cue['end']]==cue['text']
            for number in r['numbers']:assert text[number['start']:number['end']]==number['text']
            for binding in r.get('binding_suggestions',[]):
                assert binding['status']=='pending'
                for ref in (binding['endpoint_reference'],binding['value_reference'],binding['arm']):
                    if ref:assert text[ref['start']:ref['end']]==ref['text']
        for r in found['relations']:
            assert r['subject'] in entities and r['object'] in entities
            assert text[r['evidence']['start']:r['evidence']['end']]==r['evidence']['text']
        checks.append({'document_id':source['id'],'title':source['metadata']['title'],'characters':len(text),
            'models':found['models'],'relation_version':overview['active_relation']['version'],
            'extraction_id':found['id'],'entities':len(entities),'concepts':len(found['concepts']),
            'relations':len(found['relations']),'candidate_records':len(found['records']),'model_windows':found['model_windows'],
            'binding_suggestions':sum(len(r.get('binding_suggestions',[])) for r in found['records']),
            'explicit_arm_suggestions':sum(b['arm'] is not None for r in found['records'] for b in r.get('binding_suggestions',[])),
            'context_modified_entities':sum(bool(e.get('context',{}).get('modifiers')) for e in entities.values()),
            'context_blocked_entities':sum(bool(e.get('context',{}).get('blocked_modifiers')) for e in entities.values()),
            'cited_or_mixed_records':sum(r.get('provenance_context',{}).get('scope') in ('cited_study_cues','mixed_cues') for r in found['records']),
            'negative_efficacy_records':sum(bool(r['qualifiers']['negation']) and r['category']=='efficacy' for r in found['records']),
            'all_offsets_and_references_valid':True,'pipeline_source_sha256':fingerprints})
    assert call('/api/medical/search?q='+urllib.parse.quote('肺癌'))
    out=Path('workspace/medical/service_check.json');out.write_text(json.dumps({'documents':checks,'search_verified':True,
        'scope':'Functional checks on public papers; not independently expert-annotated extraction accuracy.'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(checks,ensure_ascii=False))


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    main()
