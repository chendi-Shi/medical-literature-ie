"""Download two public CC-BY JATS papers with retries; import exact source bytes."""
from pathlib import Path
import json
import sys
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nlp_lab.data import digest,save_json
from nlp_lab.medical.documents import parse_document
from nlp_lab.medical.store import Store


def main():
    root=Path('workspace/medical');sources=root/'sources';sources.mkdir(parents=True,exist_ok=True)
    store=Store(root/'literature.sqlite3');records=[]
    for pmc in ['PMC10918247','PMC11534547']:
        url=f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextXML';path=sources/(pmc+'.xml')
        if not path.exists():
            for attempt in range(3):
                try:
                    data=urllib.request.urlopen(url,timeout=30).read();path.write_bytes(data);break
                except Exception:
                    if attempt==2:raise
        document=store.add_document(parse_document(path.read_text(encoding='utf-8'),'jats',source=f'https://pmc.ncbi.nlm.nih.gov/articles/{pmc}/'))
        records.append({'pmc':pmc,'url':url,'source_sha256':digest(path),'document_id':document['id'],
                        'text_sha256':document['sha256'],'characters':len(document['text']),
                        'title':document['metadata']['title'],'licenses':document['metadata']['licenses']})
    save_json(root/'paper_sources.json',records);print(json.dumps(records,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
