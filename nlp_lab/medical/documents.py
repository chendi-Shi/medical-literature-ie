"""Lossless evidence offsets in an explicitly linearized JATS/text representation."""
from pathlib import Path
import hashlib
import re
import xml.etree.ElementTree as ET


def content(element):
    return ''.join(element.itertext()).strip()


def parse_document(raw, format='text', title='', source='user-provided'):
    if not isinstance(raw,str) or not raw.strip() or len(raw)>500_000:
        raise ValueError('Document must contain 1–500,000 characters')
    blocks=[]; metadata={'source':source,'title':title,'format':format}
    def add(text,section,kind='paragraph',locator=''):
        if text.strip(): blocks.append({'text':text.strip(),'section':section,'kind':kind,'locator':locator})
    if format=='text':
        section='正文'
        for i,line in enumerate(raw.splitlines()):
            if re.fullmatch(r'\s*(?:#{1,6}\s*)?(?:摘要|背景|目的|方法|结果|结论|讨论|不良反应|安全性)\s*',line): section=line.strip().lstrip('#').strip()
            add(line,section,locator=f'line:{i+1}')
    elif format=='jats':
        # Standard JATS XML declares an external DTD. Strip that declaration without
        # loading it; internal subsets/entities remain rejected, as does malformed DTD syntax.
        safe_xml=re.sub(r'<!DOCTYPE\s+article\s+(?:PUBLIC\s+"[^"]*"\s+"[^"]*"|SYSTEM\s+"[^"]*")\s*>','',raw)
        if re.search(r'<!\s*(?:DOCTYPE|ENTITY)',safe_xml,re.I): raise ValueError('Internal/malformed DTD and XML entities are not accepted')
        root=ET.fromstring(safe_xml)
        if root.tag!='article': raise ValueError('Expected a JATS article XML root')
        titles=root.findall('./front/article-meta/title-group/article-title')
        metadata['title']=title or (content(titles[0]) if titles else '')
        metadata['article_ids']={x.get('pub-id-type','unknown'):content(x) for x in root.findall('./front/article-meta/article-id')}
        metadata['licenses']=[{'text':content(x),'href':x.get('{http://www.w3.org/1999/xlink}href')} for x in root.findall('.//permissions/license')]
        def visit(node,section,path):
            for i,child in enumerate(node):
                loc=f'{path}/{child.tag}[{i+1}]'
                if child.tag=='sec':
                    heading=child.find('title');visit(child,section+' / '+(content(heading) if heading is not None else 'section'),loc)
                elif child.tag=='p': add(content(child),section,locator=loc)
                elif child.tag=='table-wrap':
                    caption=child.find('caption')
                    if caption is not None:add(content(caption),section,'table_caption',loc+'/caption')
                    for j,tr in enumerate(child.findall('.//tr')):
                        cells=[content(c) for c in tr if c.tag in ('td','th')]
                        add(' | '.join(cells),section,'table_row',loc+f'/tr[{j+1}]')
                elif child.tag in ('list','list-item','boxed-text','disp-quote'): visit(child,section,loc)
        meta=root.find('./front/article-meta')
        if meta is not None:
            for i,abstract in enumerate(list(meta.findall('abstract'))+list(meta.findall('trans-abstract'))):
                section='摘要 / '+abstract.get('{http://www.w3.org/XML/1998/namespace}lang','unspecified')
                visit(abstract,section,f'article-meta/abstract[{i+1}]')
                if not abstract.findall('p') and not abstract.findall('sec'):add(content(abstract),section,locator=f'abstract[{i+1}]')
        body=root.find('body')
        if body is not None:visit(body,'正文','body')
        # References are intentionally excluded from the evidence text.
        metadata['representation']='JATS paragraphs/table rows linearized with newline separators; offsets reference this text, not XML bytes'
    else:raise ValueError('Supported formats: text, jats')
    if not blocks:raise ValueError('No extractable paragraphs')
    parts=[];offset=0
    for i,block in enumerate(blocks):
        block.update(id=f'b{i}',start=offset,end=offset+len(block['text']))
        parts.append(block['text']);offset=block['end']+1
    text='\n'.join(parts)
    return {'text':text,'blocks':blocks,'metadata':metadata,'sha256':hashlib.sha256(text.encode()).hexdigest(),
            'raw_sha256':hashlib.sha256(raw.encode()).hexdigest()}


def sentences(document):
    """Chinese delimiters plus English sentence periods; preserve decimal numbers."""
    output=[]
    for block in document['blocks']:
        start=0;text=block['text']
        cuts=[m.end() for m in re.finditer(r'[。！？!?]+|\.(?=\s+[A-Z])',text)]+[len(text)]
        for stop in sorted(set(cuts)):
            piece=text[start:stop]; leading=len(piece)-len(piece.lstrip());trailing=len(piece.rstrip())
            a=block['start']+start+leading;b=block['start']+start+trailing
            if a<b:output.append({'start':a,'end':b,'text':document['text'][a:b],'block':block['id'],
                                  'section':block['section'],'locator':block['locator'],'kind':block['kind']})
            start=stop
    return output
