"""Literal, sentence-local outcome/value suggestions; abstain rather than invent arms.

These rules are not a learned event model. Every suggested field references exact
Unicode offsets, and every suggestion remains pending clinical review.
"""
import re

VERSION = 'literal_bindings_v1'
NUMBER = r'\d+(?:\.\d+)?'
UNIT = r'%|％|个月|天|周|年'
CI = r'95%\s*(?:CI|置信区间)\s*[:：]?\s*(?:'+NUMBER+r'|NA|NR)\s*[-–~]\s*(?:'+NUMBER+r'|NA|NR)'
VALUE = r'(?P<value>'+NUMBER+r')(?:\s*[（(]\s*(?P<ci>'+CI+r')\s*[)）])?\s*(?P<unit>'+UNIT+r')?'
VALUE_RE = re.compile(VALUE, re.I)
ENDPOINT_RE = re.compile(
    r'(?P<median>中位)?(?P<endpoint>总生存期|无进展生存期|客观缓解率|缓解率|'
    r'(?<![A-Za-z])(?:OS|PFS|ORR)(?![A-Za-z])|'
    r'(?:≥\s*3级|3[-–]4级|任意等级|总体)?(?:治疗相关)?(?:不良事件|不良反应|TRAEs?|TEAEs?|AEs?)(?:发生率)?)', re.I)
CONNECTOR = re.compile(r'\s*(?:[（(][A-Za-z ,\-]+[)）]\s*)?(?:(?:分别)?(?:为|达到|达)|发生率(?:为|达到)|[（(])\s*')
COUNT_PERCENT = re.compile('('+NUMBER+r')\s*例\s*[（(]\s*('+NUMBER+r')\s*[%％]\s*[)）]')
PAIR = re.compile(r'\s*(?:和|与|vs\.?|versus)\s*', re.I)
ARM_RE = re.compile(r'[\u4e00-\u9fffA-Za-z0-9-]{1,24}?组')
LEADING_ARM_WORDS = re.compile(r'^(?:而|其中|在|结果显示|研究发现|发现|匹配前|匹配后|接受)+')
MAPPING = {'OS':'OS','总生存期':'OS','PFS':'PFS','无进展生存期':'PFS',
           'ORR':'ORR','客观缓解率':'ORR','缓解率':'ORR'}


def reference(text, base, start, end):
    return {'text':text[start:end], 'start':base+start, 'end':base+end}


def suggestions(sentence):
    text = sentence['text']; base = sentence['start']; output = []; abstentions = []
    endpoints = []
    definitions = list(re.finditer(r'[（(][A-Za-z ,\-]+[)）]',text))
    for endpoint in ENDPOINT_RE.finditer(text):
        if any(m.start()<endpoint.start()<m.end() for m in definitions):continue
        endpoints.append(endpoint)
    if sentence['kind'] == 'table_row':
        return [], [{'reason':'table_headers_not_bound'}] if endpoints else []
    for index, endpoint in enumerate(endpoints):
        stop = endpoints[index+1].start() if index+1<len(endpoints) else len(text)
        tail = text[endpoint.end():stop]
        connector = CONNECTOR.match(tail)
        if not connector:
            abstentions.append({'endpoint':reference(text,base,endpoint.start(),endpoint.end()),
                                'reason':'no_explicit_value_connector'}); continue
        start = endpoint.end()+connector.end()
        canonical = MAPPING.get(endpoint['endpoint'].upper(),MAPPING.get(endpoint['endpoint'],endpoint['endpoint']))
        count = COUNT_PERCENT.match(text,start,stop) if canonical not in ('OS','PFS','ORR') else None
        if count:start=count.start(2)
        first = VALUE_RE.match(text,start,stop)
        if not first:
            abstentions.append({'endpoint':reference(text,base,endpoint.start(),endpoint.end()),
                                'reason':'non_numeric_or_unreached_endpoint'}); continue
        values = [first]; join = PAIR.match(text,first.end(),stop)
        if join:
            second = VALUE_RE.match(text,join.end(),stop)
            if second: values.append(second)
        # A unit must be explicit on at least the final parallel value; shared
        # units are allowed only in a single literal two-value sequence.
        units = [v['unit'] for v in values]
        unit = units[-1]
        if not unit or any(u and u!=unit for u in units):
            abstentions.append({'endpoint':reference(text,base,endpoint.start(),endpoint.end()),
                                'reason':'missing_or_mixed_units'}); continue
        percentage = unit in ('%','％')
        if (canonical in ('OS','PFS') and percentage) or (canonical not in ('OS','PFS') and not percentage):
            abstentions.append({'endpoint':reference(text,base,endpoint.start(),endpoint.end()),
                                'reason':'endpoint_unit_mismatch'}); continue
        if len(values)==2 and PAIR.match(text,values[-1].end(),stop):
            abstentions.append({'endpoint':reference(text,base,endpoint.start(),endpoint.end()),
                                'reason':'more_than_two_parallel_values'}); continue
        # Limit explicit arm attribution to the same comma/semicolon clause.
        # Never inherit an arm from earlier endpoints or a previous sentence.
        clause_start=max(text.rfind(c,0,endpoint.start()) for c in ('，',',','；',';','。'))+1
        prefix = text[clause_start:endpoint.start()]
        arms=[]
        for a in ARM_RE.finditer(prefix):
            cleaned=LEADING_ARM_WORDS.sub('',a.group()).lstrip('和与及')
            if cleaned in ('两组','二组','各组','亚组','分组'): continue
            arms.append(reference(text,base,clause_start+a.end()-len(cleaned),clause_start+a.end()))
        assign = len(arms)==len(values) and (len(values)==1 or '分别' in connector.group())
        if len(values)==2 and assign:
            between=text[arms[0]['end']-base:arms[1]['start']-base]
            assign=bool(re.fullmatch(r'\s*(?:和|与|及)\s*',between))
        for position, v in enumerate(values):
            item={'endpoint':canonical,'statistic':'median' if endpoint['median'] else 'reported',
                  'endpoint_reference':reference(text,base,endpoint.start(),endpoint.end()),
                  'value':float(v['value']),'unit':'%' if unit=='％' else unit,
                  'value_reference':reference(text,base,v.start(),v.end()),
                  'unit_inherited':v['unit'] is None,'arm':arms[position] if assign else None,
                  'arm_binding':'explicit_same_clause' if assign else 'unassigned',
                  'method':VERSION,'status':'pending','parallel_position':position if len(values)==2 else None}
            if v['ci']:
                from .evidence import numeric_atom
                item['confidence_interval']=numeric_atom(v['ci'])
            if count:
                item['count']=int(count[1]) if float(count[1]).is_integer() else float(count[1])
                item['count_reference']=reference(text,base,count.start(1),count.end(1))
            output.append(item)
    return output, abstentions
