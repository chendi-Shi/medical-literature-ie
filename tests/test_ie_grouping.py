from nlp_lab.ie_api import group_relations

def test_same_relation_multiple_mentions_grouped_without_alias_merging():
    entities=[{'id':'a','type':'人物'},{'id':'b','type':'图书作品'},{'id':'c','type':'图书作品'}]
    relation={'subject':'b','object':'a','subject_text':'三体','object_text':'刘慈欣','predicate':'作者','slot':'@value','label':'作者','confidence':.8}
    result={'entities':entities,'relations':[relation,{**relation,'subject':'c','confidence':.9},
                                            {**relation,'subject_text':'The Three-Body Problem'}]}
    groups=group_relations(result)
    assert len(groups)==2 and len(groups[0]['mention_links'])==2 and groups[0]['confidence']==.9
    assert groups[0]['subject_type']=='图书作品' and groups[0]['object_type']=='人物'
