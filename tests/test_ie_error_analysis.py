from scripts.ie_error_analysis import categories

def test_role_diagnostic_distinguishes_missing_entities_from_wrong_role():
    rows=[{'text':'片甲人乙地丙','source_file':'test','source_line':1,
           'triples':[[0,2,'作品',0,2,4,'人物'],[0,2,'作品',2,4,6,'地点']]}]
    errors=[{'source_file':'test','source_line':1,'missing':[['片甲','作品',0,'人乙','人物'],['片甲','作品',2,'地丙','地点']],
             'extra':[['片甲','作品',1,'人乙','人物']],
             'predicted_entities':[[0,2,'作品'],[2,4,'人物']]}]
    schemas={'schemas':[{'label':'导演'},{'label':'主演'},{'label':'地点'}]}
    result=categories(rows,errors,schemas,exclude_first=0)
    assert result['counts']=={'missing_despite_both_typed_surfaces_detected':1,
                              'missing_with_participant_surface_or_type_absent':1,
                              'missing_with_wrong_role_on_same_typed_pair':1,'extra_wrong_role_on_gold_typed_pair':1}
    assert result['role_confusions']==[{'gold':'导演','predicted':'主演','count':1}]
