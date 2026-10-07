import torch
from transformers import BertConfig,BertModel,AutoTokenizer
from nlp_lab.ie.data import MODEL
from nlp_lab.medical.ner import MedicalNER,examples,collate,scores


def test_nested_medical_supervision_gradients_and_offset_metrics():
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    row={'id':'nested','text':'😀肺癌患者咳嗽。','entities':[(1,3,'dis'),(1,5,'sym'),(5,7,'sym')]}
    batch=examples([row],tokenizer);inputs,mask,target=collate(batch)
    assert target.sum()==3
    encoder=BertModel(BertConfig(vocab_size=tokenizer.vocab_size,hidden_size=32,num_hidden_layers=1,num_attention_heads=4,intermediate_size=64),add_pooling_layer=False)
    model=MedicalNER(encoder)
    from nlp_lab.ie.model import pointer_loss
    loss=pointer_loss(model(inputs,mask),target);loss.backward()
    assert torch.isfinite(loss) and model.pointer.projection.weight.grad.abs().sum()>0
    metric,errors=scores([row],[[(1,3,'dis',1.),(1,5,'sym',1.),(5,7,'sym',1.)]])
    assert metric['micro']['f1']==1 and not errors
    metric,_=scores([row],[[(1,3,'sym',1.)]])
    assert metric['micro']['f1']==0


def test_long_document_windows_preserve_nested_span_coverage():
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    text='患者。'*200+'肺癌。'+'患者。'*200
    a=text.index('肺癌');row={'id':'long','text':text,'entities':[(a,a+2,'dis')]}
    encoded=examples([row],tokenizer)
    assert len(encoded)>1 and all(len(x[0]['input_ids'])<=384 for x in encoded)
    assert sum(len(x[2]) for x in encoded)>=1
