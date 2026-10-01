"""Typed-span GPLinker and a NER -> pair-classifier decoding baseline."""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F


def rotate_half(x):
    paired=x.reshape(*x.shape[:-1],-1,2)
    return torch.stack((-paired[...,1],paired[...,0]),dim=-1).flatten(-2)


class EfficientPointer(nn.Module):
    def __init__(self, hidden, heads, size=64, rope=True, triangular=True):
        super().__init__()
        self.projection=nn.Linear(hidden,2*size)
        self.bias=nn.Linear(hidden,2*heads)
        self.size,self.heads,self.rope,self.triangular=size,heads,rope,triangular

    def forward(self, hidden, content_mask):
        q,k=self.projection(hidden).chunk(2,dim=-1)
        if self.rope:
            frequencies=torch.exp(-math.log(10000)*torch.arange(0,self.size,2,device=hidden.device).float()/self.size)
            angles=torch.arange(hidden.shape[1],device=hidden.device).float()[:,None]*frequencies[None,:]
            cosine=angles.cos().repeat_interleave(2,-1).to(q.dtype)
            sine=angles.sin().repeat_interleave(2,-1).to(q.dtype)
            q=q*cosine+rotate_half(q)*sine
            k=k*cosine+rotate_half(k)*sine
        scores=torch.einsum('bld,bmd->blm',q,k).float()/math.sqrt(self.size)
        biases=self.bias(hidden).transpose(1,2).float()
        result=scores[:,None]+biases[:,:self.heads,:,None]+biases[:,self.heads:,None,:]
        valid=content_mask[:,None,:,None]&content_mask[:,None,None,:]
        if self.triangular:
            valid=valid&torch.ones(hidden.shape[1],hidden.shape[1],device=hidden.device,dtype=torch.bool).triu()
        return result.masked_fill(~valid,-10000.)


def pointer_loss(logits, target):
    """Balanced multilabel categorical loss, mean examples and label heads."""
    flat=logits.float().flatten(-2)
    truth=target.flatten(-2).bool()
    zero=torch.zeros_like(flat[...,:1])
    negative=torch.cat((flat.masked_fill(truth,-10000.),zero),-1)
    positive=torch.cat((-flat.masked_fill(~truth,10000.),zero),-1)
    return (torch.logsumexp(negative,-1)+torch.logsumexp(positive,-1)).mean()


class Extractor(nn.Module):
    def __init__(self, encoder, entity_types, relations, architecture='joint', use_tail=True):
        super().__init__()
        self.encoder=encoder
        self.architecture,self.use_tail=architecture,use_tail
        hidden=encoder.config.hidden_size
        self.dropout=nn.Dropout(.1)
        self.entity=EfficientPointer(hidden,entity_types)
        if architecture=='joint':
            self.head=EfficientPointer(hidden,relations,rope=False,triangular=False)
            self.tail=EfficientPointer(hidden,relations,rope=False,triangular=False)
        elif architecture=='pipeline':
            self.pair=nn.Sequential(nn.Linear(3*hidden,256),nn.ReLU(),nn.Dropout(.1),nn.Linear(256,relations))
        else:
            raise ValueError('未知信息抽取架构')

    def pair_scores(self, hidden, pairs):
        prefix=torch.cat((torch.zeros_like(hidden[:,:1]),hidden.cumsum(1)),dim=1)
        index=torch.arange(hidden.shape[0],device=hidden.device)[:,None]
        sh,st,oh,ot=pairs.unbind(-1)
        subject=(prefix[index,st+1]-prefix[index,sh])/(st-sh+1).unsqueeze(-1)
        obj=(prefix[index,ot+1]-prefix[index,oh])/(ot-oh+1).unsqueeze(-1)
        cls=hidden[:,0,None].expand_as(subject)
        return self.pair(torch.cat((subject,obj,cls),dim=-1)).float()

    def forward(self, inputs, content_mask, pairs=None):
        hidden=self.dropout(self.encoder(**inputs).last_hidden_state)
        result={'entity':self.entity(hidden,content_mask)}
        if self.architecture=='joint':
            result.update(head=self.head(hidden,content_mask),tail=self.tail(hidden,content_mask))
        else:
            result['hidden']=hidden
            if pairs is not None:
                result['pair']=self.pair_scores(hidden,pairs)
        return result


def joint_objective(outputs, targets, use_tail=True):
    components={name:pointer_loss(outputs[name],targets[name]) for name in ('entity','head','tail') if name!='tail' or use_tail}
    return sum(components.values()),components


def pipeline_objective(outputs, targets):
    entity=pointer_loss(outputs['entity'],targets['entity'])
    loss=F.binary_cross_entropy_with_logits(outputs['pair'],targets['pair'],reduction='none',
                                           pos_weight=outputs['pair'].new_tensor(3.))
    mask=targets['pair_mask']
    relation=(loss*mask).sum()/mask.sum().clamp_min(1)
    return entity+2*relation,{'entity':entity,'pair':relation}
