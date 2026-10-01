"""Context and signed-position relation representation on the frozen IE pipeline."""
from pathlib import Path
import torch
from torch import nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from safetensors.torch import load_file
from .data import read_json
from .ie.model import Extractor
from .ie.predict import Predictor

FEATURE_RECIPE={'window':4,'distance_edges':[-64,-32,-16,-8,-4,-2,-1,0,1,2,4,8,16,32,64],
                'distance_dimensions':16,'features':['subject','object','between','subject_left','subject_right','object_left','object_right','cls'],
                'mask':'local means exclude special and padding tokens','pair_hidden':256}

class ContextExtractor(Extractor):
    def __init__(self,encoder,entity_types,relations,architecture='pipeline',use_tail=True):
        if architecture!='pipeline':raise ValueError('Context extractor requires pipeline architecture')
        super().__init__(encoder,entity_types,relations,architecture,use_tail)
        self.distance=nn.Embedding(16,16)
        self.register_buffer('distance_edges',torch.tensor(FEATURE_RECIPE['distance_edges']),persistent=False)
        self.pair=nn.Sequential(nn.Linear(8*encoder.config.hidden_size+16,256),nn.ReLU(),nn.Dropout(.1),nn.Linear(256,relations))

    def forward(self,inputs,content_mask,pairs=None):
        self._content_mask=content_mask
        return super().forward(inputs,content_mask,pairs)

    def pair_features(self,hidden,pairs,content_mask=None):
        mask=content_mask if content_mask is not None else self._content_mask
        prefix=torch.cat((torch.zeros_like(hidden[:,:1]),(hidden*mask.unsqueeze(-1)).cumsum(1)),1)
        counts=torch.cat((torch.zeros_like(mask[:,:1],dtype=torch.long),mask.long().cumsum(1)),1)
        index=torch.arange(hidden.shape[0],device=hidden.device)[:,None]
        length=hidden.shape[1]
        def pool(a,b):
            a=a.clamp(0,length);b=b.clamp(0,length)
            return (prefix[index,b]-prefix[index,a])/(counts[index,b]-counts[index,a]).clamp_min(1).unsqueeze(-1)
        sh,st,oh,ot=pairs.unbind(-1)
        left=torch.where(st<oh,st+1,torch.where(ot<sh,ot+1,sh))
        right=torch.where(st<oh,oh,torch.where(ot<sh,sh,sh))
        features=[pool(sh,st+1),pool(oh,ot+1),pool(left,right),pool(sh-4,sh),pool(st+1,st+5),
                  pool(oh-4,oh),pool(ot+1,ot+5),hidden[:,0,None].expand(*sh.shape,hidden.shape[-1])]
        bucket=torch.bucketize((oh-sh).contiguous(),self.distance_edges)
        return torch.cat(features+[self.distance(bucket)],-1)

    def pair_scores(self,hidden,pairs):
        return self.pair(self.pair_features(hidden,pairs)).float()

def fit_context(run,device=None):
    # Explicit, process-local model factory; frozen trainer/optimizer/loss are unchanged.
    from .ie import train
    original=train.Extractor
    try:
        train.Extractor=ContextExtractor
        return train.fit(run,device=device)
    finally:train.Extractor=original

class ContextPredictor(Predictor):
    def __init__(self,run,device='cpu'):
        run=Path(run);meta=read_json(run/'run.json')
        if meta['status']!='completed' or meta.get('relation_features')!='context_position':
            raise ValueError('必须加载已完成的上下文位置模型')
        manifest=read_json(run/'schema.json');config=meta['config']
        encoder=AutoModel.from_config(AutoConfig.from_pretrained(run),add_pooling_layer=False)
        model=ContextExtractor(encoder,len(manifest['entity_types']),len(manifest['schemas']))
        model.load_state_dict(load_file(str(run/'model.safetensors')))
        tokenizer=AutoTokenizer.from_pretrained(run/'tokenizer',local_files_only=True,trust_remote_code=False)
        super().__init__(device=device,model=model,tokenizer=tokenizer,manifest=manifest,config=config)
        if (run/'thresholds.json').exists():self.thresholds=read_json(run/'thresholds.json')

def predictor_for(run,device='cpu'):
    return (ContextPredictor if read_json(Path(run)/'run.json').get('relation_features')=='context_position' else Predictor)(run,device=device)

if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser();parser.add_argument('run');parser.add_argument('text')
    args=parser.parse_args();print(json.dumps(predictor_for(args.run).extract(args.text),ensure_ascii=False,indent=2))
