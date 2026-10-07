"""Explicitly download the pinned encoder; training itself remains local-only."""
from pathlib import Path
from huggingface_hub import model_info,snapshot_download

if __name__=='__main__':
    model='BAAI/bge-small-zh-v1.5';revision='7999e1d3359715c523056ef9478215996d62a620'
    # Frozen trainers resolve the local main alias. Check it before downloading
    # so a fresh cache has that alias without silently using a changed encoder.
    if model_info(model,revision='main').sha!=revision:
        raise RuntimeError('Encoder main changed; use a separately versioned training protocol instead of overwriting the frozen experiment.')
    path=snapshot_download(model,revision='main',
        allow_patterns=['*.json','*.txt','*.safetensors','pytorch_model.bin'])
    if Path(path).name!=revision:raise RuntimeError('Downloaded encoder revision does not match protocol')
    print(path)
