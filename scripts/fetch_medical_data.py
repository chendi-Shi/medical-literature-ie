"""Fetch public CBLUE mirrors; enforce the exact locally audited corpus bytes."""
from pathlib import Path
import hashlib
import urllib.request

FILES={
    'CMeEE_train.json':('https://raw.githubusercontent.com/Z-MU-Z/cmeee/master/data/CBLUEDatasets/CMeEE/CMeEE_train.json','984f0bea3932c8c5afd96d9cccac4895b87ec77f57bd3e825d832a17943cd5d3'),
    'CMeEE_dev.json':('https://huggingface.co/datasets/Aunderline/CMeEE/resolve/bea81a21dc077bb21f6df5a0385410f4768b97bd/CMeEE_dev.json','bbe6be016bdb1a418362507024902b70587b951a884cf4a96a6d1246822cd07c'),
    'CMeIE_train.json':('https://raw.githubusercontent.com/yehqi/RelationExtraction/main/CMeIE_train.json','cfa38f1a74c7ad798485043cb11f9ba54a4fb4e013ae7771f0a3b8ab7809984d'),
    'CMeIE_dev.json':('https://raw.githubusercontent.com/yehqi/RelationExtraction/main/CMeIE_dev.json','3ae4dcf92a8741a16591537901860e7e54bc057447dd07712691401fa48b985a'),
    '53_schemas.json':('https://raw.githubusercontent.com/yehqi/RelationExtraction/main/53_schemas.json','d642657dfecf7716bacb0de44afc3fc29ae4cc9428da7c95cd4e9c03e9a95642'),
}


def main():
    root=Path('workspace/medical/sources');root.mkdir(parents=True,exist_ok=True)
    for name,(url,expected) in FILES.items():
        path=root/name
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Existing source checksum mismatch: '+name)
            print(name,'verified');continue
        for attempt in range(3):
            try:
                request=urllib.request.Request(url,headers={'User-Agent':'NLP-Training-Lab/0.4'})
                with urllib.request.urlopen(request,timeout=40) as response:data=response.read(25_000_001)
                if len(data)>25_000_000:raise ValueError('Unexpected source size')
                if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Source mirror changed: '+name)
                path.write_bytes(data);print(name,'downloaded and verified');break
            except Exception:
                if attempt==2:raise


if __name__=='__main__':main()
