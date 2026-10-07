"""Portable integrity check; does not need raw datasets or model checkpoints."""
from pathlib import Path
import hashlib
import json


def main():
    root=Path(__file__).resolve().parents[1]/'docs/medical-results'
    fingerprints=json.loads((root/'fingerprints.json').read_text(encoding='utf-8'))
    for name,expected in fingerprints.items():
        if Path(name).name!=name:raise ValueError('Invalid snapshot path')
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Snapshot hash mismatch: '+name)
    print(f'Verified {len(fingerprints)} medical report snapshots; this checks integrity, not model quality.')


if __name__=='__main__':main()
