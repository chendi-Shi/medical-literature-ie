import argparse
from pathlib import Path
from nlp_lab.ie.suite import suite

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',default='workspace')
    args=parser.parse_args();suite(Path(args.workspace))
