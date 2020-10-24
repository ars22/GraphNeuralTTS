"""
Borrows most of the code from preprocess.py, with additional functionality 
to create a vocab based on the HRGs.
"""

# To import from src
import sys
sys.path.insert(0, '.')

from data.preprocess import preprocess, process_utterance
from pathlib import Path
from collections import Counter, defaultdict
from multiprocessing import cpu_count
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from src.utils import AudioProcessor
import os
from copy import deepcopy
import numpy as np
import argparse
import yaml
from tqdm import tqdm
import json


def preprocess_hrg(args, only_vocab_creation):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    preprocess(args)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess ljspeech dataset')
    parser.add_argument('--data-dir', type=str,
                        help='Directory to raw dataset')
    parser.add_argument('--old-meta', type=str,
                        help='previous old meta file', required=False)
    parser.add_argument('--n-jobs', default=cpu_count(), type=int,
                        help='Number of jobs used for feature extraction', required=False)
    parser.add_argument('--ratio-test', default=0.1, 
                        type=float, help='ratio of testing examples', required=False)
    parser.add_argument('--config', type=str,
                        help='configure file', required=True)
    args = parser.parse_args()
    preprocess_hrg(args)
