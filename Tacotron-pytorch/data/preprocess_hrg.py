"""
Borrows most of the code from preprocess.py, with additional functionality 
to create a vocab based on the HRGs.
"""
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
import sys
# To import from src
sys.path.insert(0, '.')


def preprocess_hrg(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # preprocess(args)
    vocab = Vocab(config['solver']['meta_path']['train'])
    meta_path = config['solver']['data_dir']
    with open(f"{meta_path}/tok2id.json", "w") as f:
        json.dump(vocab.tok2id, f)
    with open(f"{meta_path}/id2tok.json", "w") as f:
        json.dump(vocab.id2tok, f)
    with open(f"{meta_path}/tok2count.json", "w") as f:
        json.dump(vocab.tok2count, f)


class Vocab:

    def __init__(self, meta_path):
        # Load meta
        # ---------
        # text: texts
        # mel : filenames of mel-spectrogram
        # spec: filenames of (linear) spectrogram
        #

        self.hrgs = []
        with open(meta_path) as f:
            for line in f.readlines():
                # If there is '\n' in text, it will be discarded when calling symbols.txt2seq
                fmel, fspec, n_frames, hrg = line.split('|')
                self.hrgs.append(json.loads(hrg))
        self.init_vocab()

    def init_vocab(self):
        tokens = Counter(list(self.get_tokens_from_hrg()))
        self.tok2count = {tok: count for tok, count in tokens.items() if count > 1}
        tokens = [w[0] for w in tokens.items() if w[1] > 1]
        tokens.extend([str(i) for i in range(20)])  # position
        tokens.extend(["<W>", "<SYLL>", "<UNK>"])
        self.tok2id = {w: i for i, w in enumerate(tokens)}
        self.id2tok = {i: w for w, i in self.tok2id.items()}
        
        self.n_vocab = len(self.tok2id)


    def get_tokens_from_hrg(self):
        def _get_tokens_from_word_rep(word_rep):
            tokens = []
            tokens.append(word_rep["word"])
            for daughter in word_rep["daughters"]:
                syllnode = []
                for syll in daughter:
                    tokens.append(syll["syll"])
                    syllnode.append(syll["syll"])
                tokens.append("".join(syllnode))
            return tokens
        
        tokens = []
        for hrg in tqdm(self.hrgs, total=len(self.hrgs), desc="Parsing HRGs"):
            
            
            for word_rep in hrg:
                tokens.extend(_get_tokens_from_word_rep(word_rep))
        return tokens


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess ljspeech dataset')
    parser.add_argument('--data-dir', type=str,
                        help='Directory to raw dataset')
    parser.add_argument('--output-dir', default='training_data/',
                        type=str, help='Directory to store output', required=False)
    parser.add_argument('--old-meta', type=str,
                        help='previous old meta file', required=True)
    parser.add_argument('--n-jobs', default=cpu_count(), type=int,
                        help='Number of jobs used for feature extraction', required=False)
    parser.add_argument('--config', type=str,
                        help='configure file', required=True)
    args = parser.parse_args()
    preprocess_hrg(args)
