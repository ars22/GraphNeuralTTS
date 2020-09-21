import os
import numpy as np
import argparse
import yaml
from tqdm import tqdm
import random

import sys
# To import from src
sys.path.insert(0, '.')

from src.utils import AudioProcessor
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path



def split_and_save(meta_all_path, args):
    meta_dir = os.path.dirname(os.path.realpath(meta_all_path))
    meta_tr_path = os.path.join(meta_dir, 'meta_train.txt')
    meta_te_path = os.path.join(meta_dir, 'meta_test.txt')
    with open(meta_all_path) as f:
        meta_all = f.readlines()
        meta_tr = []
        meta_te = []

    n_meta = len(meta_all)
    n_test = int(args.ratio_test * n_meta)
    indice_te = random.sample(range(n_meta), n_test)

    for idx, line in enumerate(meta_all):
        if idx in indice_te:
            meta_te.append(line)
        else:
            meta_tr.append(line)

    with open(meta_tr_path, 'w') as ftr:
        ftr.write(''.join(meta_tr))
    with open(meta_te_path, 'w') as fte:
        fte.write(''.join(meta_te))

def preprocess(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # # Make directory if not exist
    os.makedirs(config['solver']['data_dir'], exist_ok=True)
    print('')
    print('[INFO] Root directory:', args.data_dir)

    AP = AudioProcessor(**config['audio'])
    executor = ProcessPoolExecutor(max_workers=args.n_jobs)
    fid = []
    text = []
    wav = []
    futures = []
    with open(args.old_meta, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            fpath = os.path.join(args.data_dir, '%s.wav' % parts[0])
            text = parts[2]
            job = executor.submit(partial(process_utterance, fpath, text, config['solver']['data_dir'], AP))
            futures += [job]

    print('[INFO] Preprocessing', end=' => ')
    print(len(futures), 'audio files found')
    results = [future.result() for future in tqdm(futures)]
    fpath_meta = os.path.join(config['solver']['data_dir'], 'ljspeech_meta.txt')
    with open(fpath_meta, 'w') as f:
        for x in results:
            s = map(lambda x: str(x), x)
            f.write('|'.join(s) + '\n')

    print(f"Creating Train/Test splits with ratio: {1.0-args.ratio_test}/{args.ratio_test}")
    split_and_save(fpath_meta, args)


def process_utterance(fpath, text, output_dir,
        audio_processor, store_mel=True, store_linear=True):
    wav = audio_processor.load_wav(fpath)
    mel = audio_processor.melspectrogram(wav).astype(np.float32).T
    linear = audio_processor.spectrogram(wav).astype(np.float32).T
    n_frames = linear.shape[0]
    fid = fpath.split('/')[-1].split('.')[0]
    fpath_mel = fid + '-mel.npy'
    fpath_linear = fid + '-linear.npy'
    if store_mel:
        np.save(os.path.join(output_dir, fpath_mel), mel, allow_pickle=False)
    if store_linear:
        np.save(os.path.join(output_dir, fpath_linear), linear, allow_pickle=False)
    return fpath_mel, fpath_linear, n_frames, text



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess ljspeech dataset')
    parser.add_argument('--data-dir', type=str, help='Directory to raw dataset')
    parser.add_argument('--old-meta', type=str, help='previous old meta file', required=True)
    parser.add_argument('--n-jobs', default=cpu_count(), type=int, help='Number of jobs used for feature extraction', required=False)
    parser.add_argument('--config', type=str, help='configure file', required=True)
    parser.add_argument('--ratio-test', default=0.1, 
                        type=float, help='ratio of testing examples', required=False)
    args = parser.parse_args()
    preprocess(args)

