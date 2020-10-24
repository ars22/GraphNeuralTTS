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


def preprocess(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # # Make directory if not exist
    os.makedirs(config['solver']['data_dir'], exist_ok=True)
    print('')
    print('[INFO] Root directory:', args.audio_dir)

    AP = AudioProcessor(**config['audio'])
    executor = ProcessPoolExecutor(max_workers=args.n_jobs)
    fid = []
    text = []
    wav = []
    futures = []
    with open(args.metadata, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            fpath = os.path.join(args.audio_dir, '%s.wav' % parts[0])
            text = "|".join(parts[2:])              # Merge additional info with text (separated by |)
            job = executor.submit(partial(process_utterance, fpath, text, config['solver']['data_dir'], AP))
            futures += [job]

    print('[INFO] Preprocessing', end=' => ')
    print(len(futures), 'audio files found')
    results = [future.result() for future in tqdm(futures)]
    fpath_meta = os.path.join(config['solver']['data_dir'], 'meta_all.txt')
    with open(fpath_meta, 'w') as f:
        for x in results:
            s = map(lambda x: str(x), x)
            f.write('|'.join(s) + '\n')


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
    parser = argparse.ArgumentParser(description='Preprocess dataset')
    parser.add_argument('--audio-dir', type=str, help='Directory to raw audio')
    parser.add_argument('--metadata', type=str, help='meta file', required=True)
    parser.add_argument('--n-jobs', default=cpu_count(), type=int, help='Number of jobs used for feature extraction', required=False)
    parser.add_argument('--config', type=str, help='configure file', required=True)
    args = parser.parse_args()
    preprocess(args)

