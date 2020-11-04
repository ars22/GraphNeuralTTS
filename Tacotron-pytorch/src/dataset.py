import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from .symbols import txt2seq
from functools import partial
from tqdm import tqdm
import json

def getDataLoader(mode, meta_path, data_dir, batch_size, r, n_jobs, use_gpu, **kwargs):
    bs = batch_size
    if mode == 'train':
        shuffle = True
    elif mode == 'test':
        shuffle = False
    else:
        raise NotImplementedError
    
    vocab = kwargs["vocab"] if "vocab" in kwargs else None
    add_info_vocab = kwargs["add_info_vocab"] if "add_info_vocab" in kwargs else None
    add_info_headers = kwargs["add_info_headers"] if "add_info_headers" in kwargs else None

    if len(add_info_headers):
        DS = MyDatasetAddInfo(meta_path, data_dir, add_info_vocab=add_info_vocab, add_info_headers=add_info_headers)
    else:
        DS = MyDataset(meta_path, data_dir)

    DL = DataLoader(
            DS, batch_size=batch_size, shuffle=shuffle, drop_last=False,
            num_workers=n_jobs, collate_fn=partial(collate_fn, r=r), pin_memory=use_gpu)
    return DL


def _pad(seq, max_len):
    seq = np.pad(seq, (0, max_len - len(seq)),
            mode='constant', constant_values=0)
    return seq


def _pad_2d(x, max_len):
    x = np.pad(x, [(0, max_len - len(x)), (0, 0)],
            mode="constant", constant_values=0)
    return x

class VocabAddInfo:
    def __init__(self, id2tok, tok2id):
        self.id2tok = id2tok
        self.tok2id = tok2id
        self.n_vocab = len(self.id2tok)
        assert len(id2tok) == len(tok2id), "Invalid vocab"

    def get_tok2id(self, tok):
        if tok in self.tok2id:
            return self.tok2id[tok]
        return self.tok2id["<UNK>"]

    @staticmethod
    def from_dataset(dataset, entity):
        """Reads and initializes vocab from a directory

        Args:
            dataset ([type]): [list of dictionaries of additional info]
            entity: key for additional info (speaker/accent)

        Returns:
            [type]: [HRGVocab]
        """
        tok2id = {}
        for add_info_dict in dataset:
            if add_info_dict[entity] not in tok2id:
                tok2id[add_info_dict[entity]] = len(tok2id)
        id2tok = {v: k for k,v in tok2id.items()}

        print("Add info:", entity)
        print("id2tok", id2tok)
        print("tok2id", tok2id)
    
        return VocabAddInfo(id2tok=id2tok, tok2id=tok2id)

    def __len__(self):
        return len(self.tok2id)


class MyDataset(Dataset):
    """Dataset
    """
    def __init__(self, meta_path, data_dir):
        # Load meta
        # ---------
        # text: texts
        # mel : filenames of mel-spectrogram
        # spec: filenames of (linear) spectrogram
        #
        meta = {'id': [], 'text':[], 'mel': [], 'spec': []}
        with open(meta_path) as f:
            for line in f.readlines():
                # If there is '\n' in text, it will be discarded when calling symbols.txt2seq
                parts = line.split('|')
                fmel, fspec, n_frames, text = parts[:4]             # Extract only first 4 entries and ignore add info
                meta['id'].append("-".join(fmel.split("-")[:-1]))
                meta['text'].append(text)
                meta['mel'].append(fmel)
                meta['spec'].append(fspec)

        self.X = meta['text']
        self.Y_mel = [os.path.join(data_dir, f) for f in meta['mel']]
        self.Y_spec = [os.path.join(data_dir, f) for f in meta['spec']]
        assert len(self.X) == len(self.Y_mel) == len(self.Y_spec)
        # Text to id sequence
        self.X = [txt2seq(x) for x in self.X]

        # dummy vocab (since char model doesnt need one)
        self.vocab = None
        self.add_info_vocab = None
        print("Char model using text2seq function in symbols. No vocab needed")
        self.meta = meta

    def __getitem__(self, idx):
        item = (self.meta['id'][idx],
                self.X[idx],
                np.load(self.Y_mel[idx]),
                np.load(self.Y_spec[idx]))
        return item

    def __len__(self):
        return len(self.X)

class MyDatasetAddInfo(Dataset):
    """Dataset
    """
    def __init__(self, meta_path, data_dir, add_info_headers, add_info_vocab=None):
        # Load meta
        # ---------
        # text: texts
        # mel : filenames of mel-spectrogram
        # spec: filenames of (linear) spectrogram
        #
        meta = {'id': [], 'text':[], 'mel': [], 'spec': [], 'add_info': []}
        with open(meta_path) as f:
            for line in f.readlines():
                # If there is '\n' in text, it will be discarded when calling symbols.txt2seq
                # Read the file and integrate any additional info with the text itself
                fmel, fspec, n_frames, text, add_info = line.split('|')
                meta['id'].append("-".join(fmel.split("-")[:-1]))
                #meta['id'].append(fmel.split("-")[0])
                meta['text'].append(text)
                meta['mel'].append(fmel)
                meta['spec'].append(fspec)
                meta['add_info'].append(add_info)

        # Separate text and additional info
        self.X = meta['text']
        self.add_info = [ json.loads(t) for t in meta['add_info'] ]
        headers = add_info_headers

        # make vocab for each additional info
        if add_info_vocab is None:
            print("Creating add_info vocab")
            self.add_info_vocab = {}
            for h in headers:
                self.add_info_vocab[h] = VocabAddInfo.from_dataset(self.add_info, h)
        else:
            print("Reusing add_info vocab")
            self.add_info_vocab = add_info_vocab
         
        # Convert to ids
        self.add_info = [ {h:self.add_info_vocab[h].get_tok2id(t[h]) for h in headers} for t in self.add_info ]
        # get max vocab size for all the additional info
        self.n_add_info_vocab = max([self.add_info_vocab[h].n_vocab for h in headers])

        self.Y_mel = [os.path.join(data_dir, f) for f in meta['mel']]
        self.Y_spec = [os.path.join(data_dir, f) for f in meta['spec']]
        self.X = [txt2seq(x) for x in meta['text']]
        assert len(self.X) == len(self.Y_mel) == len(self.Y_spec) == len(self.add_info)

        # dummy vocab (since char model doesnt need one)
        self.vocab = None
        print("Char model using text2seq function in symbols. No vocab needed")
        self.meta = meta
        
    def __getitem__(self, idx):
        item = (self.meta['id'][idx],
                self.X[idx],
                np.load(self.Y_mel[idx]),
                np.load(self.Y_spec[idx]),
                self.add_info[idx])
        return item

    def __len__(self):
        return len(self.X)


def collate_fn(batch, r):
    """Create batch"""
    num_inputs = len(batch[0])
    ids = [x[0] for x in batch]
    input_lengths = [len(x[1]) for x in batch]
    max_input_len = np.max(input_lengths)
    # (r9y9's comment) Add single zeros frame at least, so plus 1
    max_target_len = np.max([len(x[2]) for x in batch]) + 1
    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0

    a = np.array([_pad(x[1], max_input_len) for x in batch], dtype=np.int)
    x_batch = torch.LongTensor(a)

    input_lengths = torch.LongTensor(input_lengths)

    b = np.array([_pad_2d(x[2], max_target_len) for x in batch],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    c = np.array([_pad_2d(x[3], max_target_len) for x in batch],
                 dtype=np.float32)
    spec_batch = torch.FloatTensor(c)

    if num_inputs > 4:
        add_info = [x[-1] for x in batch]
        return ids, x_batch, input_lengths, mel_batch, spec_batch, add_info
    else:
        return ids, x_batch, input_lengths, mel_batch, spec_batch, None



