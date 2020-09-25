import os
import numpy as np
import pandas as pd
import json
import torch
from collections import Counter
from functools import partial
from torch_geometric.data import Data, Dataset
import torch_geometric
from .symbols import txt2seq
from .utils import read_json_from_pth


def getDataLoader(mode, meta_path, data_dir, batch_size, r, n_jobs, use_gpu, **kwargs):
    bs = batch_size
    if mode == 'train':
        shuffle = True
    elif mode == 'test':
        shuffle = False
    else:
        raise NotImplementedError
    DS = MyDataset(meta_path, data_dir)
    DL = torch.utils.data.DataLoader(
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

class HRGVocab:
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
    def from_dir(pth):
        """Reads and initializes vocab from a directory

        Args:
            pth ([type]): [path that has id2tok and tok2id jsons.]

        Returns:
            [type]: [HRGVocab]
        """
        with open(f"{pth}/id2tok.json", "r") as f:
            id2tok = json.load(f)
        with open(f"{pth}/tok2id.json", "r") as f:
            tok2id = json.load(f)
        return HRGVocab(id2tok=id2tok, tok2id=tok2id)

    def __len__(self):
        return len(self.tok2id)

class HRG:
    """Place holder class for HRGs.
    Currently just a wrapper around json
    """

    def __init__(self, hrg_json, vocab: HRGVocab):
        """

        Args:
            hrg_json ([dict]): [HRG in json format]
            vocab ([HRGVocab]): [a vocab object that can be used to map the symbols in this HRG]
        """
        self.hrg_json = hrg_json
        self.vocab = vocab
    
    def to_pytorch_geom_graph(self) -> torch_geometric.data.Data:
        """
        Converts the HRG to graph,

        NOTE: idx -> index, a way to identify each node in the graph
            ids -> id for a token returned by the vocab.
            Idxs are primarily used for specifying the connectivity of the graph
        Returns:
            Edge index: (num_edges, 2)
            Node features: (num_nodes, feature_dim)
        """
        words, sylls = [], []
        node_idx = {}
        node_ids = []
        x = []

        edges = []

        syll_node_idxs = []
        for i, word_rep in enumerate(self.hrg_json):
            word_node = f"{word_rep['word']}-{i}"
            word_node_id = self.vocab.get_tok2id(word_rep['word'])
            node_idx[word_node] = len(node_idx)
            x.append(word_node_id)

            for j, daughter in enumerate(word_rep["daughters"]):
                # make syll node
                syll_parent_node = ""
                for syll in daughter:
                    syll_parent_node += syll["syll"]
                syll_parent_node_id = self.vocab.get_tok2id(syll_parent_node)
                x.append(syll_parent_node_id)
                syll_parent_node = f"{syll_parent_node}-{i}-{j}"
                node_idx[syll_parent_node] = len(node_idx)
                edges.append([node_idx[word_node], node_idx[syll_parent_node]])
                # now prepare phone nodes
                for k, syll in enumerate(daughter):
                    
                    syll_node = f"{syll['syll']}-{i}-{j}-{k}"
                    syll_node_id = self.vocab.get_tok2id(syll['syll'])
                    node_idx[syll_node] = len(node_idx)
                    x.append(syll_node_id)
                    syll_node_idxs.append(node_idx[syll_node])

                    edges.append([node_idx[syll_parent_node], node_idx[syll_node]])

        return Data(x=torch.tensor(x, dtype=torch.long), edge_index=torch.tensor(edges, dtype=torch.long).contiguous().t(),
                    syll_nodes=torch.tensor(syll_node_idxs, dtype=torch.long))


class MyDataset(Dataset):
    """Graph Dataset
    """

    def __init__(self, meta_path, data_dir):
        # Load meta
        # ---------
        # text: texts
        # mel : filenames of mel-spectrogram
        # spec: filenames of (linear) spectrogram
        #
        meta = {'hrg': [], 'mel': [], 'spec': []}
        with open(meta_path) as f:
            for line in f.readlines():
                # If there is '\n' in text, it will be discarded when calling symbols.txt2seq
                fmel, fspec, n_frames, hrg= line.split('|')
                meta['hrg'].append(hrg)
                meta['mel'].append(fmel)
                meta['spec'].append(fspec)
        
        # make vocab
        self.vocab = HRGVocab.from_dir(data_dir)
        self.n_vocab = len(self.vocab)

        # Read HRGs, convert each HRG to a Pytorch Geom object
        self.hrgs = [HRG(hrg_json=json.loads(hrg), vocab=self.vocab) for hrg in meta['hrg']]
        self.X = [hrg.to_pytorch_geom_graph() for hrg in self.hrgs]

        # Read audios
        self.Y_mel = [os.path.join(data_dir, f) for f in meta['mel']]
        self.Y_spec = [os.path.join(data_dir, f) for f in meta['spec']]
        assert len(self.X) == len(self.Y_mel) == len(self.Y_spec)

    def __getitem__(self, idx):
        item = (self.X[idx],
                np.load(self.Y_mel[idx]),
                np.load(self.Y_spec[idx]))
        return item

    def __len__(self):
        return len(self.X)


    

def collate_fn(batch, r):
    """
    returns:
    x_batch: List[Data]
    n_phone_nodes: List[int]
    mel_batch: torch.FloatTensor (bsz * max_tgt_len)
    spec_batch: torch.FloatTensor (bsz * max_tgt_len)
    """

    x_batch = [x[0] for x in batch]

    n_phone_nodes = [len(x[0].syll_nodes) for x in batch]
    n_phone_nodes = torch.LongTensor(n_phone_nodes)

    # (r9y9's comment) Add single zeros frame at least, so plus 1
    max_target_len = np.max([len(x[1]) for x in batch]) + 1
    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0

    b = np.array([_pad_2d(x[1], max_target_len) for x in batch],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    c = np.array([_pad_2d(x[2], max_target_len) for x in batch],
                 dtype=np.float32)
    spec_batch = torch.FloatTensor(c)

    return x_batch, n_phone_nodes, mel_batch, spec_batch
