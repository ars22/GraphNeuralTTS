import os
import numpy as np
import pandas as pd
import torch
from .symbols import txt2seq
from functools import partial
from torch_geometric.data import Data, Dataset


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


def hrg_to_graph(hrg):
    """
    Converts the HRG to graph,
    
    Returns:
        Edge index: (num_edges, 2)
        Node features: (num_nodes, feature_dim)
    """
    # words, sylls = [], []
    # node_idx = {}
    # edges = []
    # node_idxs = []
    # syll_node_idxs = []
    # for i, word_rep in enumerate(hrg["hrg"]):
    #     word_node = f"{word_rep['word']}-{i}"
    #     word_node_id = get_tok2id(word_rep['word'])
        
    #     node_idx[word_node] = len(node_idx)
    #     node_features.append(embeddings[word_node_id, :])
    #     for j, syll in enumerate(word_rep["daughters"]):
    #         syll_node = f"{syll['syll']}-{i}-{j}"
    #         syll_node_id = get_tok2id(syll['syll'])
            
    #         node_idx[syll_node] = len(node_idx)
    #         syll_node_idxs.add(node_idx[syll_node])
            
    #         node_idxs.append(syll_node_id)
    #         edges.append([node_idx[word_node], node_idx[syll_node]])
    
    n_nodes = np.random.randint(10) + 1
    n_edges = n_nodes + 1
    n_phone_nodes = np.random.randint(n_nodes) + 1
    node_idxs = np.arange(n_nodes)
    syll_node_idxs = np.random.choice(n_nodes, size=n_phone_nodes)
    edges = np.random.choice(n_nodes, size=(2, n_edges))
    return Data(x=torch.tensor(node_idxs, dtype=torch.long), edge_index=torch.tensor(edges, dtype=torch.long).contiguous(),\
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
        meta = {'hrg':[], 'mel': [], 'spec': []}
        with open(meta_path) as f:
            for line in f.readlines():
                # If there is '\n' in text, it will be discarded when calling symbols.txt2seq
                fmel, fspec, n_frames, hrg = line.split('|')
                meta['hrg'].append(hrg)
                meta['mel'].append(fmel)
                meta['spec'].append(fspec)

        self.X = [os.path.join(data_dir, f) for f in meta['hrg']]
        self.Y_mel = [os.path.join(data_dir, f) for f in meta['mel']]
        self.Y_spec = [os.path.join(data_dir, f) for f in meta['spec']]
        assert len(self.X) == len(self.Y_mel) == len(self.Y_spec)
        

    def __getitem__(self, idx):
        item = (hrg_to_graph(self.X[idx]),
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