# [NOTE]
# This file is highly based on r9y9's tacotron implementation
# : https://github.com/r9y9/tacotron_pytorch
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data.batch import Batch
from torch_geometric.data import Data
from torch_geometric.nn import SGConv
from .module import CBHG, Encoder, MelDecoder
from typing import List
from collections import OrderedDict
from src.utils import get_tokens_from_additional_info


class EmbeddingHRG(nn.Module):
    def __init__(self, n_vocab, embedding_size, hidden_size):
        super(EmbeddingHRG, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.embedding.weight.data.normal_(0, 0.3)
        self.conv1 = GCNConv(embedding_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.pad_vector = nn.Parameter(
            torch.randn(hidden_size), requires_grad=True)
        
        # initialization scheme from 
        # https://github.com/festvox/festvox/blob/c7f6fa1b51a1ed6251148f8849fd879c2d7263f4/voices/arctic/rms/local/model.py#L193
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def forward(self, graphs):
        """Embeds a list of HRGs

        Args:
            data ([type]): [List of Pytorch Geom Data objects]

        Returns:
            [type]: [description]
        """

        #  step 1: assign embedding to each node
        graphs = self._assign_node_features(graphs)

        #  step 2: batch the graphs together
        batch = Batch.from_data_list(graphs).to(self.device)

        #  step 3: learn node representations using GCN over the batch
        x = self._gcn(batch)
        #  (num_utterances * ?, hidden_size)

        #  step 4: break the batch into utterances again (reverse of step 2),
        #  but only retain phone/syll nodes
        x, max_seq_len = self._break_into_utterances(x, batch)
        #  (num_utterances, ?, hidden_size), max_seq_len (max no. of syll nodes across utterances)
        
        #  step 5: pad the gcn outputs of utterances with a random vector
        x = self._pad_embeddings(x, max_seq_len)
        #  (num_utterances, max_seq_len, hidden_size)
        return x

    def _assign_node_features(self, graphs: List[Data]) -> Data:
        res = []
        for graph in graphs:
            res.append(Data(x=self.embedding(graph.x.to(self.device)), edge_index=graph.edge_index, syll_nodes=graph.syll_nodes))
        return res


    def _gcn(self, batch):
        """Runs GCN over a pytorch geom batch

        Args:
            batch ([type]): [description]

        Returns:
            [type]: [description]
        """
        x, edge_index = batch.x, batch.edge_index
        return batch.x


    def _break_into_utterances(self, x, batch):
        offset = 0
        res = []
        batch_graphs = batch.to_data_list()
        max_seq_len = -1
        for graph in batch_graphs:
            res.append(x[offset + graph.syll_nodes])
            offset += graph.x.shape[0]
            max_seq_len = max(max_seq_len, len(graph.syll_nodes))
        return res, max_seq_len

    def _pad_embeddings(self, x, max_seq_len):
        res = []
        for gcn_output in x:
            res.append(torch.cat(
                [gcn_output, self.pad_vector.repeat(max_seq_len - gcn_output.shape[0], 1)], dim=0))
        return torch.stack(res ,dim=0)




class TacotronHRG(nn.Module):
    def __init__(self, n_vocab, embedding_size=256, gcn_hidden_size=128, add_info_embedding_size=32, mel_size=80, linear_size=1025, r=5, 
            add_info_headers=[], n_add_info_vocab=0):
        super(TacotronHRG, self).__init__()
        self.mel_size = mel_size
        self.linear_size = linear_size
        # main embedding for HRGs
        self.embedding = EmbeddingHRG(n_vocab, hidden_size=gcn_hidden_size, embedding_size=embedding_size)
        # if there are additional headers, create an embedding file for each
        self.add_info_headers = add_info_headers
        self.add_info_embedding = nn.Sequential(OrderedDict([
            (header, nn.Embedding(n_add_info_vocab, add_info_embedding_size))
            for header in self.add_info_headers
        ]))
        
        for header in self.add_info_headers:
            self.add_info_embedding._modules[header].weight.data.normal_(0, 0.3)
        # the embedding size scales with more additional headers
        self.encoder = Encoder(embedding_size)
        self.mel_decoder = MelDecoder(mel_size, r, add_info_headers, add_info_embedding_size)
        self.postnet = CBHG(mel_size, K=8, hidden_sizes=[256, mel_size])
        self.last_proj = nn.Linear(mel_size * 2, linear_size)

    def forward(self, texts, add_info=None, melspec=None, text_lengths=None):
        txt_feat = self.embedding(texts)
        batch_size = len(texts)
        # -> (batch_size, timesteps (encoder), text_dim)
        encoder_outputs = self.encoder(txt_feat, text_lengths)

        # if there are additional headers like speaker or accent we
        # append them to encoder output
        if len(self.add_info_headers):
            additional_embeddings = []
            for header in self.add_info_headers:
                add_info_tensor = get_tokens_from_additional_info(add_info, header).to(encoder_outputs.device)
                additional_embeddings.append(
                    self.add_info_embedding._modules[header](add_info_tensor).unsqueeze(1).repeat(1, encoder_outputs.size(1), 1))
            encoder_outputs = torch.cat([encoder_outputs] + additional_embeddings, dim=-1)
            # encoder_outputs now has concatenated embeddings

        mel_outputs, alignments = self.mel_decoder(encoder_outputs, melspec)
        # Reshape mel_outputs
        # -> (batch_size, timesteps (decoder), mel_size)
        mel_outputs = mel_outputs.view(batch_size, -1, self.mel_size)
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_proj(linear_outputs)
        return mel_outputs, linear_outputs, alignments

