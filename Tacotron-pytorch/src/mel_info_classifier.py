from torch.utils.data import Dataset
import torch
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import glob
import os
from collections import Counter
from tensorboardX import SummaryWriter
from tqdm import tqdm
import subprocess

def get_1dconv(in_channels, out_channels, max_pool=False):
    return nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1),
                      nn.ELU(),
                      nn.BatchNorm1d(out_channels),
                      nn.MaxPool1d(3, stride=2) if max_pool else nn.Identity(),
                      nn.Dropout(p=0.1))
                      
class MelClassifier(nn.Module):
    def __init__(self, 
                 num_class,
                 mel_spectogram_dim: int = 80,
                 gru_hidden_size=32,
                 gru_num_layers=2):
        super(MelClassifier, self).__init__()
        self.num_class = num_class
        self.conv_blocks = nn.Sequential(
                        get_1dconv(in_channels=mel_spectogram_dim, out_channels=128))
#                         get_1dconv(in_channels=64, out_channels=128),
#                         get_1dconv(in_channels=128, out_channels=128, max_pool=True),
#                         get_1dconv(in_channels=128, out_channels=128, max_pool=True),
#                         get_1dconv(in_channels=128, out_channels=128, max_pool=True))
            
        self.gru = nn.GRU(input_size=128, hidden_size=gru_hidden_size, num_layers=gru_num_layers,\
                          bidirectional=True, batch_first=True, dropout=0.3)
        num_directions = 2
        self.mlp = nn.Linear(gru_hidden_size * gru_num_layers * num_directions, self.num_class)

    def forward(self, mel_batch, input_lengths):
        batch_size = len(mel_batch)
        # mel_batch -> (batch_size, max_time_step, 80)
        conv_output = self.conv_blocks(mel_batch.permute(0, 2, 1)).permute(0, 2, 1)
        # conv_output -> (batch_size, max_time_step, 32)

        output, h_n = self.gru(conv_output)
        # h_n -> (4, batch_size, 32)
        
        h_n = h_n.permute(1, 0, 2).reshape(batch_size, -1)
        return h_n, self.mlp(h_n)
        