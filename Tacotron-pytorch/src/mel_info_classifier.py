"""Trains a classifier from mel spectrogram -> speaker characteristic

Usage:
    mel_info_classifier.py [options]

Options:
    --data-path=<str>                 [Folder that contains the mel files]
    --checkpoint-path=<str>          [Folder where the classifier checkpoint is supposed to be stored]
"""
from docopt import docopt
from os import stat
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
                         nn.MaxPool1d(
                             3, stride=2) if max_pool else nn.Identity(),
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

        self.gru = nn.GRU(input_size=128, hidden_size=gru_hidden_size, num_layers=gru_num_layers,
                          bidirectional=True, batch_first=True, dropout=0.3)
        num_directions = 2
        self.mlp = nn.Linear(
            gru_hidden_size * gru_num_layers * num_directions, self.num_class)

    def forward(self, mel_batch, input_lengths):
        batch_size = len(mel_batch)
        # mel_batch -> (batch_size, max_time_step, 80)
        conv_output = self.conv_blocks(
            mel_batch.permute(0, 2, 1)).permute(0, 2, 1)
        # conv_output -> (batch_size, max_time_step, 32)

        output, h_n = self.gru(conv_output)
        # h_n -> (4, batch_size, 32)

        h_n = h_n.permute(1, 0, 2).reshape(batch_size, -1)
        return h_n, self.mlp(h_n)


def _pad_2d(x, max_len):
    x = np.pad(x, [(0, max_len - len(x)), (0, 0)],
               mode="constant", constant_values=0)
    return x


class DatasetUtils(object):
    def get_accent_from_filename(self, filename: str) -> str:
        raise NotImplementedError()

    def get_speaker_from_filename(self, filename: str):
        raise NotImplementedError()


class ArcticUtils(DatasetUtils):
    """Arctic dataset specific methods for extracting the speaker
    and accent information from the name of a mel file.
    """
    speaker_accent_dict = {
        ""
    }

    def get_accent_from_filename(self, filename: str) -> str:
        speaker = self.get_speaker_from_filename(filename)
        return speaker
        return ArcticUtils.speaker_accent_dict[speaker]

    def get_speaker_from_filename(self, filename: str):
        filename = Path(filename).parts[-1]  # remove the basepath if present
        # files are named arctic_aew_a0050-linear.npy
        _, speaker, _ = filename.split("_")
        return speaker


class MelClassifierDataset(Dataset):
    """Dataset for the MEL based accent/speaker classification

    Args:
        Dataset ([type]): [description]
    """

    def __init__(self, basepth: str, label_from_filename_func):
        self.mel_files = glob.glob(f"{basepth}/*mel*")
        print(f"{len(self.mel_files)} mel-files found")
        # the files are supposed to be named dataset_speaker_*.wav,
        # e.g. australian_s02_362.wav
        self.labels = [label_from_filename_func(
            mel_file_pth) for mel_file_pth in self.mel_files]
        self.label_dict = {k: i for i, k in enumerate(
            sorted(Counter(self.labels).keys()))}
        print(self.label_dict)

    def __getitem__(self, i):
        return np.load(self.mel_files[i])

    def __len__(self):
        return len(self.mel_files)

    @staticmethod
    def batchify(dataset, bsz, shuffle=True):
        idx = list(range(len(dataset)))
        if shuffle:
            np.random.shuffle(idx)

        for begin in range(0, len(dataset), bsz):
            end = min(begin + bsz, len(dataset))
            # read all the mels for this batch, find the max length
            mels = [dataset[idx[i]] for i in range(begin, end)]
            seq_lengths = torch.LongTensor([len(mel) for mel in mels])
            max_target_len = seq_lengths.max().item()

            b = np.array([_pad_2d(mel, max_target_len) for mel in mels],
                         dtype=np.float32)
            mel_batch = torch.FloatTensor(b)
            labels = torch.LongTensor([dataset.label_dict[dataset.labels[idx[i]]]
                                       for i in range(begin, end)])

            seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

            yield mel_batch[perm_idx], labels[perm_idx], seq_lengths


class MelClassifierTrainer(object):
    def __init__(self, datapth: str, checkpoint_pth: str,
                 dataset_utils_class=ArcticUtils(),
                 attribute: str = "accent",
                 bsz: int = 32, num_epochs=30) -> None:

        self.datapth = datapth
        self.checkpoint_pth = checkpoint_pth
        if attribute == "accent":
            self.dataset = MelClassifierDataset(
                datapth, label_from_filename_func=dataset_utils_class.get_accent_from_filename)
        else:
            raise NotImplementedError()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = MelClassifier(
            len(self.dataset.label_dict)).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3,
                                           betas=(0.9, 0.99),
                                           eps=1e-6,
                                           weight_decay=0.01)
        self.bsz, self.num_epochs = bsz, num_epochs

    def run(self):
        loss_func = nn.CrossEntropyLoss()

        losses = []
        accuracy = []
        for epoch in range(self.num_epochs):

            dataloader = self.dataset.batchify(self.dataset, bsz=self.bsz)

            for i, (mels, labels, input_lengths) in enumerate(dataloader):
                mels = mels.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                h_n, logits = self.model(mels, input_lengths)
                loss = loss_func(logits, labels).mean()
                accuracy.append(sum(torch.argmax(logits, dim=1)
                                    == labels).item() * 100. / len(labels))

                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                if i % 50 == 0:
                    print(
                        f"Epoch = {epoch} iter = {i} Loss = {round(np.array(losses).mean(), 2)} Acc = {round(np.array(accuracy).mean(), 2)}")
                    losses = []


if __name__ == "__main__":
    args = docopt(__doc__)
    trainer = MelClassifierTrainer(
        datapth=args["--data-path"], checkpoint_pth=args["--checkpoint-path"])
    trainer.run()
