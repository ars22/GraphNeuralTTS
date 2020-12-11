"""Trains a classifier from mel spectrogram -> speaker characteristic

Usage:
    mel_info_classifier.py [options]

Options:
    --data-path=<str>                 [Folder that contains the mel files]
    --splits-path=<str>               [Path to the splits file]
    --checkpoint-path=<str>           [Folder where the classifier checkpoint is supposed to be stored]
    --attribute=<str>                 [Attribute (speaker/accent)]
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
from typing import List


class MelClassifier(nn.Module):
    @staticmethod
    def get_1dconv(in_channels, out_channels, max_pool=False):
        return nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1),
                             nn.ELU(),
                             nn.BatchNorm1d(out_channels),
                             nn.MaxPool1d(
            3, stride=2) if max_pool else nn.Identity(),
            nn.Dropout(p=0.1))

    def __init__(self,
                 num_class,
                 mel_spectogram_dim: int = 80,
                 gru_hidden_size=32,
                 gru_num_layers=2,
                 gru_dropout=0.3):
        super(MelClassifier, self).__init__()
        self.num_class = num_class
        self.conv_blocks = nn.Sequential(
            MelClassifier.get_1dconv(in_channels=mel_spectogram_dim, out_channels=128))
#                         get_1dconv(in_channels=64, out_channels=128),
#                         get_1dconv(in_channels=128, out_channels=128, max_pool=True),
#                         get_1dconv(in_channels=128, out_channels=128, max_pool=True),
#                         get_1dconv(in_channels=128, out_channels=128, max_pool=True))

        self.gru = nn.GRU(input_size=128, hidden_size=gru_hidden_size, num_layers=gru_num_layers,
                          bidirectional=True, batch_first=True, dropout=gru_dropout)
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
        "aew": "american", "bdl": "american", "clb": "american", "eey": "american",
        "jmk": "american", "ljm": "american", "lnh": "american", "rms": "american",
        "slt": "american", "ahw": "european", "awb": "european", "fem": "european",
        "rxr": "european", "aup": "hindi", "axb": "hindi", "slp": "hindi", "gka": "telugu", "ksp": "telugu"
    }

    def get_accent_from_filename(self, filename: str) -> str:
        speaker = self.get_speaker_from_filename(filename)
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

    def __init__(self, datapath: str, mel_files: List[str], label_from_filename_func):
        self.mel_files = []
        for mel_file in mel_files:
            basename = mel_file.split(".")[0]
            self.mel_files.append(f"{datapath}/{basename}-mel.npy")

        # the files are supposed to be named dataset_speaker_*.wav,
        # e.g. australian_s02_362.wav
        self.labels = [
            f"{datapath}/{label_from_filename_func(mel_file_pth)}" for mel_file_pth in self.mel_files]
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
    def __init__(self, splits_pth: str, checkpoint_pth: str,
                 datapath: str,
                 dataset_utils_class=ArcticUtils(),
                 attribute: str = "accent",
                 bsz: int = 32, num_epochs=30) -> None:

        with open(splits_pth, "r") as f:
            splits = json.load(f)
            train_files = splits["train"]
            test_files = splits["test"]
            val_files = splits["val"]

        self.checkpoint_pth = checkpoint_pth
        if attribute == "accent":
            self.train_dataset = MelClassifierDataset(
                datapath=datapath,
                mel_files=train_files, label_from_filename_func=dataset_utils_class.get_accent_from_filename)
            self.test_dataset = MelClassifierDataset(
                datapath=datapath,
                mel_files=test_files, label_from_filename_func=dataset_utils_class.get_accent_from_filename)
            self.val_dataset = MelClassifierDataset(
                datapath=datapath,
                mel_files=val_files, label_from_filename_func=dataset_utils_class.get_accent_from_filename)

        elif attribute == "speaker":
            self.train_dataset = MelClassifierDataset(
                datapath=datapath,
                mel_files=train_files, label_from_filename_func=dataset_utils_class.get_speaker_from_filename)
            self.test_dataset = MelClassifierDataset(
                datapath=datapath,
                mel_files=test_files, label_from_filename_func=dataset_utils_class.get_speaker_from_filename)
            self.val_dataset = MelClassifierDataset(
                datapath=datapath,
                mel_files=val_files, label_from_filename_func=dataset_utils_class.get_speaker_from_filename)
        else:
            raise NotImplementedError()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = MelClassifier(
            len(self.train_dataset.label_dict)).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3,
                                           betas=(0.9, 0.99),
                                           eps=1e-6,
                                           weight_decay=0.01)
        self.bsz, self.num_epochs = bsz, num_epochs

    def run(self):
        loss_func = nn.CrossEntropyLoss()

        losses = []
        accuracy = []
        best_val_acc = float("-inf")
        for epoch in range(self.num_epochs):

            dataloader = MelClassifierDataset.batchify(
                self.train_dataset, bsz=self.bsz)

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
            val_acc = self.eval(self.val_dataset, split="val")
            if val_acc > best_val_acc:
                print(f"Accuracy improved from {best_val_acc} to {val_acc}")
                best_val_acc = val_acc
                self.save_model(epoch=epoch, iternumber=i, val_acc=val_acc)

        self.eval(self.test_dataset, split="test")

    def save_model(self, epoch: int, iternumber: int, val_acc: float):
        val_acc = int(val_acc)
        pth = f"{self.checkpoint_pth}/checkpoint_{epoch}_{iternumber}_{val_acc}"
        print(f"Model saved at {pth}")
        torch.save(self.model.state_dict(), pth)

    def eval(self, dataset, split) -> float:
        self.model.eval()
        dataloader = MelClassifierDataset.batchify(dataset, bsz=self.bsz)
        loss_func = nn.CrossEntropyLoss()
        losses = []
        accuracy = []
        for i, (mels, labels, input_lengths) in tqdm(enumerate(dataloader), total=len(dataset) // 32, desc=f"Evaluating {split}"):
            mels = mels.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                h_n, logits = self.model(mels, input_lengths)
                loss = loss_func(logits, labels).mean()
                accuracy.append(sum(torch.argmax(logits, dim=1)
                                    == labels).item() * 100. / len(labels))
                losses.append(loss.item())
        self.model.train()
        print(
            f"Evaluating {split}: loss = {round(np.array(losses).mean(), 2)} accuracy = {round(np.array(accuracy).mean(), 2)}")
        return round(np.array(accuracy).mean(), 2)


if __name__ == "__main__":
    args = docopt(__doc__)
    trainer = MelClassifierTrainer(
        datapath=args["--data-path"],
        splits_pth=args["--splits-path"],
        checkpoint_pth=args["--checkpoint-path"],
        attribute=args["--attribute"])
    trainer.run()
