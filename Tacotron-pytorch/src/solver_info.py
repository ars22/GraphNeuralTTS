from math import inf
import os
import math
import numpy as np
import json
from numpy.lib.utils import info
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
# from .dataset_hrg import getDataLoader
# from .module_hrg import TacotronHRG as Tacotron
from .utils import AudioProcessor, make_spec_figure, make_attn_figure, clip_gradients_custom
import shutil
from torch_geometric.data import Batch
from matplotlib import pyplot as plt
from .mel_info_classifier import MelClassifier

# Imports based on HRG or No HRG
MODE = "HRG"
print("MODE : ", MODE)
if MODE == "HRG":
    from .dataset_hrg import getDataLoader
    from .module_hrg import TacotronHRG as Tacotron
else:
    from .dataset import getDataLoader
    from .module import Tacotron


class Solver(object):
    """Super class Solver for all kinds of tasks (train, test)"""

    def __init__(self, config, args):
        self.use_gpu = args.gpu and torch.cuda.is_available()
        self.device = torch.device(
            'cuda') if self.use_gpu else torch.device('cpu')
        self.config = config
        self.args = args

    def verbose(self, msg):
        print(' '*100, end='\r')
        if self.args.verbose:
            print("[INFO]", msg)

    def progress(self, msg):
        if self.args.verbose:
            print(msg + ' '*40, end='\r')


class Trainer(Solver):
    """Handle training task"""

    def __init__(self, config, args):
        super(Trainer, self).__init__(config, args)

        # Best validation error, initialize it with a large number
        self.best_val_err = 1e10
        # Logger Settings
        name = Path(args.checkpoint_dir).stem
        self.log_dir = str(Path(args.log_dir, name))
        self.log_writer = SummaryWriter(self.log_dir)
        self.checkpoint_path = args.checkpoint_path
        self.checkpoint_dir = args.checkpoint_dir
        self.add_info_headers = []
        if "add_info_headers" in config['solver']:
            self.add_info_headers = config['solver']['add_info_headers']
            config['model']['tacotron']['add_info_headers'] = self.add_info_headers
            print("Using additional headers", self.add_info_headers)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.config = config
        self.audio_processor = AudioProcessor(**config['audio'])
        # Training detail
        self.step = 0
        self.max_step = config['solver']['total_steps']

    def load_data(self):
        """Load data for train and validation"""
        self.verbose("Load data")

        _config = self.config['solver']
        print("--" * 10, "Training Data", "--" * 10)
        # Training dataset
        self.data_tr = getDataLoader(
            mode='train',
            meta_path=_config['meta_path']['train'],
            data_dir=_config['data_dir'],
            batch_size=_config['batch_size'],
            r=self.config['model']['tacotron']['r'],
            n_jobs=_config['n_jobs'],
            use_gpu=self.use_gpu,
            add_info_headers=self.add_info_headers)
        print("--" * 10, "Validation Data", "--" * 10)
        # Validation dataset
        self.data_va = getDataLoader(
            mode='test',
            meta_path=_config['meta_path']['test'],
            data_dir=_config['data_dir'],
            batch_size=_config['batch_size'],
            r=self.config['model']['tacotron']['r'],
            n_jobs=_config['n_jobs'],
            use_gpu=self.use_gpu,
            add_info_headers=self.add_info_headers,
            vocab=self.data_tr.dataset.vocab,
            add_info_vocab=self.data_tr.dataset.add_info_vocab)

        # vocab sizes
        if hasattr(self.data_tr.dataset, 'n_vocab'):
            self.config['model']['tacotron']['n_vocab'] = self.data_tr.dataset.n_vocab
        if len(self.add_info_headers):
            self.config['model']['tacotron']['n_add_info_vocab'] = self.data_tr.dataset.n_add_info_vocab

    def build_model(self):
        """Build model"""
        self.verbose("Build model")

        self.model = Tacotron(
            **self.config['model']['tacotron']).to(device=self.device)

        self.info_classifier = MelClassifier(
            num_class=self.config['model']['tacotron']['n_add_info_vocab'], gru_dropout=0.0).to(device=self.device)
        self.info_classifier.load_state_dict(torch.load(
            self.config['solver']['classifier_checkpoint']))

        for param in self.info_classifier.parameters():
            param.requires_grad = False
        self.info_classifier.conv_blocks.eval()
        self.info_classifier.mlp.eval()

        self.info_criterion = torch.nn.CrossEntropyLoss()
        print(self.info_classifier.gru.dropout)
        self.criterion = torch.nn.L1Loss()

        # Optimizer
        _config = self.config['model']
        lr = _config['optimizer']['lr']
        optim_type = _config['optimizer'].pop('type', 'Adam')
        self.optim = getattr(torch.optim, optim_type)
        self.optim = self.optim(self.model.parameters(),
                                **_config['optimizer'])
        # Load checkpoint if specify
        if self.checkpoint_path is not None:
            self.load_ckpt()

    def update_optimizer(self):
        warmup_steps = 4000.0
        step = self.step + 1.
        init_lr = self.config['model']['optimizer']['lr']
        current_lr = init_lr * warmup_steps**0.5 * np.minimum(
            step * warmup_steps**-1.5, step**-0.5)
        for param_group in self.optim.param_groups:
            param_group['lr'] = current_lr
        return current_lr

    def exec(self):
        """Train"""
        local_step = 0
        fs = self.config['audio']['sample_rate']
        linear_dim = self.model.linear_size
        n_priority_freq = int(3000 / (fs * 0.5) * linear_dim)
        self.model.train()
        self.verbose('Start training: {} batches'.format(len(self.data_tr)))
        while self.step < self.max_step:
            for curr_b, (_, txt, text_lengths, mel, spec, add_info) in enumerate(self.data_tr):
                # Sort data by length
                sorted_lengths, indices = torch.sort(
                    text_lengths.view(-1), dim=0, descending=True)
                indices = indices.long().numpy()
                sorted_lengths = sorted_lengths.long().numpy()
                if type(txt) == list:
                    txt = [txt[idx] for idx in indices]
                else:
                    txt = txt[indices]
                    txt = txt.to(device=self.device)

                mel, spec = mel[indices], spec[indices]

                mel = mel.to(device=self.device)
                spec = spec.to(device=self.device)
                
                info_labels = torch.tensor(
                    [x[self.add_info_headers[0]] for x in add_info]).to(device=self.device)

                info_labels = info_labels[indices]


                # Decay learning rate
                current_lr = self.update_optimizer()

                # Forwarding
                self.optim.zero_grad()
                mel_outputs, linear_outputs, attn = self.model(
                    txt, add_info=add_info, melspec=mel, text_lengths=sorted_lengths)
                mel_loss = self.criterion(mel_outputs, mel)

                # info loss

                h_n, logits = self.info_classifier(mel_outputs, sorted_lengths)

                info_loss = self.info_criterion(logits, info_labels).mean()
                # Count linear loss
                linear_loss = 0.5 * self.criterion(linear_outputs, spec) \
                    + 0.5 * \
                    self.criterion(
                        linear_outputs[:, :, :n_priority_freq], spec[:, :, :n_priority_freq])

                loss = mel_loss + linear_loss + info_loss
                loss.backward()

                # Switching to a diff. grad norm scheme
                # https://github.com/festvox/festvox/blob/c7f6fa1b51a1ed6251148f8849fd879c2d7263f4/challenges/msrlid2020/partA/local/util.py#L845
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config['solver']['grad_clip'])
                # max_grad, _ = clip_gradients_custom(self.model, self.config['solver']['grad_clip'])

                # Skip this update if grad is NaN
                if math.isnan(grad_norm):
                    self.verbose(
                        'Error : grad norm is NaN @ step ' + str(self.step))
                else:
                    self.optim.step()

                # Add to tensorboard
                if self.step % 5 == 0:
                    self.write_log('Loss', {
                        'total_loss': loss.item(),
                        'mel_loss': mel_loss.item(),
                        'linear_loss': linear_loss.item(),
                        'info_loss': info_loss.item()
                    })
                    # self.write_log('max_grad_norm', max_grad)
                    self.write_log('l2_grad_norm', grad_norm)
                    self.write_log('learning_rate', current_lr)

                if self.step % self.config['solver']['log_interval'] == 0:
                    log = '[{}] total_loss: {:.3f}. mel_loss: {:.3f}, linear_loss: {:.3f}, info_loss: {:.3f}, l2_grad_norm: {:.3f}, lr: {:.5f}'.format(
                        self.step, loss.item(), mel_loss.item(), linear_loss.item(), info_loss.item(), grad_norm, current_lr)
                    self.progress(log)

                if self.step % self.config['solver']['validation_interval'] == 0 and local_step != 0:
                    with torch.no_grad():
                        val_err = self.validate()
                    if val_err < self.best_val_err:
                        # Save checkpoint
                        self.save_ckpt()
                        self.best_val_err = val_err
                    self.model.train()

                if self.step % self.config['solver']['save_checkpoint_interval'] == 0 and local_step != 0:
                    self.save_ckpt()

                # Global step += 1
                self.step += 1
                local_step += 1

    def save_ckpt(self):
        ckpt_path = os.path.join(
            self.checkpoint_dir, "checkpoint_step{}.pth".format(self.step))
        torch.save({
            "state_dict": self.model.state_dict(),
            "optimizer": self.optim.state_dict(),
            "global_step": self.step,
            "vocab": self.data_tr.dataset.vocab,
            "add_info_vocab": self.data_tr.dataset.add_info_vocab,
        }, ckpt_path)
        self.verbose(
            "@ step {} => saved checkpoint: {}".format(self.step, ckpt_path))

    def load_ckpt(self):
        self.verbose("Load checkpoint from: {}".format(self.checkpoint_path))
        ckpt = torch.load(self.checkpoint_path)
        vocab = ckpt['vocab']
        add_info_vocab = ckpt['add_info_vocab']

        if MODE == "HRG":
            self.config['model']['tacotron']['n_vocab'] = vocab.n_vocab
            if add_info_vocab:
                self.config['model']['tacotron']['n_add_info_vocab'] = add_info_vocab.n_add_info_vocab
            self.model = Tacotron(**self.config['model']['tacotron'])

        self.model.load_state_dict(ckpt['state_dict'])
        self.model = self.model.to(device=self.device)
        self.optim.load_state_dict(ckpt['optimizer'])
        self.step = ckpt['global_step']

        self.data_tr = getDataLoader(
            mode='train',
            meta_path=self.config['solver']['meta_path']['train'],
            data_dir=self.config['solver']['data_dir'],
            batch_size=self.config['solver']['batch_size'],
            r=self.config['model']['tacotron']['r'],
            n_jobs=self.config['solver']['n_jobs'],
            use_gpu=self.use_gpu,
            add_info_headers=self.add_info_headers,
            vocab=vocab,
            add_info_vocab=add_info_vocab)

        self.data_va = getDataLoader(
            mode='test',
            meta_path=self.config['solver']['meta_path']['test'],
            data_dir=self.config['solver']['data_dir'],
            batch_size=self.config['solver']['batch_size'],
            r=self.config['model']['tacotron']['r'],
            n_jobs=self.config['solver']['n_jobs'],
            use_gpu=self.use_gpu,
            add_info_headers=self.add_info_headers,
            vocab=vocab,
            add_info_vocab=add_info_vocab)

    def write_log(self, val_name, val_dict):
        if type(val_dict) == dict:
            self.log_writer.add_scalars(val_name, val_dict, self.step)
        else:
            self.log_writer.add_scalar(val_name, val_dict, self.step)

    def validate(self):
        # (r9y9's comment) Turning off dropout of decoder's prenet causes serious performance
        # drop, not sure why.
        self.model.embedding.eval()
        self.model.encoder.eval()
        self.model.postnet.eval()

        fs = self.config['audio']['sample_rate']
        linear_dim = self.model.linear_size
        n_priority_freq = int(3000 / (fs * 0.5) * linear_dim)

        mel_loss_avg = 0.0
        linear_loss_avg = 0.0
        total_loss_avg = 0.0

        for curr_b, (_, txt, text_lengths, mel, spec, add_info) in enumerate(self.data_va):
            # Sort data by length
            sorted_lengths, indices = torch.sort(
                text_lengths.view(-1), dim=0, descending=True)
            indices = indices.long().numpy()
            sorted_lengths = sorted_lengths.long().numpy()
            if type(txt) == list:
                txt = [txt[idx] for idx in indices]
            else:
                txt = txt[indices]
                txt = txt.to(device=self.device)
            mel, spec = mel[indices], spec[indices]

            mel = mel.to(device=self.device)
            spec = spec.to(device=self.device)

            # Forwarding
            mel_outputs, linear_outputs, attn = self.model(
                txt, add_info=add_info, melspec=mel, text_lengths=sorted_lengths)

            mel_loss = self.criterion(mel_outputs, mel)
            # Count linear loss
            linear_loss = 0.5 * self.criterion(linear_outputs, spec) \
                + 0.5 * \
                self.criterion(
                    linear_outputs[:, :, :n_priority_freq], spec[:, :, :n_priority_freq])
            loss = mel_loss + linear_loss

            mel_loss_avg += mel_loss.item()
            linear_loss_avg += linear_loss.item()
            total_loss_avg += loss.item()

            NUM_GL = 5
            if curr_b < NUM_GL:
                fig_spec = make_spec_figure(
                    linear_outputs[0].cpu().numpy(), self.audio_processor)
                fig_attn = make_attn_figure(attn[0].cpu().numpy())
                # Predicted audio signal
                waveform = self.audio_processor.inv_spectrogram(
                    linear_outputs[0].cpu().numpy().T)
                waveform = np.clip(waveform, -1, 1)
                # Tensorboard
                self.log_writer.add_figure(
                    'spectrogram-%d' % curr_b, fig_spec, self.step)
                self.log_writer.add_figure(
                    'attn-%d' % curr_b, fig_attn, self.step)
                self.log_writer.add_audio(
                    'wav-%d' % curr_b, waveform, self.step, sample_rate=fs)

                # Decode non-teacher-forced
                mel_outputs, linear_outputs, attn = self.model(
                    [txt[0]], add_info=[add_info[0]] if add_info else None)
                fig_spec = make_spec_figure(
                    linear_outputs[0].cpu().numpy(), self.audio_processor)
                fig_attn = make_attn_figure(attn[0].cpu().numpy())
                # Predicted audio signal
                waveform = self.audio_processor.inv_spectrogram(
                    linear_outputs[0].cpu().numpy().T)
                waveform = np.clip(waveform, -1, 1)
                # Tensorboard
                self.log_writer.add_figure(
                    'non-teacher-forced-spectrogram-%d' % curr_b, fig_spec, self.step)
                self.log_writer.add_figure(
                    'non-teacher-forced-attn-%d' % curr_b, fig_attn, self.step)
                self.log_writer.add_audio(
                    'non-teacher-forced-wav-%d' % curr_b, waveform, self.step, sample_rate=fs)

            # Perform Griffin-Lim to generate waveform: "GL"
            header = '[GL-{}/{}]'.format(curr_b + 1, NUM_GL) if curr_b < NUM_GL else '[VAL-{}/{}]'.format(
                curr_b + 1, len(self.data_va))
            # Terminal log
            log = header + ' total_loss: {:.3f}. mel_loss: {:.3f}, linear_loss: {:.3f}'.format(
                loss.item(), mel_loss.item(), linear_loss.item())

            self.progress(log)

        mel_loss_avg /= len(self.data_va)
        linear_loss_avg /= len(self.data_va)
        total_loss_avg /= len(self.data_va)

        self.verbose('@ step {} => total_loss: {:.3f}, mel_loss: {:.3f}, linear_loss: {:.3f}'.format(
            self.step, total_loss_avg, mel_loss_avg, linear_loss_avg))

        self.write_log('Loss', {
            'total_loss_val': total_loss_avg,
            'mel_loss_val': mel_loss_avg,
            'linear_loss_val': linear_loss_avg
        })
        return linear_loss_avg

    def embed_similarity(self):

        self.model.embedding.eval()
        diff_phoneme_feat = []
        for curr_b, (txt, text_lengths, mel, spec, add_info) in enumerate(self.data_va):
            syll_nodes = [d.syll_nodes for d in txt]
            prev_syll_nodes = [np.insert(s[:-1], 0, -1) for s in syll_nodes]
            word_bound = [np.append(np.where((s-p) != 1)[0], len(s))
                          for (s, p) in zip(syll_nodes, prev_syll_nodes)]
            text_feat = self.model.embedding(txt).detach()     # B x S x D
            word_feat = []
            for i, word in enumerate(word_bound):
                for w, w1 in zip(word[:-1], word[1:]):
                    word_feat.append(text_feat[i][w:w1])
            avg_word_feat = [wf.mean(axis=0) for wf in word_feat]
            diff_word_feat = [(torch.sum(torch.abs(wf - awf)**2, axis=-1)**(1./2)).to('cpu')
                              for wf, awf in zip(word_feat, avg_word_feat)]
            diff_phoneme_feat.append(np.concatenate(diff_word_feat).ravel())
        diff_phoneme_feat = np.concatenate(diff_phoneme_feat).ravel()
        print("Number of phonemes in total: %d" % len(diff_phoneme_feat))
        sns.kdeplot(diff_phoneme_feat, shade=True, clip=(0, 0.5))
        plt.savefig(
            'plots/arctic-hrg-val-phoneme-diff-avg-l2.gcn3.epoch0.density.png')
        plt.hist(diff_phoneme_feat, bins=20, range=(0, 0.25))
        plt.savefig(
            'plots/arctic-hrg-val-phoneme-diff-avg-l2.gcn3.epoch0.hist.png')
