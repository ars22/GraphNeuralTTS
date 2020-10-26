"""
Runs inference over trained HRG model
"""
import argparse
import yaml
import torch
import numpy as np
from src.utils import AudioProcessor
import json
from tqdm import tqdm

# Imports based on HRG or No HRG
MODE = "HRG"
print("MODE : ", MODE)
if MODE == "HRG":
    from src.dataset_hrg import getDataLoader
    from src.module_hrg import TacotronHRG as Tacotron
else:
    from src.dataset import getDataLoader
    from src.module import Tacotron



def generate_speech(args):
    
    config = yaml.load(open(args.config, 'r'))
    # read model
    model, vocab, add_info_vocab = load_ckpt(config, ckpt_path=args.checkpoint_path)
    model = model.cuda()
    # create dataset
    inference_dataloader = getDataLoader(
        mode='test',
        meta_path=args.test_file,
        data_dir=config['solver']['data_dir'],
        batch_size=32,
        r=config['model']['tacotron']['r'],
        n_jobs=config['solver']['n_jobs'],
        use_gpu=args.gpu,
        add_info_headers=config['solver']['add_info_headers'],
        vocab=vocab,
        add_info_vocab=add_info_vocab)
    
    # Decode 
    ap = AudioProcessor(**config['audio'])
    with torch.no_grad():
        for batch_i, (graphs, txt_lengths, _, _, add_info) in enumerate(inference_dataloader):
            sorted_lengths, indices = torch.sort(txt_lengths.view(-1), dim=0, descending=True)
            indices = indices.long().numpy()
            sorted_lengths = sorted_lengths.long().numpy()
            if type(graphs) == list:
                graphs = [graphs[idx] for idx in indices]
            else:
                graphs = graphs[indices]

            mel, spec, attn = model(graphs, text_lengths=sorted_lengths, add_info=add_info)
            # Generate wav file
            num_files_to_write = len(graphs)
            for i in tqdm(range(num_files_to_write), desc="Generating speech"):
                wav = ap.inv_spectrogram(spec[i].cpu().numpy().T)
                ap.save_wav(wav, f"{args.output_dir}/{batch_i*32 + i}.wav")


def load_ckpt(config, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    
    vocab = ckpt['vocab']
    add_info_vocab = ckpt['add_info_vocab']
    print(ckpt['add_info_vocab'])
    if MODE == "HRG":
        config['model']['tacotron']['n_vocab'] = vocab.n_vocab
        if add_info_vocab:
            config['model']['tacotron']['n_add_info_vocab'] = max([add_info_vocab[h].n_vocab for h in add_info_vocab])
            config['model']['tacotron']['add_info_headers'] = list(add_info_vocab.keys())
    model = Tacotron(**config['model']['tacotron'])
    model.load_state_dict(ckpt['state_dict'])
    # This yeilds the best performance, not sure why
    # model.mel_decoder.eval()
    model.embedding.eval()
    model.encoder.eval()
    model.postnet.eval()
    return model, vocab, add_info_vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synthesize speech')
    parser.add_argument('--test-file', type=str, help='Path to a meta_test file', required=True)
    parser.add_argument('--hrg', action='store_true', help='run tacotron hrg', default=False)
    parser.add_argument('--output-dir', type=str, help='Output path where the wav files will be generated', required=False)
    parser.add_argument('--checkpoint-path', type=str, help='Checkpoint path', required=True)
    parser.add_argument('--config', type=str, help='Path to experiment config file')
    parser.add_argument('--seed', default=0, type=int, help='Random seed for reproducible results.', required=False)
    parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU training')
    parser.add_argument('--no-msg', action='store_true', help='Hide all messages')
    
    args = parser.parse_args()
    
    generate_speech(args)




