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
import os

# Imports based on HRG or No HRG
MODE = "CHAR"
print("MODE : ", MODE)
if MODE == "HRG":
    from src.dataset_hrg import getDataLoader
    from src.module_hrg import TacotronHRG as Tacotron
else:
    from src.dataset import getDataLoader
    from src.module import Tacotron



def generate_speech(args):

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    config = yaml.load(open(args.config, 'r'))
    # read model
    model, vocab, add_info_vocab = load_ckpt(args, config, ckpt_path=args.checkpoint_path)
    model = model.cuda()
    batch_size = args.top_k
    
    # create dataset
    inference_dataloader = getDataLoader(
        mode='test',
        meta_path=args.test_file,
        data_dir=config['solver']['data_dir'],
        batch_size=batch_size,
        r=config['model']['tacotron']['r'],
        n_jobs=config['solver']['n_jobs'],
        use_gpu=args.gpu,
        add_info_headers=config['solver']['add_info_headers'],
        vocab=vocab,
        add_info_vocab=add_info_vocab)
    
    # Decode 
    generated = 0
    ap = AudioProcessor(**config['audio'])
    with torch.no_grad():
        for batch_i, (id, graphs, txt_lengths, _, _, add_info) in enumerate(inference_dataloader):
            if generated == args.top_k:
                break
            sorted_lengths, indices = torch.sort(txt_lengths.view(-1), dim=0, descending=True)
            indices = indices.long().numpy()
            sorted_lengths = sorted_lengths.long().numpy()
            if type(graphs) == list:
                graphs = [graphs[idx] for idx in indices]
            else:
                graphs = graphs[indices]
                graphs = graphs.to(device=torch.device('cuda') if args.gpu else torch.device('cpu'))
            
            mel, spec, attn = model(graphs, text_lengths=sorted_lengths, add_info=add_info)
            # Generate wav file
            num_files_to_write = len(graphs)
            for (new_idx, orig_idx) in tqdm(enumerate(indices), desc="Generating speech"):
                wav = ap.inv_spectrogram(spec[new_idx].cpu().numpy().T)
                savename =  "{}/{}.wav".format(args.output_dir, id[orig_idx])
                if add_info is not None:
                    accent = inference_dataloader.dataset.add_info_vocab["accent"].id2tok[add_info[orig_idx]["accent"]]
                    savename =  "{}/{}_target_{}.wav".format(args.output_dir, accent, id[orig_idx])
                ap.save_wav(wav, savename)
                generated += 1
                if generated == args.top_k:
                    break


def load_ckpt(args, config, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    
    vocab = ckpt['vocab']
    add_info_vocab = ckpt['add_info_vocab']
    
    if MODE == "HRG":
        config['model']['tacotron']['n_vocab'] = vocab.n_vocab
    if add_info_vocab:
        config['model']['tacotron']['n_add_info_vocab'] = {h:add_info_vocab[h].n_vocab for h in add_info_vocab}
        config['model']['tacotron']['add_info_headers'] = list(add_info_vocab.keys())
    model = Tacotron(**config['model']['tacotron'])
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device=torch.device('cuda') if args.gpu else torch.device('cpu'))
   
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
    parser.add_argument('--top-k', type=int, help='Only evaluate top-k', required=False, default=10)
    parser.add_argument('--config', type=str, help='Path to experiment config file')
    parser.add_argument('--seed', default=0, type=int, help='Random seed for reproducible results.', required=False)
    parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU training')
    parser.add_argument('--no-msg', action='store_true', help='Hide all messages')
    
    args = parser.parse_args()
    
    generate_speech(args)




