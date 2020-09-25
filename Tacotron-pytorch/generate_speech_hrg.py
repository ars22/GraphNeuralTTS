import argparse
import yaml
import torch
import numpy as np
from src.module_hrg import Tacotron
from src.utils import AudioProcessor
from dataset_hrg import HRGVocab, HRG

def generate_speech(args):
    print(args)
    # read HRGs
    vocab = HRGVocab.from_dir(args.data_dir)
    graph = HRG(args.hrg_file, vocab=vocab).to_pytorch_geom_graph()
    print(f"Vocab size: {len(vocab)}")

    # read model
    config = yaml.load(open(args.config, 'r'))
    model = load_ckpt(config, vocab=vocab, ckpt_path=args.checkpoint_path).cuda()
   
    # Decode
    with torch.no_grad():
        mel, spec, attn = model([graph])
    # Generate wav file
    ap = AudioProcessor(**config['audio'])
    wav = ap.inv_spectrogram(spec[0].cpu().numpy().T)
    ap.save_wav(wav, args.output)


def load_ckpt(config, vocab, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model = Tacotron(n_vocab = len(vocab), **config['model']['tacotron'])
    model.load_state_dict(ckpt['state_dict'])
    # This yeilds the best performance, not sure why
    # model.mel_decoder.eval()
    model.encoder.eval()
    model.postnet.eval()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synthesize speech')
    parser.add_argument('--data-dir', type=str, help='Directory where the vocab files are present. Typically the data dir', required=True)
    parser.add_argument('--hrg-file', type=str, help='File which contains the HRG to be synthesized', required=True)
    parser.add_argument('--output', default='output.wav', type=str, help='Output path', required=False)
    parser.add_argument('--checkpoint-path', type=str, help='Checkpoint path', required=True)
    parser.add_argument('--config', default='config/config.yaml', type=str, help='Path to config file', required=False)
    args = parser.parse_args()
    generate_speech(args)



