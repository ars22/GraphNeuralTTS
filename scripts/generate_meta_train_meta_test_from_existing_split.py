META_TRAIN_IN = '/home1/asetlur/GraphNeuralTTS/Tacotron-pytorch/data_splits/accent_db/train_split_native.txt'
META_TEST_IN = '/home1/asetlur/GraphNeuralTTS/Tacotron-pytorch/data_splits/accent_db/test_split_native.txt'

META_TRAIN_OUT = '/home1/asetlur/GraphNeuralTTS/Tacotron-pytorch/data_splits/accent_db/meta_train_native.txt'
META_TEST_OUT = '/home1/asetlur/GraphNeuralTTS/Tacotron-pytorch/data_splits/accent_db/meta_test_native.txt'

METADATA = '/home1/asetlur/datasets/accent_db/metadata.csv'

# load metadata
metadata = {}
with open(METADATA, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        key = line.split('|')[0]
        metadata[key] = line


with open(META_TRAIN_IN, 'r') as f:
    with open(META_TRAIN_OUT, 'w') as fout:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            key = line.split('-')[0]
            new_line = line.split("|") + metadata[key].split('|')[2:]
            fout.write("|".join(new_line) + '\n')


with open(META_TEST_IN, 'r') as f:
    with open(META_TEST_OUT, 'w') as fout:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            key = line.split('-')[0]
            new_line = line.split("|") + metadata[key].split('|')[2:]
            fout.write("|".join(new_line) + '\n')

