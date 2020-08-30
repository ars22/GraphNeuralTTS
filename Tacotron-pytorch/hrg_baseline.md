## Notes on the first HRG baseline

### Issues

- [ ] Too many open files, probable cause is collating/dataset. [link](https://stackoverflow.com/questions/48250053/pytorchs-dataloader-too-many-open-files-error-when-no-files-should-be-open)

### Todo
- [x] Training end to end with real HRGs
- [ ] Use two level HRG graphs

### Running
- Dataset metadata_hrg_2.csv: https://drive.google.com/file/d/1d7oXSvgmQBTWs5_yOKxNRVIZHg3Sc4rd/view?usp=sharing

- Commands
``sh
python data/preprocess.py --output-dir training-hrg/ --data-dir ${PATH_TO_DATA}/LJSpeech-1.1/wavs/ --old-meta ${PATH_TO_DATA}/LJSpeech-1.1/metadata_hrg_2.csv --config config/config.yaml

python data/train_test_split.py --meta-all training-hrg/ljspeech_meta.txt  --ratio-test 0.1

python -u main.py --config config/config-hrg.yaml  --checkpoint-dir checkpoint
``
