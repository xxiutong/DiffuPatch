# Diffu-Patch

## Diffu-Patch: Denoising Diffusion Probabilistic Model for Learning Bug-Fixing Patches

## Python library dependencies:
+ bert_score
+ blobfile
+ nltk
+ numpy
+ packaging
+ psutil
+ PyYAML
+ setuptools
+ spacy
+ torch==1.9.0+cu111
+ torchmetrics
+ tqdm
+ transformers==4.22.2
+ wandb
+ datasets

---

## Dataset:

Prepare datasets and put them under the datasets folder.The two datasets we used are placed under the `datasets` folder, namely `datasets/bugfix` and `datasets/bugfixlen`. Corresponding vocabulary files are placed in their respective folders, named `vocab.txt` and `vocablen.txt`.

---
## Diffu-Patch Training

```bash
cd scripts
bash train.sh
```

Arguments explanation:
- ```--dataset```: the name of datasets, just for notation
- ```--data_dir```: the path to the saved datasets folder, containing ```train.jsonl,test.jsonl,valid.jsonl```
- ```--seq_len```: the max length of sequence $z$ ($x\oplus y$)
- ```--resume_checkpoint```: if not none, restore this checkpoint and continue training
- ```--vocab```: the tokenizer is initialized using bert or load your own preprocessed vocab dictionary (e.g. using BPE or our vocab)

Additional argument:
- ```--learned_mean_embed```: set whether to use the learned soft absorbing state.
- ```--denoise```: set whether to add discrete noise
- ```--use_fp16```: set whether to use mixed precision training
- ```--denoise_rate```: set the denoise rate, with 0.5 as the default
---

## Diffu-Patch Detecting
```bash
cd scripts
bash run_decode.sh
```

## Diffu-Patch Speed-up Detecting
```bash
cd scripts
bash run_decode_solver.sh
```













