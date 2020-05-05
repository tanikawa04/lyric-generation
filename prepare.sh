#!/bin/sh

RAW_PATH=data/lyric/raw/
PRETOKENIZED_PATH=data/lyric/pretokenized/
VOCAB_SIZE=8000
SPM_PREFIX=models/$VOCAB_SIZE

echo "Pre-tokenize corpus."
python src/preprocess/pretokenize.py \
  $RAW_PATH \
  $PRETOKENIZED_PATH

echo "SentencePiece model training."
python src/preprocess/spm_train.py \
  $PRETOKENIZED_PATH \
  $SPM_PREFIX \
  --vocab-size $VOCAB_SIZE

echo "Convert corpus to PyTorch tensor."
python src/preprocess/make_dataset.py \
  $RAW_PATH \
  data/lyric/tensor/ \
  --sp-model-path "$SPM_PREFIX.model"
