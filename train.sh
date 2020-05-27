#!/bin/sh

case "$1" in
"LSTMTransformer")
  echo "Train LSTM+Transformer model."
  python -O src/train.py \
    --data data/lyric/tensor \
    --model LSTMTransformer \
    --emsize 512 \
    --nhid 1024 \
    --nlayers 5 \
    --nhead 8 \
    --clip 5.0 \
    --epochs 10 \
    --batch-size 16 \
    --bptt 64 \
    --dropout 0.1 \
    --seed 1111 \
    --cuda \
    --spm-path models/sp_8000.model \
    --val-interval 1000 \
    --log-interval 100 \
    --tb-log logs/lstmtransformer \
    --save models/lstmtransformer.pth
  ;;
"Transformer")
  echo "Train Transformer model."
  python -O src/train.py \
    --data data/lyric/tensor \
    --model Transformer \
    --emsize 512 \
    --nhid 1024 \
    --nlayers 6 \
    --nhead 8 \
    --clip 5.0 \
    --epochs 10 \
    --batch-size 16 \
    --bptt 64 \
    --dropout 0.1 \
    --seed 1111 \
    --cuda \
    --spm-path models/sp_8000.model \
    --val-interval 1000 \
    --log-interval 100 \
    --save models/transformer.pth \
    --tb-log logs/transformer
  ;;
"LSTM")
  echo "Train LSTM model."
  python -O src/train.py \
    --data data/lyric/tensor \
    --model LSTM \
    --emsize 512 \
    --nhid 512 \
    --nlayers 3 \
    --clip 5.0 \
    --epochs 10 \
    --batch-size 16 \
    --bptt 64 \
    --dropout 0.2 \
    --seed 1111 \
    --cuda \
    --spm-path models/sp_8000.model \
    --val-interval 1000 \
    --log-interval 100 \
    --save models/lstm.pth \
    --tb-log logs/lstm
  ;;
*)
  echo "Usage:"
  echo "./train.sh {LSTMTransformer,Transformer,LSTM}"
  ;;
esac
