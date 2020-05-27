#!/bin/sh

case "$1" in
"LSTMTransformer")
  echo "Generate text by LSTM+Transformer model."
  python -O src/generate.py \
    --checkpoint models/lstmtransformer.pth \
    --spm-path models/sp_8000.model \
    --outf result/generated_lstmtransformer.txt \
    --words 1000 \
    --bptt 64 \
    --temperature 1.0 \
    --cuda
  ;;
"Transformer")
  echo "Generate text by Transformer model."
  python -O src/generate.py \
    --checkpoint models/transformer.pth \
    --spm-path models/sp_8000.model \
    --outf result/generated_transformer.txt \
    --words 1000 \
    --bptt 64 \
    --temperature 1.0 \
    --cuda
  ;;
"LSTM")
  echo "Generate text by LSTM model."
  python -O src/generate.py \
    --checkpoint models/lstm.pth \
    --spm-path models/sp_8000.model \
    --outf result/generated_lstm.txt \
    --words 1000 \
    --temperature 1.0 \
    --cuda
  ;;
*)
  echo "Usage:"
  echo "./generate.sh {LSTMTransformer,Transformer,LSTM}"
  ;;
esac
