import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import glob
import click
import sentencepiece as spm


@click.command()
@click.argument('input_path')
@click.argument('output_prefix')
@click.option('--vocab-size', type=int, default=8000)
def main(input_path, output_prefix, vocab_size):
    input_file_paths = ','.join(glob.glob(os.path.join(input_path,'*.txt')))
    spm.SentencePieceTrainer.Train(
        f'--input={input_file_paths} '
        f'--model_prefix={output_prefix} '
        f'--vocab_size={vocab_size} '
        f'--user_defined_symbols=<br>,<nbsp>,<pad>')


if __name__ == '__main__':
    main()
