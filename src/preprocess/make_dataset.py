import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import glob
import random
import click
import torch
from tokenizer import Tokenizer


@click.command()
@click.argument('input_path')
@click.argument('output_path')
@click.option('--sp-model-path', type=str, default='../../models/sp_8000.model',
              help='SentencePiece model path')
@click.option('--n-val', type=int, default=100,
              help='number of validation instance')
@click.option('--n-test', type=int, default=100,
              help='number of test instance')
@click.option('--seed', type=int, default=1111,
              help='random seed')
def main(input_path, output_path, sp_model_path, n_val, n_test, seed):
    tokenizer = Tokenizer(sp_model_path, bos_eos=True)

    train_dir = os.path.join(output_path, 'train')
    val_dir = os.path.join(output_path, 'val')
    test_dir = os.path.join(output_path, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    input_file_paths = sorted(glob.glob(os.path.join(input_path, '*.txt')))
    random.seed(seed)
    random.shuffle(input_file_paths)

    for i, input_file_path in enumerate(input_file_paths):
        print(f'\r{i + 1} / {len(input_file_paths)}', end='')

        file_name = os.path.basename(input_file_path)
        with open(input_file_path) as f:
            tids = tokenizer.encode(f.read())
        tids = torch.tensor(tids, dtype=torch.long)

        if i < n_val:
            torch.save(tids, os.path.join(val_dir, file_name.replace('.txt', '.pt')))
        elif n_val <= i < n_val + n_test:
            torch.save(tids, os.path.join(test_dir, file_name.replace('.txt', '.pt')))
        else:
            torch.save(tids, os.path.join(train_dir, file_name.replace('.txt', '.pt')))

    print()
    print('done.')


if __name__ == '__main__':
    main()
