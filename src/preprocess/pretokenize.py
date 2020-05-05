import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import glob
import click
from sudachipy import tokenizer, dictionary

# tokenizer.py と一部重複してしまっているので冗長


class PreTokenizer:
    def __init__(self):
        self._tokenizer = dictionary.Dictionary().create()
        self._mode = tokenizer.Tokenizer.SplitMode.B

    def __call__(self, s):
        return [m.surface() for m in self._tokenizer.tokenize(s, self._mode)]


def to_symbol(token):
    control_symbols = {
        '\n': '<br>',
        ' ': '<nbsp>'
    }
    return control_symbols.get(token, token)


@click.command()
@click.argument('input_path')
@click.argument('output_path')
def main(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    pre_tokenizer = PreTokenizer()

    input_file_paths = glob.glob(os.path.join(input_path, '*.txt'))
    for i, input_file_path in enumerate(input_file_paths):
        print(f'\r{i + 1} / {len(input_file_paths)}', end='')

        file_name = os.path.basename(input_file_path)
        output = ''
        with open(input_file_path) as f:
            for line in f:
                _line = line.strip()
                if len(_line) > 0:
                    tokens = pre_tokenizer(_line)
                    tokens = [to_symbol(token) for token in tokens]
                    output += ' '.join(tokens) + '\n'
                else:
                    output += '\n'

        with open(os.path.join(output_path, file_name), 'w') as f:
            f.write(output)

    print()
    print('done.')


if __name__ == '__main__':
    main()
