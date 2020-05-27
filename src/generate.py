import click
import torch
from tokenizer import Tokenizer
import data


@click.command()
@click.option('--checkpoint', type=str, default='./model.pt',
              help='model checkpoint to use')
@click.option('--spm-path', type=str, default='../models/sp_8000.model',
              help='SentencePiece model path')
@click.option('--outf', type=str, default='generated.txt',
              help='output file for generated text')
@click.option('--words', 'n_words', type=int, default='1000',
              help='max number of words to generate')
@click.option('--bptt', type=int, default=35,
              help='sequence length')
@click.option('--seed', type=int, default=None,
              help='random seed')
@click.option('--cuda', 'use_cuda', is_flag=True,
              help='use CUDA')
@click.option('--temperature', type=float, default=1.0,
              help='temperature - higher will increase diversity')
def main(checkpoint, spm_path, outf, n_words, bptt, seed, use_cuda, temperature):
    if seed:
        torch.manual_seed(seed)

    if torch.cuda.is_available():
        if not use_cuda:
            print('WARNING: You have a CUDA device, so you should probably run with --cuda')
    device = torch.device('cuda' if use_cuda else 'cpu')

    if temperature < 1e-3:
        parser.error('--temperature has to be greater or equal 1e-3')

    tokenizer = Tokenizer('models/sp_8000.model')

    with open(checkpoint, 'rb') as f:
        model = torch.load(f).to(device)
    model.eval()

    model_type = model.model_type if hasattr(model, 'model_type') else None

    if model_type == 'LSTMTransformer':
        hidden = model.init_hidden(1)
        mems = None
    elif model_type == 'Transformer':
        pass
    else:
        hidden = model.init_hidden(1)

    input = torch.tensor([[1]], dtype=torch.long).to(device)

    s = []
    with torch.no_grad():  # no tracking history
        for i in range(n_words):
            if model_type == 'LSTMTransformer':
                output, hidden, mems = model(input, hidden, mems)
                word_weights = output[-1].squeeze().div(temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                input.fill_(word_idx)
            elif model_type == 'Transformer':
                output = model(input, False)
                word_weights = output[-1].squeeze().div(temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                input = torch.cat([input, word_tensor], 0)[-bptt:]
            else:
                output, hidden = model(input, hidden)
                word_weights = output.squeeze().div(temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                input.fill_(word_idx)

            s.append(int(word_idx))
            if word_idx == 2:
                break

    txt = tokenizer.decode(s)
    with open(outf, 'w') as f:
        f.write(txt)

    print(txt)


if __name__ == '__main__':
    main()
