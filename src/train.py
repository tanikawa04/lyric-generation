import os
from functools import partial
import math
import click
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from ignite.engine import Engine, Events
from data import load, BucketBatchSampler
from model import RNNModel, TransformerModel, LSTMTransformerModel, TransformerLR, repackage_hidden
from tokenizer import Tokenizer


def get_loader(data_dir, batch_size, padding_value=-1):
    train_data = load(data_dir)
    train_sampler = BucketBatchSampler(train_data, batch_size, False, sort_key=lambda i: len(train_data[i]))
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_sampler=train_sampler,
        collate_fn=partial(torch.nn.utils.rnn.pad_sequence, padding_value=padding_value))

    return train_loader


@click.command()
@click.option('--data', 'data_dir', type=str, default='../data/tokenized',
              help='location of the data corpus')
@click.option('--model', 'model_type', type=str, default='LSTM',
              help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer, LSTMTransformer)')
@click.option('--emsize', type=int, default=200,
              help='size of word embeddings')
@click.option('--nhid', type=int, default=200,
              help='number of hidden units per layer')
@click.option('--nlayers', type=int, default=2,
              help='number of layers')
@click.option('--nhead', type=int, default=2,
              help='the number of heads in the transformer model')
@click.option('--warm-up', type=int, default=4000,
              help='number of steps to warm up')
@click.option('--step-size', type=int, default=1,
              help='period of learning rate decay')
@click.option('--clip', type=float, default=5.0,
              help='gradient clipping')
@click.option('--epochs', type=int, default=40,
              help='upper epoch limit')
@click.option('--batch-size', type=int, default=20,
              help='batch size')
@click.option('--bptt', type=int, default=35,
              help='sequence length')
@click.option('--dropout', type=float, default=0.2,
              help='dropout applied to layers (0 = no dropout)')
@click.option('--tied', is_flag=True,
              help='tie the word embedding and softmax weights')
@click.option('--seed', type=int, default=1111,
              help='random seed')
@click.option('--cuda', 'use_cuda', is_flag=True,
              help='use CUDA')
@click.option('--spm-path', type=str, default='../models/sp_8000.model',
              help='SentencePiece model path')
@click.option('--log-interval', type=int, default=200,
              help='report interval')
@click.option('--val-interval', type=int, default=5000,
              help='validation interval')
@click.option('--tb-log', type=str, default='logs',
              help='path to save tensorboard log')
@click.option('--save', 'save_path', type=str, default='model.pt',
              help='path to save the final model')
def main(data_dir, model_type, emsize, nhid, nlayers, nhead, warm_up, step_size,
         clip, epochs, batch_size, bptt, dropout, tied, seed, use_cuda, spm_path,
         log_interval, val_interval, tb_log, save_path):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        if not use_cuda:
            print('WARNING: You have a CUDA device, so you should probably run with --cuda')
    device = torch.device('cuda' if use_cuda else 'cpu')

    tokenizer = Tokenizer(spm_path, bos_eos=True)
    ntokens = tokenizer.size()
    padding_index = tokenizer.to_id('<pad>')
    train_loader = get_loader(os.path.join(data_dir, 'train'), batch_size, padding_value=padding_index)

    if model_type == 'LSTMTransformer':
        model = LSTMTransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    elif model_type == 'Transformer':
        model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    else:
        model = RNNModel(model_type, ntokens, emsize, nhid, nlayers, dropout, tied).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1)
    scheduler = TransformerLR(optimizer, emsize, warmup_steps=warm_up, step_size=step_size)

    def create_supervised_trainer(model, optimizer, device=None):
        model_type = model.model_type if hasattr(model, 'model_type') else None
        criterion = nn.NLLLoss(ignore_index=padding_index)

        def _update(engine, batch):
            model.train()
            optimizer.zero_grad()

            _batch = batch.to(device)
            data, targets = _batch[:-1], _batch[1:].view(-1)
            batch_size = batch.size(1)

            if model_type != 'Transformer':
                hidden = model.init_hidden(batch_size)

            if model_type == 'LSTMTransformer':
                hidden = repackage_hidden(hidden)
                output, hidden, _ = model(data, hidden)
                output = output.view(-1, ntokens)
            elif model_type == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                hidden = repackage_hidden(hidden)
                output, hidden = model(data, hidden)

            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            loss = loss.item()
            ppl = math.exp(loss)
            return {'loss': loss, 'ppl': ppl}

        trainer = Engine(_update)
        return trainer

    writer = SummaryWriter(log_dir=tb_log)
    trainer = create_supervised_trainer(model, optimizer, device=device)

    @trainer.on(Events.ITERATION_COMPLETED(every=1))
    def log_training_loss(engine):
        writer.add_scalar('train/loss', engine.state.output['loss'], engine.state.iteration)
        writer.add_scalar('train/ppl', engine.state.output['ppl'], engine.state.iteration)

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def print_training_loss(engine):
        print('Epoch[{}] Iteration[{}/{}] Loss: {:.2f} PPL: {:.2f}'
              ''.format(engine.state.epoch, engine.state.iteration, len(train_loader),
                        engine.state.output['loss'], engine.state.output['ppl']))

    @trainer.on(Events.ITERATION_COMPLETED(every=val_interval))
    def print_validation_results(engine):
        loss, ppl = 0, 0
        print('Validation Results - Epoch[{}] Loss: {:.2f} PPL: {:.2f}'
              ''.format(engine.state.epoch, loss, ppl))

    trainer.run(train_loader, max_epochs=epochs)

    writer.close()


if __name__ == '__main__':
    main()
