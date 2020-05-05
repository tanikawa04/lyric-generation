from sudachipy import tokenizer, dictionary
import sentencepiece as spm


control_symbols = {
    '\n': '<br>',
    ' ': '<nbsp>'
}


def to_symbol(token):
    return control_symbols.get(token, token)


def desymbolize(txt):
    _txt = txt
    for k, v in control_symbols.items():
        _txt = _txt.replace(v, k)
    return _txt


class Tokenizer:
    def __init__(self, sp_model_path, bos_eos=True):
        self._sudachi_tokenizer = dictionary.Dictionary().create()
        self._sudachi_mode = tokenizer.Tokenizer.SplitMode.A
        self._sp_tokenizer = spm.SentencePieceProcessor()
        self._sp_tokenizer.load(sp_model_path)
        if bos_eos:
            self._sp_tokenizer.set_encode_extra_options('bos:eos')

    def tokenize(self, txt):
        # word tokenize
        pretokenized_txt = ''
        for line in txt.split('\n'):
            _line = line.strip()
            if len(_line) > 0:
                pretokens = [m.surface() for m in self._sudachi_tokenizer.tokenize(_line, self._sudachi_mode)]
                pretokens = [to_symbol(token) for token in pretokens]
                pretokenized_txt += ' '.join(pretokens)
            pretokenized_txt += '<br>'

        # subword tokenize
        tokens = self._sp_tokenizer.encode_as_pieces(pretokenized_txt)

        if tokens[-1] == '<br>':
            tokens = tokens[:-1]
        return tokens

    def detokenize(self, tokens):
        txt = self._sp_tokenizer.decode_pieces(tokens)
        txt = txt.replace(' ', '')
        txt = desymbolize(txt)
        return txt

    def encode(self, txt):
        tokens = self.tokenize(txt)
        ids = [self._sp_tokenizer.piece_to_id(token) for token in tokens]
        return ids

    def decode(self, ids):
        tokens = [self._sp_tokenizer.id_to_piece(i) for i in ids]
        txt = self.detokenize(tokens)
        return txt

    def to_id(self, piece):
        return self._sp_tokenizer.piece_to_id(piece)

    def to_piece(self, i):
        return self._sp_tokenizer.id_to_piece(i)

    def size(self):
        return self._sp_tokenizer.get_piece_size()
