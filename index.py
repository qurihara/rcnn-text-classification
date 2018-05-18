# -*- coding: utf-8 -*-
from bottle import route, run, template
import urllib.parse

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

import argparse
import sys, time

import chainer
import chainer.optimizers
import chainer.serializers
import chainer.functions as F
from chainer import Variable
from chainer import cuda
import numpy as np
from model import LetterClassifyer

# ファイルから1文字単位の列とラベルを取得
def letter_list(fname):
    with open(fname,'r',encoding="utf-8_sig") as f:
        for l in f:
            body = l[:-3]
            val = int(l[-2])
            x = list(''.join(body.split()))
            x.append('</s>')
            yield x, val
def letter_list_text(t):
    x = list(''.join(t.split()))
    x.append('</s>')
    return x

class Vocabulary:
    def __init__(self, fname):
        self.fname = fname
        self.l2i = {}
        self.i2l = []
        if not fname is None:
            self.load_vocab()
    def stoi(self, letter):
        if letter in self.l2i:
            return self.l2i[letter]
        return self.l2i['<unk>']
    def itos(self, id):
        if id < len(self.i2l):
            return self.i2l[id]
        return '<unk>'

    def append_letter(self, l):
        if l in self.l2i:
            return
        self.i2l.append(l)
        id = len(self.i2l) -1
        self.l2i[l] = id
    def load_vocab(self):
        self.append_letter('<unk>')
        self.append_letter('<s>')
        self.append_letter('</s>')
        with open(self.fname,'r',encoding="utf-8_sig") as f:
            for line in f:
                nline = line[:-3]
                for l in nline:
                    self.append_letter(l)

    def save_vocab(self, filename):
        with open(filename, 'w',encoding="utf-8_sig") as f:
            for l in self.i2l:
                f.write(l + "\n")
    @staticmethod
    def load_from_file(filename):
        vocab = Vocabulary(None)
        with open(filename,'r',encoding="utf-8_sig") as f:
            for l in f:
                l = l[:-1]
                vocab.append_letter(l)
        return vocab

def forward(src_batch, t, model, is_training, vocab, xp):
    batch_size = len(src_batch)
    src_len = len(src_batch[0])
    src_stoi = vocab.stoi
    x_batch = [Variable(xp.asarray([[src_stoi(x)]], dtype=xp.int32)) for x in src_batch[0]]
    y = model.forward(x_batch)
    if is_training:
        t = Variable(xp.asarray([t], dtype=xp.int32))
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        return y, acc, loss
    else:
        return y


def eval(name):
    xp = cuda.cupy
    cuda.get_device(0).use()
    vocab = Vocabulary.load_from_file("model.vocab")
    m = LetterClassifyer(3000, 200, 1000, 2)
    chainer.serializers.load_hdf5("model.hdf5", m)
    m.to_gpu()
    x_batch = [letter_list_text(name)]
    output = forward(x_batch, None, m, False, vocab, xp)
    return np.argmax(output.data)

@route('/dazai_akuta/<name>')
def index(name):
    txt = urllib.parse.unquote(name)
    res = eval(txt)
    print(txt)
    mes = "dazai" if res == "0" else "akutagawa" 
    return template('<b>{{name}}</b>', name=mes)

run(host='localhost', port=8080)
