# -*- coding: utf-8 -*-
from bottle import route, run, template

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
    res = eval(name)
    return template('<b>{{name}}</b>', name=res)

run(host='localhost', port=8080)
