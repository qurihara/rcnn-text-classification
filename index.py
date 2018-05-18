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

def argument():
    parser = argparse.ArgumentParser()
#    parser.add_argument('mode')
#    parser.add_argument('file')
    parser.add_argument('--embed', default=200, type=int)
    parser.add_argument('--vocab', default=3000, type=int)
    parser.add_argument('--hidden', default=1000, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--model', default="model")
    parser.add_argument('--classes', default=2, type=int)
    parser.add_argument('--use-gpu', action='store_true', default=False)
    parser.add_argument('--unchain', action='store_true', default=False)
    args = parser.parse_args()
    return args

def eval(name):
    args = argument()
    if args.use_gpu:
        xp = cuda.cupy
        cuda.get_device(0).use()
    else:
        xp = np
    vocab = Vocabulary.load_from_file("%s.vocab" % args.model)
    m = LetterClassifyer(args.vocab, args.embed, args.hidden, args.classes)
    chainer.serializers.load_hdf5("%s.hdf5" % args.model, m)
    if args.use_gpu:
        m.to_gpu()
    x_batch = [letter_list_text(name)]
    output = forward(x_batch, None, m, False, vocab, xp)
    return np.argmax(output.data)

@route('/dazai_akuta/<name>')
def index(name):
    res = eval(name)
    return template('<b>{{name}}</b>', name=res)

run(host='localhost', port=8080)
