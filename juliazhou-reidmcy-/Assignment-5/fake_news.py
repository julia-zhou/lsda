#juliazhou-reidmcy a5

import argparse
import chainer
from chainer import cuda, utils, Variable
import chainer.functions as F
import chainer.links as L
import cPickle as pickle
import json
import numpy as np
import pandas as pd
import random
import re
import string
import sys
import math
import unicodedata
import datetime
from collections import Counter
fname = '/project/cmsc25025/uci-news-aggregator/{cat}_article.json'

"""# Notes

We have made two major changes made to to the RNN. First a third layer with peephole LSTM was added, this layer is more effected by the LSTM 'highway' so it adds more statefulness than a normal LSTM layer. The second major change was to add non-linearities to the net for all the inner layers. Hyperbolic tan was choosen over ReLu and sigmoid as that is what the literature suggests. We tried adding softmax to the output layer, but the performance decreased significantly very well with out it. We also removed dropout from all but the first two layers, as excess dropout can reduce the training rate. The final net was trained with 256 nodes at each hidden layer.

The submitted file uses character level vocab since that is faster to train. It shows understanding of grammar, English spelling and some sentences make sense. e.g. 

'announced it has paid through several charges include 5:20 passengers approximately 22,000 on the leftover with a deal before the bank of greater suniticia warned that they have 45.5 million placents to watch greenhouse gases emitted by loss-cash interest rates.'

We think this is good for a character level network, we would have liked to test for longer but getting GPU access on RCC was tricky and we were only able to test a couple models. You will note this file outputs a text file every 10 epochs so if running it you will get lots of files, named after their starting time.

The final losses and perplexity were at epoch 128, at which point it had converged:
* Epoch 128 train: loss=0.938773393631 perp=2.55684325386 valid: loss=1.77523815632 perp=5.90168649289

but the lowest testing perplexity (~4.77501639546) was encountered before this around epoch 20, but we let it over fit since  the subjective quality kept improving, which is not suprising. An epoch 20 sentence looks like:

'arabga and oto beef it fisher nearly 16 financm and homewitera, condensate forecasts.'
"""


class FooRNN(chainer.Chain):
    """Three Layer LSTM with tanh activations and lower level dropout"""
    def __init__(self, n_vocab, n_units, train=True):
        super(FooRNN, self).__init__(  # input must be a link
            embed=L.EmbedID(n_vocab, n_units),
            l1=L.LSTM(n_units, n_units),
            l2=L.LSTM(n_units, n_units),
            l3=L.StatefulPeepholeLSTM(n_units, n_units),
            l4=L.Linear(n_units, n_vocab),
        )
        self.n_vocab = n_vocab
        self.n_units = n_units
        self.train = train

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0, train=self.train))
        h2 = self.l2(F.tanh(F.dropout(h1, train=self.train)))
        h3 = self.l3(F.tanh(h2))
        y = self.l4(F.tanh(h3))
        return y

def read_data(category='b', unit='char', thresh=50):
    raw_doc = []
    with open(fname.format(cat=category), 'r') as f:
        for line in f.readlines():
            text = json.loads(line)['text']
            if len(text.split()) >= 100:
                raw_doc.append(
                    unicodedata.normalize('NFKD', text)
                    .encode('ascii', 'ignore').lower()
                    .translate(string.maketrans("\n", " "))
                    .strip()
                )
    raw_doc = ' '.join(raw_doc)

    if unit == 'char':
        vocab = {el: i for i, el in enumerate(set(raw_doc))}
        id_to_word = {i: el for el, i in vocab.iteritems()}
    else: # unit == 'word':
        raw_doc = re.split('(\W+)', raw_doc)
        count = Counter(raw_doc)

        vocab = {}
        ii = 0
        for el in count:
            if count[el] >= thresh:
                vocab[el] = ii
                ii += 1

        id_to_word = {i: el for el, i in vocab.iteritems()}


    doc = [vocab[el] for el in raw_doc if el in vocab]
    print '  * doc length: {}'.format(len(doc))
    print '  * vocabulary size: {}'.format(len(vocab))
    sys.stdout.flush()

    return doc, vocab, id_to_word


def convert(data, batch_size, ii, gpu_id=-1):
    xp = np if gpu_id < 0 else cuda.cupy
    offsets = [t * len(data) // batch_size for t in xrange(batch_size)]
    x = [data[(offset + ii) % len(data)] for offset in offsets]
    x_in = chainer.Variable(xp.array(x, dtype=xp.int32))
    y = [data[(offset + ii + 1) % len(data)] for offset in offsets]
    y_in = chainer.Variable(xp.array(y, dtype=xp.int32))
    return x_in, y_in


def gen_text(model, curr, id_to_word, text_len, gpu_id=-1):
    xp = np if gpu_id < 0 else cuda.cupy

    n_vocab = len(id_to_word)
    gen = [id_to_word[curr]] * text_len
    model.predictor.reset_state()
    for ii in xrange(text_len):
        output = model.predictor(
            chainer.Variable(xp.array([curr], dtype=xp.int32))
        )
        p = F.softmax(output).data[0]
        if gpu_id >= 0:
            p = cuda.to_cpu(p)
        curr = np.random.choice(n_vocab, p=p)
        gen[ii] = id_to_word[curr]

    return ''.join(gen)


def main():
    gpu_id = 0
    n_epoch = 150
    train_len = 6000000
    valid_len = 50000
    batch_size = 2048
    nu = 256
    thresh = 50
    n_text = 10
    out_len = 1000

    outType = 'b'
    unit = 'char'

    outputName = datetime.datetime.now().strftime('%H-%M-%S')

    print "{} starting run".format(outputName)

    print "loading doc...."
    sys.stdout.flush()

    doc, vocab, id_to_word = read_data(
      category=outType, unit=unit, thresh = thresh
    )
    n_vocab = len(vocab)

    if train_len + valid_len > len(doc):
        raise Exception(
            'train len {} + valid len {} > doc len {}'.format(
                train_len, valid_len, len(doc)
            )
        )
    train = doc[:train_len]
    valid = doc[(train_len+1):(train_len+1+valid_len)]

    print "initializing...."
    sys.stdout.flush()
    model = L.Classifier(FooRNN(n_vocab, nu, train=True))
    sys.stdout.flush()
    model.predictor.reset_state()
    #optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(100))

    if gpu_id >= 0:
        cuda.get_device(gpu_id).use()
        model.to_gpu()

    # main training loop
    print "training loop...."
    sys.stdout.flush()
    xp = np if gpu_id < 0 else cuda.cupy
    for t in xrange(n_epoch):
        train_loss = train_acc = n_batches = loss = 0
        model.predictor.reset_state()
        for i in range(0, len(train) // batch_size + 1):
            x, y = convert(train, batch_size, i, gpu_id)
            batch_loss = model(x, y)
            loss += batch_loss
            if (i+1) % min(len(train) // batch_size, 30) == 0:
                model.cleargrads()
                loss.backward()
                loss.unchain_backward()
                optimizer.update()
            train_loss += batch_loss.data
            n_batches += 1
        train_loss = train_loss / n_batches
        train_acc = train_acc / n_batches

        # validation
        valid_loss = valid_acc = n_batches = 0
        for i in range(0, len(valid) // batch_size + 1):
            x, y = convert(valid, batch_size, i, gpu_id)
            batch_loss = model(x, y)
            valid_loss += batch_loss.data
            n_batches += 1
        valid_loss = valid_loss / n_batches
        valid_acc = valid_acc / n_batches

        print '  * Epoch {} train: loss={} perp={} valid: loss={} perp={}'.format(
            t,
            train_loss,
            math.exp(train_loss), #since loss is cross entropy
            valid_loss,
            math.exp(valid_loss), #since loss is cross entropy
        )
        sys.stdout.flush()

        if t >= 1 and xp.abs(train_loss - old_tr_loss) / train_loss < 1e-5:
            print "Converged."
            sys.stdout.flush()
            break

        old_tr_loss = train_loss

        if t % 10 == 0 and t > 1:
            print "generating doc {}....".format(t)
            sys.stdout.flush()
            model.predictor.train = False
            outputFile = "{}-{}.txt".format(outputName, t)
            with open(outputFile, 'w') as f:
                for ii in xrange(n_text):
                    start = random.choice(xrange(len(vocab)))
                    fake_news = gen_text(
                        model,
                        start,
                        id_to_word,
                        text_len=out_len,
                        gpu_id=gpu_id
                    )
                    f.write(fake_news)
                    f.write('\n\n\n')

    print "generating doc final....".format(t)
    sys.stdout.flush()
    model.predictor.train = False
    outputFile = "{}-final.txt".format(outputName, t)
    with open(outputFile, 'w') as f:
        for ii in xrange(n_text):
            start = random.choice(xrange(len(vocab)))
            fake_news = gen_text(
                model,
                start,
                id_to_word,
                text_len=out_len,
                gpu_id=gpu_id
            )
            f.write(fake_news)
            f.write('\n\n\n')

    if gpu_id >= 0:
        model.to_cpu()
    with open('model_%s.pickle' % outputName, 'wb') as f:
        pickle.dump(model, f, protocol=2)
    print "Done"

if __name__ == '__main__':
    main()
