#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 4
run.py: Run Script for Simple NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>

Usage:
    run.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 16]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --input-feed                            use input feeding
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --sample-size=<int>                     sample size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model/model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.3]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""
import math
import sys
import pickle
import time


from docopt import docopt
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nmt_model import Hypothesis, NMT
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import read_corpus, batch_iter
from vocab import Vocab, VocabEntry
import os
import tensorflow as tf
from datetime import datetime
from collections import namedtuple
tf.reset_default_graph()
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])
import torch
import torch.nn.utils


def evaluate_ppl(model, sess, dev_data, batch_size=32):
    """ Evaluate perplexity on dev sentences
    @param model (NMT): NMT Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """

    cum_loss = 0.
    cum_tgt_words = 0.
    # 将输入节点取出
    source = tf.get_default_graph().get_tensor_by_name('source:0')
    target = tf.get_default_graph().get_tensor_by_name('target:0')
    source_lengths = tf.get_default_graph().get_tensor_by_name('source_lengths:0')
    target_lengths = tf.get_default_graph().get_tensor_by_name('target_lengths:0')
    batch_loss = tf.get_default_graph().get_tensor_by_name('batch_loss:0')



    # no_grad() signals backend to throw away all gradients
    for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
        source_batch, source_lengths_batch = model.vocab.src.to_input_numpy(src_sents, True)  # numpy: (b, src_len)
        target_batch, target_lengths_batch = model.vocab.tgt.to_input_numpy(tgt_sents, False)  # numpy: (b, tgt_len)

        loss = sess.run(batch_loss, feed_dict={source: source_batch, target: target_batch,
                                               source_lengths: source_lengths_batch, target_lengths: target_lengths_batch})

        cum_loss += loss
        tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
        cum_tgt_words += tgt_word_num_to_predict

    ppl = np.exp(cum_loss / cum_tgt_words)

    return ppl


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
    @param references (List[List[str]]): a list of gold-standard reference target sentences
    @param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
    @returns bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]
    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])
    return bleu_score


def train(args: Dict):
    """ Train the NMT Model.
    @param args (Dict): args from cmd line
    """
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])

    output_dir = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
    output_path = output_dir + "model.ckpt"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_save_path = output_path

    vocab = Vocab.load(args['--vocab'])

    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                vocab=vocab)

    """
    uniform_init = float(args['--uniform-init'])
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)
    """

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    # 建立静态图, forward为前向传播，其中建立了encode以及decode的静态图
    example_loss = -model.forward()
    static_batch_loss = tf.reduce_sum(example_loss, name='batch_loss')
    static_loss = tf.reduce_mean(example_loss, name='loss')

    learning_rate = tf.Variable(float(args['--lr']), trainable=False, dtype=tf.float32, name='learning_rate')
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(static_loss)
    tf.add_to_collection('train_step', train_step)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        while True:
            epoch += 1

            for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
                train_iter += 1

                batch_size = len(src_sents)

                source_batch, source_lengths_batch = model.vocab.src.to_input_numpy(src_sents, True)  # numpy: (b, src_len)
                target_batch, target_lengths_batch = model.vocab.tgt.to_input_numpy(tgt_sents, False)  # numpy: (b, tgt_len)

                # 将输入节点取出
                source = tf.get_default_graph().get_tensor_by_name('source:0')
                target = tf.get_default_graph().get_tensor_by_name('target:0')
                source_lengths = tf.get_default_graph().get_tensor_by_name('source_lengths:0')
                target_lengths = tf.get_default_graph().get_tensor_by_name('target_lengths:0')

                batch_loss, loss, _ = sess.run([static_batch_loss, static_loss, train_step],
                                                  feed_dict={source: source_batch, target: target_batch,
                                                             source_lengths: source_lengths_batch, target_lengths: target_lengths_batch})

                report_loss += batch_loss
                cum_loss += batch_loss

                tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
                report_tgt_words += tgt_words_num_to_predict
                cum_tgt_words += tgt_words_num_to_predict
                report_examples += batch_size
                cum_examples += batch_size

                if train_iter % log_every == 0:
                    print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                          'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                             report_loss / report_examples,
                                                                                             math.exp(report_loss / report_tgt_words),
                                                                                             cum_examples,
                                                                                             report_tgt_words / (time.time() - train_time),
                                                                                             time.time() - begin_time), file=sys.stderr)

                    train_time = time.time()
                    report_loss = report_tgt_words = report_examples = 0.

                # perform validation
                if train_iter % valid_niter == 0:
                    print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                             cum_loss / cum_examples,
                                                                                             np.exp(cum_loss / cum_tgt_words),
                                                                                             cum_examples), file=sys.stderr)

                    cum_loss = cum_examples = cum_tgt_words = 0.
                    valid_num += 1

                    print('begin validation ...', file=sys.stderr)

                    # compute dev. ppl and bleu
                    dev_ppl = evaluate_ppl(model, sess, dev_data, batch_size=16)   # dev batch size can be a bit larger
                    valid_metric = -dev_ppl

                    print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                    is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                    hist_valid_scores.append(valid_metric)

                    if is_better:
                        patience = 0
                        print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                        model.save(model_save_path, sess)
                    elif patience < int(args['--patience']):
                        patience += 1
                        print('hit patience %d' % patience, file=sys.stderr)

                        if patience == int(args['--patience']):
                            num_trial += 1
                            print('hit #%d trial' % num_trial, file=sys.stderr)
                            if num_trial == int(args['--max-num-trial']):
                                print('early stop!', file=sys.stderr)
                                exit(0)

                            # decay lr, and restore from previously best checkpoint
                            learning_rate = learning_rate.assign(learning_rate * float(args['--lr-decay']))
                            print('load previously best model and decay learning rate to %f' % learning_rate.eval(), file=sys.stderr)

                            # load model
                            saver = tf.train.Saver()
                            saver.restore(sess, model_save_path)

                            # reset patience
                            patience = 0

                    if epoch == int(args['--max-epoch']):
                        print('reached maximum number of epochs!', file=sys.stderr)
                        exit(0)


def decode(args: Dict[str, str]):
    """ Performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    @param args (Dict): args from cmd line
    """

    print("load test source sentences from [{}]".format(args['TEST_SOURCE_FILE']), file=sys.stderr)
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        print("load test target sentences from [{}]".format(args['TEST_TARGET_FILE']), file=sys.stderr)
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print("load model from {}".format(args['MODEL_PATH']), file=sys.stderr)
    with tf.Session() as sess:
        NMT.load(args['MODEL_PATH'], sess)

        hypotheses = beam_search(sess, args, test_data_src,
                                 beam_size=int(args['--beam-size']),
                                 max_decoding_time_step=int(args['--max-decoding-time-step']))

        if args['TEST_TARGET_FILE']:
            bleu_score = compute_corpus_level_bleu_score(test_data_tgt, hypotheses)
            print('Corpus BLEU: {}'.format(bleu_score * 100), file=sys.stderr)

        with open(args['OUTPUT_FILE'], 'w') as f:
            for src_sent, hyps in zip(test_data_src, hypotheses):
                top_hyp = hyps
                hyp_sent = ' '.join(top_hyp.value)
                f.write(hyp_sent + '\n')


def beam_search(sess, args, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int):
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model (NMT): NMT Model
    @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    static_example_hyps = NMT.beam_search(beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)
    beam_search_src_sents = tf.get_default_graph().get_tensor_by_name('beam_search_src_sents:0')
    vocab = Vocab.load(args['--vocab'])

    hypotheses = []
    for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
        src_sent = vocab.src.words2indices(src_sent)
        example_hyps = sess.run(static_example_hyps, feed_dict={beam_search_src_sents: src_sent})
        value = vocab.tgt.id2word(example_hyps.predicted_ids)

        hypotheses.append(Hypothesis(value=value, score=example_hyps.scores))

    return hypotheses


def main():
    """ Main func.
    """
    args = docopt(__doc__)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
