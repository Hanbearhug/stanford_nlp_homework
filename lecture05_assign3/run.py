#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2018-19: Homework 3
run.py: Run the dependency parser.
Sahil Chopra <schopra8@stanford.edu>
"""
from datetime import datetime
import os
import pickle
import math
import time
from tqdm import tqdm
from torch import nn, optim
import tensorflow as tf

from parser_model import ParserModel
from utils.parser_utils import minibatches, load_and_preprocess_data, AverageMeter
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
tf.reset_default_graph()
# -----------------
# Primary Functions
# -----------------
def train(parser, train_data, dev_data, output_path, batch_size=256, n_epochs=10, lr=0.0005):
    """ Train the neural dependency parser.

    @param parser (Parser): Neural Dependency Parser
    @param train_data ():
    @param dev_data ():
    @param output_path (str): Path to which model weights and results are written.
    @param batch_size (int): Number of examples in a single batch
    @param n_epochs (int): Number of training epochs
    @param lr (float): Learning rate
    """
    best_dev_UAS = 0


    ### YOUR CODE HERE (~2-7 lines)
    ### TODO:
    ###      1) Construct Adam Optimizer in variable `optimizer`
    ###      2) Construct the Cross Entropy Loss Function in variable `loss_func`
    ###
    ### Hint: Use `parser.model.parameters()` to pass optimizer
    ###       necessary parameters to tune.
    ### Please see the following docs for support:
    ###     Adam Optimizer: https://pytorch.org/docs/stable/optim.html
    ###     Cross Entropy Loss: https://pytorch.org/docs/stable/nn.html#crossentropyloss
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    print(parser.model.n_classes)
    y_true = tf.placeholder(tf.int32, shape=(None,3))
    x = tf.placeholder(tf.int32, shape=(None, parser.model.n_features),name='x')

    y_pred = parser.model.forward(x)
    loss = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)))
    train_step = optimizer.minimize(loss)

    y_pred_softmax = tf.nn.softmax(y_pred,name='y_pred_softmax')

    init = tf.global_variables_initializer()

    ### END YOUR CODE
    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True)) as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
            dev_UAS = train_for_epoch(parser, train_data, dev_data, batch_size, sess, train_step, x, y_true, loss, y_pred_softmax)
            if dev_UAS > best_dev_UAS:
                best_dev_UAS = dev_UAS
                print("New best dev UAS! Saving model.")
                save_path = saver.save(sess, output_path)
                print("Model saved in file: ", save_path)


def train_for_epoch(parser, train_data, dev_data, batch_size, sess, train_step, x, y_true, loss, y_pred_softmax):
    """ Train the neural dependency parser for single epoch.

    Note: In PyTorch we can signify train versus test and automatically have
    the Dropout Layer applied and removed, accordingly, by specifying
    whether we are training, `model.train()`, or evaluating, `model.eval()`

    @param parser (Parser): Neural Dependency Parser
    @param train_data ():
    @param dev_data ():
    @param optimizer (nn.Optimizer): Adam Optimizer
    @param loss_func (nn.CrossEntropyLoss): Cross Entropy Loss Function
    @param batch_size (int): batch size
    @param lr (float): learning rate

    @return dev_UAS (float): Unlabeled Attachment Score (UAS) for dev data
    """
    parser.model.is_train=True # Places model in "train" mode, i.e. apply dropout layer
    n_minibatches = math.ceil(len(train_data) / batch_size)
    loss_meter = AverageMeter()
    with tqdm(total=(n_minibatches)) as prog:
        for i, (train_x, train_y) in enumerate(minibatches(train_data, batch_size)):
            #optimizer.zero_grad()   # remove any baggage in the optimizer
            #loss = 0. # store loss for this batch here

            ### YOUR CODE HERE (~5-10 lines)
            ### TODO:
            ###      1) Run train_x forward through model to produce `logits`
            ###      2) Use the `loss_func` parameter to apply the PyTorch CrossEntropyLoss function.
            ###         This will take `logits` and `train_y` as inputs. It will output the CrossEntropyLoss
            ###         between softmax(`logits`) and `train_y`. Remember that softmax(`logits`)
            ###         are the predictions (y^ from the PDF).
            ###      3) Backprop losses
            ###      4) Take step with the optimizer
            ### Please see the following docs for support:
            ###     Optimizer Step: https://pytorch.org/docs/stable/optim.html#optimizer-step
            loss_train, _ = sess.run([loss, train_step], feed_dict={x: train_x, y_true: train_y})

            ### END YOUR CODE
            prog.update(1)
            loss_meter.update(loss_train.item())

    print ("Average Train Loss: {}".format(loss_meter.avg))

    print("Evaluating on dev set",)
    parser.model.is_train=False # Places model in "eval" mode, i.e. don't apply dropout layer
    dev_UAS, _ = parser.parse(dev_data, sess, y_pred_softmax, x)
    print("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
    return dev_UAS


if __name__ == "__main__":
    # Note: Set debug to False, when training on entire corpus
    debug = False
    # debug = False

    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    parser, embeddings, train_data, dev_data, test_data = load_and_preprocess_data(debug)

    start = time.time()
    model = ParserModel(embeddings)
    parser.model = model
    print("took {:.2f} seconds\n".format(time.time() - start))

    print(80 * "=")
    print("TRAINING")
    print(80 * "=")
    output_dir = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
    output_path = output_dir + "model.ckpt"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train(parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005)

    if not debug:
        print(80 * "=")
        print("TESTING")
        print(80 * "=")
        print("Restoring the best model weights found on the dev set")
        saver = tf.train.Saver()
        print(parser.model.n_classes)
        with tf.Session() as sess:
            # Restore variables from disk.
            saver.restore(sess, output_path)
            print("Model restored.")
            # Do some work with the model
            print("Final evaluation on test set",)
            parser.model.is_train = False
            y_pred_softmax = tf.get_default_graph().get_tensor_by_name("y_pred_softmax:0")
            x = tf.get_default_graph().get_tensor_by_name("x:0")
            UAS, dependencies = parser.parse(test_data, sess, y_pred_softmax, x)
            print("- test UAS: {:.2f}".format(UAS * 100.0))
            print("Done!")
