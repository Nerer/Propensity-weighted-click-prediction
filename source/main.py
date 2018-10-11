from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import argparse
import tensorflow as tf
import numpy as np
import os
import utils

parser = argparse.ArgumentParser()
parser.add_argument('-alg', '--algo', default='mlp', type = str)
parser.add_argument('-name', '--name', default='model', type = str)
parser.add_argument('-clip', '--clipping', default = 999999.0, type = float)
parser.add_argument('-wclip', '--wclipping', default = 9999999.0, type = float)
parser.add_argument('-batch', '--batch_size', default = 1000, type = int)
parser.add_argument('-val', '--eval', action = 'store_true')
parser.add_argument('-wd', '--wd', default = 0.00001, type = float)
parser.add_argument('-lr', '--lr', default = 0.001, type = float)
parser.add_argument('-np', '--epochs', default = 3, type = int)
parser.add_argument('-seed', '--random_seed', default = 0, type = int)
parser.add_argument('-v', '--verbose', action = 'store_true')
parser.add_argument('-save', '--save_every', default = 10, type = int)
parser.add_argument('-pred', '--predict', action = 'store_true')
parser.add_argument('-com', '--compute_score', action = 'store_true')
parser.add_argument('-vd', '--variance_decay', default = 0, type = float)
parser.add_argument('-d', '--detachw', action = 'store_true')
args = parser.parse_args()

algo = args.algo
name = args.name
clipping = args.clipping
wclipping = args.wclipping
batch_size = args.batch_size
eval = args.eval
wd = args.wd
lr = args.lr
epochs = args.epochs
random_seed = args.random_seed
verbose = args.verbose
save_every = args.save_every
get_pred = args.predict
compute_score = args.compute_score
vd = args.variance_decay
detachw = args.detachw
train_X = np.load('../data/cov_trainX.npy')
train_y = np.load('../data/cov_trainy.npy')
train_p = np.load('../data/cov_trainp.npy')

val_X = np.load('../data/cov_valX.npy')
val_y = np.load('../data/cov_valy.npy')
val_p = np.load('../data/cov_valp.npy')


dataX = train_X
datay = train_y
datap = train_p
if (not eval):
    dataX = np.concatenate([train_X, val_X])
    datay = np.concatenate([train_y, val_y])
    datap = np.concatenate([train_p, val_p])

total = datay.shape[0]
dtype = tf.float32



if (algo == 'mlp'):
    tf.set_random_seed(0)
    X = tf.placeholder(dtype)
    y = tf.placeholder(dtype)
    p = tf.placeholder(dtype)
    wclip = tf.placeholder(dtype)
    w1 = tf.Variable(tf.random_normal([378, 1024], mean=0.0, stddev=0.2, dtype=dtype),
                     name='w1', dtype=dtype)
    b1 = tf.Variable(tf.zeros([1024], dtype=dtype), name='b1', dtype=dtype)
    w2 = tf.Variable(tf.random_normal([1024, 1], mean=0.0, stddev=0.2, dtype=dtype),
                     name='w2', dtype=dtype)
    b2 = tf.Variable(tf.zeros([1], dtype=dtype), name='b2', dtype=dtype)

    xw = tf.matmul(X, w1)
    h1 = tf.nn.relu(xw + b1)
    xww = tf.matmul(h1, w2)
    piw  = p
    logits = xww + b2
    logits = tf.reshape(logits, [-1])
    sig = tf.sigmoid(logits)
    weight = tf.minimum(p, wclip)
    data_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(pos_weight=weight, targets=y, logits=logits))
    weight_loss = wd * (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(b2))
    loss = data_loss + weight_loss
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)

if (algo == 'pmlpb1'):
    tf.set_random_seed(random_seed)
    X = tf.placeholder(dtype)
    y = tf.placeholder(dtype)
    p = tf.placeholder(dtype)
    wclip = tf.placeholder(dtype)
    w1 = tf.Variable(tf.random_normal([378, 1024], mean=0.0, stddev=0.2, dtype=dtype),
                     name='w1', dtype=dtype)
    b1 = tf.Variable(tf.zeros([1024], dtype=dtype), name='b1', dtype=dtype)
    w2 = tf.Variable(tf.random_normal([1024, 1], mean=0.0, stddev=0.2, dtype=dtype),
                     name='w2', dtype=dtype)
    b2 = tf.Variable(tf.zeros([1], dtype=dtype), name='b2', dtype=dtype)

    xw = tf.matmul(X, w1)
    h1 = tf.nn.relu(xw + b1)
    xww = tf.matmul(h1, w2)

    logits = xww + b2
    logits = tf.reshape(logits, [-1])

    piw = tf.reshape(logits, [-1, 7])
    piw = tf.nn.softmax(piw)

    piw = piw[:, 0]

    piw_var = tf.reduce_mean(tf.square(p - 1 / piw))
    piw = tf.reshape(piw, [-1])

    weight = tf.minimum(piw * p, wclip)
    if (detachw):
        npiw = tf.placeholder(dtype)
        weight = tf.minimum(npiw * p, wclip)

    sig = tf.sigmoid(logits)
    logits_m = tf.reshape(logits, [-1, 7])
    data_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(pos_weight=weight, targets=y, logits=logits_m[:, 0]))
    weight_loss = wd * (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(b2))
    var_loss = vd * piw_var
    loss = data_loss + weight_loss + var_loss
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)

if (algo == 'pmlpb2' or algo == 'pmlpb2h'):
    tf.set_random_seed(random_seed)
    X = tf.placeholder(dtype)
    y = tf.placeholder(dtype)
    p = tf.placeholder(dtype)
    wclip = tf.placeholder(dtype)
    w1 = tf.Variable(tf.random_normal([378, 1024], mean=0.0, stddev=0.2, dtype=dtype),
                     name='w1', dtype=dtype)
    b1 = tf.Variable(tf.zeros([1024], dtype=dtype), name='b1', dtype=dtype)
    w2 = tf.Variable(tf.random_normal([1024, 1], mean=0.0, stddev=0.2, dtype=dtype),
                     name='w2', dtype=dtype)
    b2 = tf.Variable(tf.zeros([1], dtype=dtype), name='b2', dtype=dtype)

    xw = tf.matmul(X, w1)
    h1 = tf.nn.relu(xw + b1)
    xww = tf.matmul(h1, w2)

    logits = xww + b2
    logits = tf.reshape(logits, [-1])

    piw = tf.placeholder(dtype)
    sig = tf.sigmoid(logits)

    weight = tf.minimum(p * piw, wclip)
    data_loss = tf.reduce_mean(
        tf.nn.weighted_cross_entropy_with_logits(pos_weight=weight, targets=y, logits=logits))
    weight_loss = wd * (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(b2))
    loss = data_loss + weight_loss
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)

def shifted_scaled_sigmoid(x):
    shift = 1.1875
    scale = 850100
    s = 1 / (1 + np.exp(-x + shift))
    return (s * scale).round(2)

def get_prediction(dataX, datay, datap, mode = 'test'):
    pred = []
    total = datay.shape[0]
    total_batch = total // batch_size + 1
    for i in range(total_batch):
        le = i * batch_size
        ri = (i + 1) * batch_size
        if (ri > total):
            ri = total
        batchX = dataX[le * 7: ri * 7]
        batchy = datay[le: ri]
        batchp = datap[le: ri]
        feed_dict = {}
        feed_dict[X] = batchX
        feed_dict[y] = batchy
        if (mode == 'test'):
            score = sess.run(sig, feed_dict)
            score = shifted_scaled_sigmoid(score)
            score = np.array(score)
            for j in range(ri - le):
                m = score[j * 7: (j + 1) * 7].argmax()
                score[m] += 15
        else:
            score = sess.run(logits, feed_dict)
            score = np.array(score)

        for j in range(ri - le):
            pred.append(score[j * 7: (j + 1) * 7])

    pred = np.array(pred)
    return pred


total_batch = total // batch_size + 1
datapiw = np.ones(shape=(total, ), dtype = np.float32)
fl= open('loss.txt', 'w')
fdl = open('dloss.txt', 'w')
fwt = open('weight.txt', 'w')
fvl = open('vloss.txt', 'w')
for epoch in range(epochs):
    for i in range(total_batch):
        le = i * batch_size
        ri = (i + 1) * batch_size
        if (ri > total):
            ri = total
        batchX = dataX[le * 7 : ri * 7]
        batchy = datay[le : ri]
        batchp = datap[le : ri]
        if (algo == 'pmlpb2' or algo == 'pmlpb2h'):
            batchpiw = datapiw[le : ri]
        batchp = np.minimum(batchp, clipping)
        feed_dict = {}
        if (algo == 'mlp' or algo == 'pmlpb2' or algo == 'pmlpb2h'):
            ind = np.arange(0, 7 * (ri - le), 7)
            batchX = batchX[ind]
        feed_dict[X] = batchX
        feed_dict[y] = batchy
        feed_dict[p] = batchp
        feed_dict[wclip] = wclipping
        if (algo == 'pmlpb2' or algo == 'pmlpb2h'):
            feed_dict[piw] = batchpiw
        if (algo == 'pmlpb1'):
            feed_dict[wclip] = wclipping
            if (detachw):
                cur_piw = sess.run(piw, feed_dict)
                feed_dict[npiw] = cur_piw
        fetches = [loss, optimizer, piw, data_loss]
        l, _, wt, dl= sess.run(fetches, feed_dict)

        wt = np.array(wt)
        if (verbose):
            print('Batch %d : loss %f'%(i, l))
            fl.write('%.6f\n'%l )
            fwt.write('%.6f\n'%wt.mean())
            fdl.write('%.6f\n'%dl)
            #fvl.write('%.6f\n'%vl)
    if ((epoch + 1) % save_every == 0):
        saver = tf.train.Saver()
        saver.save(sess, os.path.join('../cov_models', (name + "%d"%i)))
    if (algo == 'pmlpb2'):
        pred = get_prediction(dataX, datay, datap, mode = 'train')
        pred = np.reshape(pred, newshape = (-1, 7))
        pred = utils.softmax(pred)
        datapiw = pred[:, 0]
    if (algo == 'pmlpb2h'):
        pred = get_prediction(dataX, datay, datap, mode='test')
        pred = np.reshape(pred, newshape=(-1, 7))
        pred = utils.softmax(pred)
        datapiw = pred[:, 0]



if (get_pred):
    dataX = np.load('../data/cov_testX.npy')
    datay = np.load('../data/cov_testy.npy')
    datap = np.load('../data/cov_testp.npy')
    pred = get_prediction(dataX, datay, datap, mode = 'test')
    np.save(name + '_pred.npy', pred)

if (compute_score):
    labels = np.load('../data/cov_testy.npy')
    p = np.load('../data/cov_testp.npy')
    result = utils.evaluation(pred, p, labels)
    n = labels.shape[0]
    pred = utils.softmax(pred)
    print(str(args))
    print(str(result))

    with open('history.txt', 'a') as f:
        f.write(str(args) + '\n')
        f.write(str(result) + '\n')
