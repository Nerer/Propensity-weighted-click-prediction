import numpy as np
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--smooth', action = 'store_true')

args = parser.parse_args()

smooth = args.smooth

dataX = np.load('../data/covtype_small_X.npy')
datay = np.load('../data/covtype_small_y.npy')

train_size = dataX.shape[0]
batch_size = 1000
wd = 0.00001
lr = 0.001
total_batch = train_size // batch_size + 1
dtype = tf.float32


X = tf.placeholder(dtype)
y = tf.placeholder(dtype)
w1 = tf.Variable(tf.random_normal([54, 1024], mean=0.0, stddev=0.2, dtype=dtype),
                                            name='w1', dtype=dtype)
b1 = tf.Variable(tf.zeros([1024], dtype=dtype), name='b1', dtype=dtype)
w2 = tf.Variable(tf.random_normal([1024, 7], mean=0.0, stddev=0.2, dtype=dtype),
                                            name='w2', dtype=dtype)
b2 = tf.Variable(tf.zeros([7], dtype=dtype), name='b2', dtype=dtype)

xw = tf.matmul(X, w1)
h1 = tf.nn.relu(xw + b1)
xww = tf.matmul(h1, w2)
logits = xww + b2

prob = tf.nn.softmax(logits)
sig = tf.nn.sigmoid(logits)
myloss = tf.reduce_sum(-y*tf.log(prob), 1)
tmploss = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y)
data_loss = tf.reduce_mean(tmploss)
loss = data_loss + \
                           wd * (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(b2))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

sess = tf.Session()

tf.global_variables_initializer().run(session=sess)

res = []
for epoch in range(0, 3):
    print('Epoch %d:'%epoch)
    for i in range(total_batch):
        le = i * batch_size
        ri = (i + 1) * batch_size
        if (ri >= train_size):
            ri = train_size
        batchX = dataX[le:ri]
        batchy = datay[le:ri]
        feed_dict = {}
        feed_dict[X] = batchX
        feed_dict[y] = batchy
        fetch = [loss, data_loss, optimizer]
        l, dl, _ = sess.run(fetch, feed_dict)
        print('Batch %d, loss = %f'%(i, dl))

        log, proba, ml, tl = sess.run([logits, prob, myloss, tmploss], feed_dict)
        print(tl.shape)
        for j in range(1):
            res.append(log[j])
            res.append(proba[j])
            res.append(batchy[j])
            res.append(tl[j])
            res.append(ml[j])

with open('log.txt', 'w') as fw:
    for i in range(len(res)):
        fw.write(str(res[i]) + '\n')
        if (i % 5 == 4):
            fw.write('\n')

train_X = np.load('../data/covtype_train_X.npy')
train_y = np.load('../data/covtype_train_y.npy')
val_X = np.load('../data/covtype_val_X.npy')
val_y = np.load('../data/covtype_va;_y.npy')
test_X = np.load('../data/covtype_test_X.npy')
test_y = np.load('../data/covtype_test_y.npy')



train_size = int(train_X.shape[0])
val_size = int(val_X.shape[0])
test_size = int(test_X.shape[0])


ind = np.arange(train_size + val_size)
for i in range(7):
    np.random.shuffle(ind)

dataX = []
for i in range(0, 5):
    dataX.append(np.copy(train_X))
for i in range(0, 5):
    dataX.append(np.copy(val_X))
dataX = np.concatenate(dataX)
dataX = dataX[ind]
dataX = np.concatenate([dataX, test_X])

datay = []
for i in range(0, 5):
    datay.append(np.copy(train_y))
for i in range(0, 5):
    datay.append(np.copy(val_y))
datay = np.concatenate(datay)
datay = datay[ind]
datay = np.concatenate([datay, test_y])






total = dataX.shape[0]

total_batch = total // batch_size + 1

import random


def weighted_choice(x):
    total = x.sum()
    th = random.uniform(0, total)
    cur = 0
    num = 7
    ret = None

    for i in range(num):
        cur += x[i]
        if (cur >= th):
            ret = i
            break
    return ret, x[ret]

def smooth_choice(x):
    b = []
    for i in range(7):
        b.append((x[i], i))
    b.sort(reverse=True)
    weight = [1., 1./2, 1./3, 1./4, 1./5, 1./6, 1./7]
    weight = np.array(weight)
    weight = weight / weight.sum()
    weight = np.array(weight)
    choice, _ = weighted_choice(weight)
    return b[choice][1], weight[choice]

feature = np.zeros(shape=(total * 7, 54 * 7), dtype = np.float32)
propensity = []
labels = []
tlabel = []
pos = 0
for i in range(total_batch):
    le = i * batch_size
    ri = (i + 1) * batch_size
    if (ri > total):
        ri = total
    batchX = dataX[le:ri]
    batchy = datay[le:ri]
    feed_dict = {}
    feed_dict[X] = batchX
    feed_dict[y] = batchy
    proba = sess.run(prob, feed_dict)
    proba = np.array(proba)
    for j in range(ri - le):
        #if (j < 3):
        #   print(batchy[j][proba[j].argmax()])
        if (smooth):
            choice, p = smooth_choice(proba[j])
        else:
            choice, p = weighted_choice(proba[j])
        feature[pos][choice * 54 : (choice + 1) * 54] = batchX[j]
        propensity.append(1 / p)
        tlabel.append(batchy[j][proba[j].argmax()])
        if (batchy[j][choice] > 0.999):
            labels.append(1)
        else:
            labels.append(0)
        ind = np.arange(7)
        #for tt in range(7):
        #    np.random.shuffle(ind)
        for k in ind:
            if (k != choice):
                pos += 1
                feature[pos][k * 54 : (k + 1) * 54] = batchX[j]
        pos += 1

propensity = np.array(propensity, dtype = np.float32)
labels = np.array(labels, dtype = np.float32)
tlabel = np.array(tlabel)
np.save('../data/covtype_prop_dataX.npy', feature)
np.save('../data/covtype_prop_datay.npy', labels)
np.save('../data/covtype_prop_propensity.npy', propensity)
np.save('../data/covtype_prop_tlabel.npy', tlabel)
