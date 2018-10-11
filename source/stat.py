import numpy as np
import pandas as pd
from matplotlib import pyplot
train_p = np.load('../data/cov_trainp.npy')
val_p  = np.load('../data/cov_valp.npy')
test_p = np.load('../data/cov_testp.npy')
test_t = np.load('../data/cov_testt.npy')
test_y = np.load('../data/cov_testy.npy')
print(test_t.sum()/test_t.shape[0])
print(test_y.sum()/test_y.shape[0])
print((test_t * test_p * test_y).sum() / test_t.shape[0])
p = np.concatenate([train_p, val_p, test_p])
pp = pd.Series(p)
res = {}
res['mean'] = pp.mean()
res['median'] = pp.median()
res['max'] = pp.max()
res['min'] = pp.min()

print(res)


p.sort()

total = p.shape[0]
now = 0
for i in range(total):
    if (p[i] < 2):
       now += 1
print(now, total)
n, bins, patches = pyplot.hist(p, [1, 2, 4, 8, 16, 32, 64], normed = True, cumulative = True)
print(n)
print(bins)
