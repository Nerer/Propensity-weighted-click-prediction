import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--large', action = 'store_true')

args = parser.parse_args()
enlarge = args.large

propensity = np.load('../data/covtype_prop_propensity.npy')
label = np.load('../data/covtype_prop_datay.npy')
X = np.load('../data/covtype_prop_dataX.npy')
tlabel = np.load('../data/covtype_prop_tlabel.npy')
total = propensity.shape[0]
if (enlarge):
    train_size = int(10 * total / 16)
    val_size = int(5 * total / 16)
    test_size = total - train_size - val_size
else:
    train_size =int(total / 2)
    val_size = int(train_size / 2)
    test_size = total - train_size - val_size

print(total, train_size, val_size, test_size)
print(X.shape[0])

trainX = X[0 : train_size * 7]
trainy = label[0 : train_size]
trainp = propensity[0 : train_size]
traint = tlabel[0 : train_size]

valX = X[train_size * 7 : train_size * 7 + val_size * 7]
valy = label[train_size : train_size + val_size]
valp = propensity[train_size : train_size + val_size]
valt = tlabel[train_size : train_size + val_size]

testX = X[train_size * 7 + val_size * 7 : total * 7]
testy = label[train_size + val_size : total]
testp = propensity[train_size + val_size : total]
testt = tlabel[train_size + val_size : total]

np.save('../data/cov_trainX.npy', trainX)
np.save('../data/cov_trainy.npy', trainy)
np.save('../data/cov_trainp.npy', trainp)
np.save('../data/cov_traintt.npy', traint)

np.save('../data/cov_valX.npy', valX)
np.save('../data/cov_valy.npy', valy)
np.save('../data/cov_valp.npy', valp)
np.save('../data/cov_valt.npy', valt)

np.save('../data/cov_testX.npy', testX)
np.save('../data/cov_testy.npy', testy)
np.save('../data/cov_testp.npy', testp)
np.save('../data/cov_testt.npy', testt)

