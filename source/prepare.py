import numpy as np
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
f = open('../data/covtype.data')
label = []
X = []
counter = [0] * 7
for lines in f:
    features = lines.split(',')
    vec = []
    cnt = 0
    for item in features:
        cnt += 1
        if (cnt <= 54):
            vec.append(float(item))
        else:
            tmp = np.zeros(shape=(7, ), dtype = np.float32)
            tmp[int(item)-1] = 1
            counter[int(item)-1] += 1
            label.append(tmp)
    X.append(vec)

print(counter)
X = np.array(X)
X = scaler.fit_transform(X)
label = np.array(label)


print(X.shape, label.shape)
total = X.shape[0]
ind = np.arange(X.shape[0])
print(ind)
for i in range(7):
    np.random.shuffle(ind)
X = X[ind]
label = label[ind]

train_size = int(total / 2)
val_size = int(train_size / 2)
test_size = total - train_size - val_size

train_X = X[0 : train_size]
train_y = label[0 : train_size]
val_X = X[train_size : train_size + val_size]
val_y = label[train_size : train_size + val_size]
test_X = X[train_size + val_size : total]
test_y = label[train_size + val_size : total]

np.save('../data/covtype_train_X.npy', train_X)
np.save('../data/covtype_train_y.npy', train_y)

np.save('../data/covtype_val_X.npy', val_X)
np.save('../data/covtype_va;_y.npy', val_y)

np.save('../data/covtype_test_X.npy', test_X)
np.save('../data/covtype_test_y.npy', test_y)

small_size = int((train_size + val_size) / 20)
small_X = X[0 : small_size]
small_y = label[0 : small_size]


np.save('../data/covtype_small_X.npy', small_X)
np.save('../data/covtype_small_y.npy', small_y)

print(total, train_size, val_size, test_size)
