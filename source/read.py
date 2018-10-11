import numpy as np
fr = open('history.txt', 'r')

cnt = 0

res = []
for lines in fr:
    cnt += 1
    if (cnt % 2 == 0):
        dict = eval(lines)
        if (dict['ips'] != dict['ips']):
            dict['ips'] = 0
        if (dict['ips_std'] != dict['ips_std']):
            dict['ips_std'] = 0
        res.append((dict['ips'], dict['one'], dict['ips_std']))


mlpb1res = [(0, 0, 0)] * 8
mlpb2hres = [(0, 0, 0)] * 8
mlpb2res = [(0, 0, 0)] * 8
a = 3
b = 8
c = 9
total = a * b * c
print(len(res))

for i in range(len(res)):
    if (i < b * c):
        clipping = i // c
        mlpb1res[clipping] = max(mlpb1res[clipping], res[i])
    else:
        if (i < 2 * b * c):
            clipping = (i - b * c) // c
            mlpb2hres[clipping] = max(mlpb2hres[clipping], res[i])
        else:
            if (i < 3 * b * c):
                clipping = (i - 2 * b * c) // c
                mlpb2res[clipping] = max(mlpb2res[clipping], res[i])


'''
for i in range(len(res)):
    if (i < b * c):
        clipping = i // c
        mlpb2hres[clipping] = max(mlpb2hres[clipping], res[i])
    else:
        if (i < 2 * b * c):
            clipping = (i - b * c) // c
            mlpb2res[clipping] = max(mlpb2res[clipping], res[i])
'''
print(mlpb1res)
print(mlpb2hres)
print(mlpb2res)

n = len(mlpb1res)
ips1 = []
ips2 = []
ips3 = []

one1 = []
one2 = []
one3 = []

std1 = []
std2 = []
std3 = []

for i in range(n):
    ips1.append(mlpb1res[i][0])
    one1.append(mlpb1res[i][1])
    std1.append(mlpb1res[i][2])


for i in range(n):
    ips2.append(mlpb2hres[i][0])
    one2.append(mlpb2hres[i][1])
    std2.append(mlpb2hres[i][2])

for i in range(n):
    ips3.append(mlpb2res[i][0])
    one3.append(mlpb2res[i][1])
    std3.append(mlpb2res[i][2])


ips1 = np.array(ips1)
one1 = np.array(one1)
std1 = np.array(std1)

ips2 = np.array(ips2)
one2 = np.array(one2)
std2 = np.array(std2)

ips3 = np.array(ips3)
one3 = np.array(one3)
std3 = np.array(std3)

np.save('./draw/ips1', ips1)
np.save('./draw/ips2', ips2)
np.save('./draw/ips3', ips3)

np.save('./draw/one1', one1)
np.save('./draw/one2', one2)
np.save('./draw/one3', one3)

np.save('./draw/std1', std1)
np.save('./draw/std2', std2)
np.save('./draw/std3', std3)


