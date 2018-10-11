import numpy as np

def softmax(pred, ax = 1):
    pred = pred - np.max(pred, axis = ax).reshape(-1, 1)
    pred = np.exp(pred)
    pred = pred / np.sum(pred, axis = ax).reshape(-1, 1)
    return pred



def evaluation(pred, p, labels):
    total = pred.shape[0]
    pred = softmax(pred)
    weight = pred[:, 0] * p
    numerator_m = weight * 1
    numerator = labels * weight
    denominator = weight
    poscnt = 0
    negcnt = 0
    for i in range(labels.shape[0]):
        if (labels[i] > 0.5):
            poscnt += 1
            if (poscnt < 10):
                print(pred[i])

        else:
            negcnt += 1
    print(poscnt, negcnt)
    modified_denominator = total
    scaleFactor = np.sqrt(total) / modified_denominator
    IPS = numerator.sum(dtype=np.longdouble) / modified_denominator
    One = numerator_m.sum(dtype=np.longdouble) / modified_denominator
    IPS_std = 2.58 * numerator.std(dtype=np.longdouble) * scaleFactor  # 99% CI
    ImpWt = denominator.sum(dtype=np.longdouble) / modified_denominator
    SNIPS = IPS/ImpWt
    ret = {}
    ret['ips'] = IPS * 1e4
    ret['one'] = One * 1e4
    ret['ips_std'] = IPS_std * 1e4
    ret['snips'] = SNIPS * 1e4
    return ret


