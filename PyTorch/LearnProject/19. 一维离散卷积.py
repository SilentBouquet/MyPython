import numpy as np


def conv1d(x, w, p=0, s=1):
    # x：输入向量
    # w：滤波器
    # p：填充长度，默认为0
    # s：步长，默认为1
    w_rot = np.array(w[::-1])
    x_padded = np.array(x)
    if p > 0:
        zero_pad = np.zeros(shape=p)
        x_padded = np.concatenate(([zero_pad, x_padded, zero_pad]))
    res = []
    for i in range(0, int((len(x_padded) - len(w_rot)) / s) + 1, s):
        res.append(np.sum(x_padded[i:i + w_rot.shape[0]] * w_rot))
    return np.array(res)


x = [1, 3, 2, 4, 5, 6, 1, 3]
w = [1, 0, 3, 1, 2]
print('Conv1d Implementation: ', conv1d(x, w, p=2, s=1))
print('Numpy Results: ', np.convolve(x, w, mode='same'))