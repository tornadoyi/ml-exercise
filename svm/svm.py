import numpy as np
import matplotlib.pyplot as plt
import operator
import time
import points


def main():
    datas, labels = load_data_set()
    b, alphas = smo(datas, labels, 0.6, 0.0001, 4000)
    w = calculate_w(alphas, datas, labels)
    print("w = \n {0}".format(w))
    print("b = \n {0}".format(b))
    plot(datas, labels, w, b)



def smo(X, Y, C, toler, max_iter, kTup=('lin', 0)):
    opt = Option(np.mat(X), np.mat(Y).T, C, toler)
    iter = 0
    scan_set = True
    alpha_change_count = 0
    while (iter < max_iter) and ((alpha_change_count > 0) or (scan_set)):
        alpha_change_count = 0
        if scan_set:
            for i in xrange(opt.N):
                alpha_change_count += inner_loop(i, opt)
            # print("fullSet, iter: %d i:%d, pairs changed %d" % (iterr, i, alpha_change_count))
            iter += 1
        else:
            nonBoundIs = np.nonzero((opt.alphas.A > 0) * (opt.alphas.A < C))[0]
            for i in nonBoundIs:
                alpha_change_count += inner_loop(i, opt)
                # print("non-bound, iter: %d i:%d, pairs changed %d" % (iterr, i, alpha_change_count))
            iter += 1
        if scan_set:
            scan_set = False
        elif (alpha_change_count == 0):
            scan_set = True
        # print("iteration number: %d" % iterr)
    return opt.b, opt.alphas


def inner_loop(i, opt):
    e_i = calculate_error(opt, i)

    ### check and pick up the alpha who violates the KKT condition
    ## satisfy KKT condition
    # 1) yi*f(i) >= 1 and alpha == 0 (outside the boundary)
    # 2) yi*f(i) == 1 and 0<alpha< C (on the boundary)
    # 3) yi*f(i) <= 1 and alpha == C (between the boundary)
    ## violate KKT condition
    # because y[i]*E_i = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, so
    # 1) if y[i]*E_i < 0, so yi*f(i) < 1, if alpha < C, violate!(alpha = C will be correct)
    # 2) if y[i]*E_i > 0, so yi*f(i) > 1, if alpha > 0, violate!(alpha = 0 will be correct)
    # 3) if y[i]*E_i = 0, so yi*f(i) = 1, it is on the boundary, needless optimized
    if ((opt.Y[i] * e_i < -opt.toler) and (opt.alphas[i] < opt.C)) or \
        ((opt.Y[i] * e_i > opt.toler) and (opt.alphas[i] > 0)):

        # step 1: select alpha j
        j, e_j = select_j(opt, i, e_i)
        alpha_i_old = opt.alphas[i].copy()
        alpha_j_old = opt.alphas[j].copy()

        # step 2: calculate the boundary L and H for alpha j
        if (opt.Y[i] != opt.Y[j]):
            L = max(0, opt.alphas[j] - opt.alphas[i])
            H = min(opt.C, opt.C + opt.alphas[j] - opt.alphas[i])
        else:
            L = max(0, opt.alphas[j] + opt.alphas[i] - opt.C)
            H = min(opt.C, opt.alphas[j] + opt.alphas[i])
        if (L == H):
            return 0

        # step 3: calculate eta (the similarity of sample i and j)
        eta = 2.0 * opt.X[i, :] * opt.X[j, :].T - opt.X[i, :] * opt.X[i, :].T - opt.X[j, :] * opt.X[j, :].T
        if eta >= 0:
            # print("eta >= 0")
            return 0

        # step 4: update alpha j
        opt.alphas[j] -= opt.Y[j] * (e_i - e_j) / eta

        # step 5: clip alpha j
        if opt.alphas[j] < L:
            opt.alphas[j] = L
        elif opt.alphas[j] > H:
            opt.alphas[j] = H

        # step 6: if alpha j not moving enough, just return
        if abs(alpha_j_old - opt.alphas[j]) < 0.00001:
            # print("j not moving enough")
            update_error(opt, j)
            return 0

        # step 7: update alpha i after optimizing aipha j
        opt.alphas[i] += opt.Y[j] * opt.Y[i] * (alpha_j_old - opt.alphas[j])

        # step 8: update threshold b
        b1 = opt.b - e_i - opt.Y[i] * (opt.alphas[i] - alpha_i_old) * opt.X[i, :] * opt.X[i, :].T - opt.Y[j] * (opt.alphas[j] - alpha_j_old) * opt.X[i, :] * opt.X[j, :].T
        b2 = opt.b - e_j - opt.Y[i] * (opt.alphas[i] - alpha_i_old) * opt.X[i, :] * opt.X[j, :].T - opt.Y[j] * (opt.alphas[j] - alpha_j_old) * opt.X[j, :] * opt.X[j, :].T
        if (0 < opt.alphas[i]) and (opt.C > opt.alphas[i]):
            opt.b = b1
        elif (0 < opt.alphas[j]) and (opt.C > opt.alphas[j]):
            opt.b = b2
        else:
            opt.b = (b1 + b2) / 2.0

        update_error(opt, i)
        update_error(opt, j)

        return 1
    else:
        return 0


def select_j(opt, i, error_i):
    opt.errors[i] = [1, error_i]
    error_indexes = np.nonzero(opt.errors[:, 0].A)[0]
    max_sub = 0
    j = 0
    error_j = 0
    if len(error_indexes) > 1:
        for k in error_indexes:
            if k == i:
                continue
            error_k = calculate_error(opt, k)
            sub = abs(error_k - error_i)
            if sub > max_sub:
                max_sub = sub
                j = k
                error_j = error_k
    else:
        j = i
        while (j == i):
            j = int(np.random.uniform(0, opt.N))
        error_j = calculate_error(opt, j)
    return j, error_j


def calculate_error(opt, k):
    fx_k = float(np.multiply(opt.alphas, opt.Y).T * (opt.X * opt.X[k, :].T)) + opt.b
    e_k = fx_k - float(opt.Y[k])
    return e_k


def calculate_w(alphas, X, Y):
    X = np.mat(X)
    labelMat = np.mat(Y).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


def plot(datas, labels, w = None, b = None):
    dataArr = np.array(datas)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    minX = float(9999)
    maxX = -minX
    for i in range(n):
        if int(labels[i]) == 1:
            xcord1.append(dataArr[i, 0])
            ycord1.append(dataArr[i, 1])
            minX, maxX = np.min([xcord1[-1], minX]), np.max([xcord1[-1], maxX])
        else:
            xcord2.append(dataArr[i, 0])
            ycord2.append(dataArr[i, 1])
            minX, maxX = np.min([xcord2[-1], minX]), np.max([xcord2[-1], maxX])


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(minX, maxX, 0.1)
    y = (-b[0, 0] - x * w[0, 0]) / w[1, 0]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()



def load_data_set():
    x = []
    y = []
    for p in points.points:
        x.append((p[0], p[1]))
        y.append(p[2])
    return x, y


class Option:
    def __init__(self, X, Y, C, toler):
        self.X = X
        self.Y = Y
        self.C = C
        self.toler = toler
        self.N = np.shape(X)[0]
        self.alphas = np.mat(np.zeros((self.N, 1)))
        self.b = 0
        self.errors = np.mat(np.zeros((self.N, 2)))


def update_error(opt, k):
    e_k = calculate_error(opt, k)
    opt.errors[k] = [1, e_k]

if __name__ == '__main__':
    main()



