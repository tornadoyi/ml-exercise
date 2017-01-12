import math
import numpy as np


def int2binary(n, dim = 64):
    x = np.zeros(dim, dtype=np.uint8)
    for i in xrange(dim):
        if n == 0: break
        b = n & 0x01
        x[i] = b
        n >>= 1
    return x

def binary2int(binary):
    ex = 1
    sum = 0
    for b in binary:
        sum += ex * b
        ex *= 2
    return int(sum)



def sigmoid(x): return 1 / (1 + np.exp(-x))

def sigmoid_d(x):
    y = sigmoid(x)
    return y * (1 - y)

def sigmoid_d_output(y): return y * (1 - y)

# config

max_number = 1
data_set = []
number_dim = 8
learning_rate = 0.3
batch_count = 10
train_steps = 10000
model_size = 20

# model

w0 = np.zeros((number_dim*2, model_size)) + 0.5#2 * np.random.random((2, number_dim)) - 1
w1 = np.zeros((model_size, number_dim)) + 0.5
b0 = np.zeros((batch_count, model_size))
b1 = np.zeros((batch_count, number_dim))

def gen_data_set():
    for i in xrange(max_number + 1):
        for j in xrange(max_number + 1):
            data_set.append((i, j))
gen_data_set()

data_index = -1
def get_batch(count):
    global data_index

    x = y = None
    for i in xrange(count):
        data_index = (data_index + 1) % len(data_set)
        d = data_set[data_index]

        a = int2binary(d[0], number_dim)
        b = int2binary(d[1], number_dim)
        c = int2binary(d[0] ^ d[1], number_dim)
        v_x = np.r_[a,b].reshape((1,-1))
        v_y = np.array(c).reshape((1, -1))

        if x is None:
            x = v_x
            y = v_y
        else:
            x = np.r_[x, v_x]
            y = np.r_[y, v_y]

    return x, y



def check(x, y):
    error = 0
    sum = x.shape[0]
    y = np.clip(y, 0, 1)
    y = np.around(y)
    for i in xrange(sum):
        a = binary2int(x[i, 0:number_dim])
        b = binary2int(x[i, number_dim:number_dim*2])
        c = a ^ b
        d = binary2int(y[i])
        if c != d: error += 1

    return float(sum-error) / float(sum)


for i in xrange(train_steps):
    x, y = get_batch(batch_count)

    # ======================== front propagate ========================
    l1 = sigmoid(np.dot(x, w0) + b0)
    l2 = sigmoid(np.dot(l1, w1) + b1)
    y_ = l2
    cost = np.square(y_ - y) / 2


    # ======================== back propagate  ========================

    # d(cost)/d(w1) = d(cost)/d(l2) * d(l2)/d(sigmoid2_input) * d(sigmoid2_input)/d(w1)
    # error_l2 = d(cost)/d(l2)
    # d_l2 = d(l2)/d(sigmoid2_input)
    # d_sigmoid2_input = d(sigmoid2_input)/d(w1)
    error_l2 = y_ - y
    d_l2 = sigmoid_d_output(l2)
    d_sigmoid2_input = l1
    delta_w1 = d_sigmoid2_input.T.dot(error_l2 * d_l2)


    # d(cost)/d(w0) = d(cost)/d(l2) * d(l2)/d(sigmoid2_input) * d(sigmoid2_input)/d(l1) * d(l1)/d(sigmoid1_input) * d(sigmoid1_input)/d(w0)
    # errorl_l1 =  d(cost)/d(l2) * d(l2)/d(sigmoid2_input) * d(sigmoid2_input)/d(l1)
    # d_l1 = d(l1)/d(sigmoid1_input)
    # d_sigmoid1_input = d(sigmoid1_input)/d(w0)
    error_l1 = (error_l2 * d_l2).dot(w1.T)
    d_l1 = sigmoid_d_output(l1)
    d_sigmoid1_input = x
    delta_w0 = d_sigmoid1_input.T.dot(error_l1 * d_l1)


    # d(cost)/d(b1) = d(cost)/d(l2) * d(l2)/d(sigmoid2_input) * d(sigmoid2_input)/d(b1)
    delta_b1 = (error_l2 * d_l2) * 1

    # d(cost)/d(b0) = d(cost)/d(l2) * d(l2)/d(sigmoid2_input) * d(sigmoid2_input)/d(l1) * d(l1)/d(sigmoid1_input) * d(sigmoid1_input)/d(b0)
    delta_b0 = error_l1 * d_l1 * 1


    # update w and b
    w0 -= learning_rate * delta_w0
    w1 -= learning_rate * delta_w1
    b0 -= learning_rate * delta_b0
    b1 -= learning_rate * delta_b1


    if i % (train_steps / 10) == 0:
        error = np.sum(np.square(cost))
        accuracy = check(x, y_) * 100.0
        print('index:{0} accuracy:{1}%'.format(i, accuracy))


print('================= result =================')

x, y = get_batch(batch_count)
l1 = sigmoid(np.dot(x, w0) + b0)
l2 = sigmoid(np.dot(l1, w1) + b1)
y_ = l2
accuracy = check(x, y_) * 100.0
print('accuracy:{0}%'.format(accuracy))
print(y)
print(y_)