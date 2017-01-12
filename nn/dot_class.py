import numpy as np
import matplotlib.pyplot as plt
import copy


# number of points per class
N = 100
# dimensionality
D = 2
# number of classes
K = 3
# total train steps
totol_steps = 10000

# random seed
np.random.seed(0)

def create_data_set():
    x = np.zeros((N*K,D))
    y = np.zeros(N*K, dtype='uint8')
    for j in xrange(K):
      ix = range(N*j,N*(j+1))
      r = np.linspace(0.0,1,N) # radius
      t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
      x[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
      y[ix] = j
    return x, y

def softmax(logits, axis = None):
    #softmax = exp(logits) / reduce_sum(exp(logits), dim)
    exp = np.exp(logits)
    sum = np.sum(exp, axis=axis, keepdims=True)
    return exp / sum




# nn construct
regular = 1e-3
learning_rate = 0.3


# data set
x, y = create_data_set()
data_count = y.shape[0]

def train_with_nx1():
    w = 0.01 * np.random.random((D, K))
    b = np.zeros((1, K))

    for i in xrange(totol_steps):
        scores = np.dot(x, w) + b
        probs = softmax(scores, axis=1)
        y_probs = probs[range(data_count), y]
        #print probs

        # cross entropy -1/n * sum( y*ln(a) + (1-y)ln(1-a) ) y=1
        cross_entropy = -(1.0/data_count) * np.sum( np.log(y_probs) )
        regularization = 0.5 * regular * np.sum(w * w)
        loss = cross_entropy + regularization

        # backpropate
        dscores = copy.deepcopy(probs)
        dscores[range(data_count), y] -= 1
        dscores /= data_count
        dw = x.T.dot(dscores) + (regular * w)
        db = np.sum(dscores, axis=0, keepdims=True)

        w -= learning_rate * dw
        b -= learning_rate * db

        if i % (totol_steps/100) == 0:
            predicted_class = np.argmax(scores, axis=1)
            accuracy = np.mean(predicted_class == y)
            print("step:{0} loss:{1} accuracy:{2}%".format(i, loss, accuracy * 100))


    # draw
    h = 0.02
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


def train_with_nx2():
    h = 100
    w0 = 0.01 * np.random.randn(D, h)  # x:300*2  2*100
    b0 = np.zeros((1, h))
    w1 = 0.01 * np.random.randn(h, K)
    b1 = np.zeros((1, K))

    for i in xrange(totol_steps):
        # activation ReLu
        l1 = np.maximum(0, np.dot(x, w0) + b0)
        scores = np.dot(l1, w1) + b1
        l2 = softmax(scores, axis=1)

        # cross entropy -1/n * sum( y*ln(a) + (1-y)ln(1-a) ) y=1
        cross_entropy = -1/data_count * np.sum(np.log(l2[range(data_count),y]))
        regularization = 0.5*regular*np.sum(w0*w0) + 0.5*regular*np.sum(w1*w1)
        loss = cross_entropy + regularization


        # backpropate the gradient to the parameters
        dscores = copy.deepcopy(l2)
        dscores[range(data_count), y] -= 1
        dscores /= data_count

        dw1 = np.dot(l1.T, dscores)
        db1 = np.sum(dscores, axis=0, keepdims=True)

        dhidden = np.dot(dscores, w1.T)
        dhidden[l1 <= 0] = 0
        dw0 = np.dot(x.T, dhidden)
        db0 = np.sum(dhidden, axis=0, keepdims=True)

        dw1 += regular * w1
        dw0 += regular * w0

        w0 -= learning_rate * dw0
        b0 -= learning_rate * db0
        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1

        hidden_layer = np.maximum(0, np.dot(x, w0) + b0)
        scores = np.dot(hidden_layer, w1) + b1
        predicted_class = np.argmax(scores, axis=1)
        if i % (totol_steps / 100) == 0:
            print 'training accuracy: %.2f' % (np.mean(predicted_class == y))


    # draw
    h = 0.02
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], w0) + b0), w1) + b1
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()



#train_with_nx1()
train_with_nx2()