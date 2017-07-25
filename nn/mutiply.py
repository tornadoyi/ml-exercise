import numpy as np
import tensorflow as tf


MAX_TRAIN_STEPS = int(3e+5)

BATCH_SIZE = 200

LR = 1e-6

'''
def next_batch(batch):
    x = (np.random.randint(0, 100, size=2 * batch) / 100.0).reshape([batch, -1])
    k1 = (np.random.randint(0, 100, size=2 * batch) / 100.0).reshape([batch, -1])
    k2 = (np.random.randint(0, 100, size=2 * batch) / 100.0).reshape([batch, -1])
    y1 = np.sum(x * k1, axis=1, keepdims=True)
    y2 = np.sum(x * k2, axis=1, keepdims=True)
    return np.hstack([k1, k2, y1, y2]), x
'''

'''
X = np.array(list(range(1, 1+2))) / 10.0

def next_batch(batch):
    index = np.random.randint(0, len(X), size=batch)
    x = X[index].reshape([batch, -1])
    k = (np.random.randint(1, 100, size=batch) / 100.0).reshape([batch, -1])
    y = np.sum(x * k, axis=1, keepdims=True)
    #return  np.hstack([k, y]), x
    return np.hstack([x, k]), y
'''

def next_batch(batch):
    x = (np.random.randint(1, 10, size=2*batch)).reshape([batch, -1])
    #x = np.random.normal(size=(2*batch)).reshape([batch, -1])
    y = np.prod(x, axis=1, keepdims=True)
    return x, y


w_init = tf.random_normal_initializer(0., .1)

optimizer = tf.train.RMSPropOptimizer(LR)

input_x = tf.placeholder(tf.float32, [None, 2])
input_y = tf.placeholder(tf.float32, [None, 1])

#inputs = tf.concat([input_x, input_y], axis=1)

# relu is better than tanh

l = tf.layers.dense(input_x, 512, tf.nn.relu, kernel_initializer=w_init)

l = tf.layers.dense(l, 256, tf.nn.relu, kernel_initializer=w_init)

l = tf.layers.dense(l, 128, kernel_initializer=w_init)

l = tf.layers.dense(l, 1, kernel_initializer=w_init)

#y1 = tf.reduce_sum(l * input_x[:, 0:2], axis=1, keep_dims=True)
#y2 = tf.reduce_sum(l * input_x[:, 2:], axis=1, keep_dims=True)

#result = tf.concat([y1, y2], axis=1)
result = l

loss = tf.nn.l2_loss(input_y - result) / BATCH_SIZE

train_op = optimizer.minimize(loss)


sess = tf.Session()

sess.run(tf.global_variables_initializer())


for i in range(MAX_TRAIN_STEPS):
    x, y = next_batch(BATCH_SIZE)

    v_loss = sess.run([train_op, loss], feed_dict={input_x: x, input_y: y})[1:]

    if (i+1) % 1000 == 0:
        print('step: {0} loss: {1}'.format(i, v_loss))



x, y = next_batch(10)


v_predict = sess.run(result, feed_dict={input_x: x, input_y: y})

print(v_predict)
print('='*100)
print(y)