import numpy as np
import tensorflow as tf


MAX_TRAIN_STEPS = 10000

BATCH_SIZE = 10

LR = 0.001

X = np.array([3, 5], dtype=np.float)

def next_batch(batch):
    x = np.random.randint(0, 10, size=2 * batch)
    x = x.reshape([batch, -1])
    y = np.dot(x, X)
    y = np.reshape(y, [-1, 1])
    return x, y


optimizer = tf.train.GradientDescentOptimizer(LR)

input_x = tf.placeholder(tf.float32, [None, 2])
input_y = tf.placeholder(tf.float32, [None, 1])


def weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name="bias")


#w1 = weight((2, 128))
#b1 = bias((128, ))
l1 = tf.layers.dense(input_x, 128)

#w2 = weight((128, 1))
l2 = tf.layers.dense(l1, 1)

#l1 = tf.multiply(input_x, w1)# + b1

#l2 = tf.multiply(l1, w2)

loss = tf.nn.l2_loss(input_y - l2) / BATCH_SIZE

train_op = optimizer.minimize(loss)


sess = tf.Session()

sess.run(tf.global_variables_initializer())


for i in range(MAX_TRAIN_STEPS):
    x, y = next_batch(BATCH_SIZE)

    v_loss, v_l1 = sess.run([train_op, loss, l1], feed_dict={input_x: x, input_y: y})[1:]

    if (i+1) % 100 == 0:
        print('step: {0} loss: {1}'.format(i, v_loss))



x, y = next_batch(10)

result = sess.run(l2, feed_dict={input_x: x, input_y: y})

print(result)
print('='*100)
print(y)