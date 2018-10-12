import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 12, 128])
y = tf.placeholder(tf.float32, [None, 1, 128])

print(x + y)