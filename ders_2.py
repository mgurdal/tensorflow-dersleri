"""
Data types
"""

import numpy as np
import tensorflow as tf

matrix_1 = tf.constant([[3, 3]])
matrix_2 = tf.constant([[3], [3]])

result = tf.matmul(matrix_1, matrix_2)

sess = tf.Session()
sonuc = sess.run(result)
print(sonuc)

state = tf.Variable(0, name="counter") # baslangic degeri, isim
print(state.name)

one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.global_variables_initializer() # degisken varsa calistirilmali

sess.run(init)
for _ in range(5):
    sess.run(update)
    print(sess.run(state))

input_1 = tf.placeholder(tf.float32) # type, shape
input_2 = tf.placeholder(tf.float32)

output = tf.multiply(input_1, input_2)

res = sess.run(output, feed_dict={input_1:[4.], input_2:[7.]})

print(res)

