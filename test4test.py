import tensorflow as tf


x = tf.ones([32, 300, 100, 200])
filter = tf.ones([100,200,400,300]) # [height, width, output_channels, in_channels]
output_shape = tf.Variable(initial_value=[32,100,200,400])
tmp = tf.nn.conv2d_transpose(x, filter=filter, output_shape=output_shape,
                             strides=[1,1,2,2], padding='SAME', data_format='NCHW')

x = tf.transpose(x,[0,2,3,1])
#x = tf.ones([32, 100, 200, 300])
filter = tf.ones([100,200,400,300]) # [height, width, output_channels, in_channels]
output_shape = tf.Variable(initial_value=[32,100,200,400])
tmp = tf.nn.conv2d_transpose(x, filter=filter, output_shape=output_shape,
                             strides=[1,1,2,2], padding='SAME', data_format='NHWC')

