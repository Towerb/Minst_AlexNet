import tensorflow as tf


class AlexNet(object):
    def __init__(self, n_input=784, n_classes=10):
        with tf.name_scope("input"):
            self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, n_input], name="x_input")
            self.y_input = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name="y_input")
        with tf.name_scope("keep_dropout"):
            self.keep_dropout = tf.placeholder(dtype=tf.float32, name="keep_dropout")

        with tf.name_scope("conv_1"):
            kernel1 = tf.Variable(tf.random_normal([11, 11, 1, 64]), name="kernel1")
            biases1 = tf.Variable(tf.random_normal([64]), name="biases1")
            # [None, 28, 28, 1] --> [None, 28, 28, 64]
            c1 = tf.nn.conv2d(input=tf.reshape(self.x_input, shape=[-1, 28, 28, 1]), filter=kernel1, strides=[1, 1, 1, 1],padding="SAME", name="conv1")
            c1 = tf.nn.bias_add(c1, biases1)
            # [None, 28, 28, 64] --> [None, 14, 14, 64]
            p1 = tf.nn.max_pool(c1, ksize=(1, 2, 2, 1), strides=[1, 2, 2, 1], padding="SAME")
            l1 = tf.nn.lrn(p1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="lrn1")
            d1 = tf.nn.dropout(l1, self.keep_dropout)

        with tf.name_scope("conv_2"):
            kernel2 = tf.Variable(tf.random_normal([5, 5, 64, 192]), name="kernel2")
            biases2 = tf.Variable(tf.random_normal([192]), name="biases2")
            # [None, 14, 14, 64] --> [None, 14, 14, 192]
            c2 = tf.nn.conv2d(input=d1, filter=kernel2, strides=[1, 1, 1, 1], padding="SAME", name="conv2")
            c2 = tf.nn.bias_add(c2, biases2)
            # [None, 14, 14, 192] --> [None, 7, 7, 192]
            p2 = tf.nn.max_pool(c2, ksize=(1, 2, 2, 1), strides=[1, 2, 2, 1], padding="SAME")
            l2 = tf.nn.lrn(p2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="lrn2")
            d2 = tf.nn.dropout(l2, self.keep_dropout)

        with tf.name_scope("conv_3"):
            kernel3 = tf.Variable(tf.random_normal([3, 3, 192, 384]), name="kernel3")
            biases3 = tf.Variable(tf.random_normal([384]), name="biases3")
            # [None, 7, 7, 192] --> [None, 7, 7, 384]
            c3 = tf.nn.conv2d(input=d2, filter=kernel3, strides=[1, 1, 1, 1], padding="SAME", name="conv3")
            c3 = tf.nn.bias_add(c3, biases3)

        with tf.name_scope("conv_4"):
            kernel4 = tf.Variable(tf.random_normal([3, 3, 384, 256]), name="kernel4")
            biases4 = tf.Variable(tf.random_normal([256]), name="biases4")
            # [None, 7, 7, 384] --> [None, 7, 7, 256]
            c4 = tf.nn.conv2d(input=c3, filter=kernel4, strides=[1, 1, 1, 1], padding="SAME", name="conv4")
            c4 = tf.nn.bias_add(c4, biases4)

        with tf.name_scope("conv_5"):
            kernel5 = tf.Variable(tf.random_normal([3, 3, 256, 256]), name="kernel5")
            biases5 = tf.Variable(tf.random_normal([256]), name="biases5")
            # [None, 7, 7, 256] --> [None, 7, 7, 256]
            c5 = tf.nn.conv2d(input=c4, filter=kernel5, strides=[1, 1, 1, 1], padding="SAME", name="conv5")
            c5 = tf.nn.bias_add(c5, biases5)
            # [None, 7, 7, 256] --> [None, 4, 4, 256]
            p5 = tf.nn.max_pool(c5, ksize=(1, 2, 2, 1), strides=[1, 2, 2, 1], padding="SAME")
            l5 = tf.nn.lrn(p5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="lrn5")
            d5 = tf.nn.dropout(l5, self.keep_dropout)

        with tf.name_scope("fc_6"):
            d5_shape_li = d5.get_shape().as_list()
            flatten = tf.reshape(d5, shape=[-1, d5_shape_li[1] * d5_shape_li[2] * d5_shape_li[3]], name="flatten")
            weight6 = tf.Variable(tf.random_normal([d5_shape_li[1] * d5_shape_li[2] * d5_shape_li[3], 1024]), name="weight6")
            biases6 = tf.Variable(tf.random_normal([1024]), name="biases6")
            f6 = tf.nn.relu(tf.matmul(flatten, weight6) + biases6)

        with tf.name_scope("fc_7"):
            weight7 = tf.Variable(tf.random_normal([1024, 1024]), name="weight7")
            biases7 = tf.Variable(tf.random_normal([1024]), name="biases7")
            f7 = tf.nn.relu(tf.matmul(f6, weight7) + biases7)

        with tf.name_scope("output"):
            weight8 = tf.Variable(tf.random_normal([1024, n_classes]), name="weight8")
            biases8 = tf.Variable(tf.random_normal([n_classes]), name="biases8")
            f8 = tf.nn.relu(tf.matmul(f7, weight7) + biases7)
            self.output = tf.matmul(f8, weight8) + biases8

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.y_input), name="loss")

        with tf.name_scope("train"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            self.grads_and_vars = optimizer.compute_gradients(self.loss)
            self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

        with tf.name_scope("prediction"):
            self.predictions = tf.argmax(self.output, 1, name="predictions")

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.y_input, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")