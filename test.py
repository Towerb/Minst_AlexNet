import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", one_hot=True)


with tf.Session() as sess:
    meta_dir = "./runs/1543995887/checkpoints/model-2000.meta"
    print("restore model from {}".format(meta_dir))
    saver = tf.train.import_meta_graph(meta_dir)
    saver.restore(sess, tf.train.latest_checkpoint("".join(meta_dir.split("/")[0:-1])))
    graph = tf.get_default_graph()

    x_input = graph.get_tensor_by_name("input/x_input:0")
    y_input = graph.get_tensor_by_name("input/y_input:0")
    keep_dropout = graph.get_tensor_by_name("keep_dropout/keep_dropout:0")
    accuracy = graph.get_tensor_by_name("accuracy/accuracy:0")
    accuracy = sess.run(accuracy, feed_dict={x_input: mnist.test.images[:200, :],
                                             y_input: mnist.test.labels[:200, :],
                                             keep_dropout: 1.0
                                             })
    print("Test accuracy: {}".format(accuracy))
