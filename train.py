import os
import time
import tensorflow as tf
import datetime
from AlexNet_model import AlexNet
from tensorflow.examples.tutorials.mnist import input_data

# Parameters
tf.app.flags.DEFINE_string("module", "test", "")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.app.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
tf.app.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("checkpoint_every", 400, "Save model after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
FLAGS = tf.app.flags.FLAGS

# Load data
print("Loading data...")
mnist = input_data.read_data_sets("./data/", one_hot=True)

alexNet = AlexNet(n_classes=10)

with tf.Session() as sess:
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # define summary
    grad_summaries = []
    for g, v in alexNet.grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)
    loss_summary = tf.summary.scalar("loss", alexNet.loss)
    acc_summary = tf.summary.scalar("accuracy", alexNet.accuracy)

    # merge all the train summary
    train_summary_merged = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, "summaries", "train"), graph=sess.graph)

    # merge all the dev summary
    dev_summary_merged = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, "summaries", "dev"), graph=sess.graph)

    # checkPoint saver
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    sess.run(tf.global_variables_initializer())

    while True:
        # train loop
        x_batch, y_batch = mnist.train.next_batch(FLAGS.batch_size)
        _, step, train_summaries, loss, accuracy = sess.run([alexNet.train_op, alexNet.global_step, train_summary_merged, alexNet.loss, alexNet.accuracy],
                                                            feed_dict={alexNet.x_input: x_batch,
                                                                       alexNet.y_input: y_batch,
                                                                       alexNet.keep_dropout: 0.8}
                                                            )
        train_summary_writer.add_summary(train_summaries, step)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

        # validation
        current_step = tf.train.global_step(sess, alexNet.global_step)

        if current_step % FLAGS.evaluate_every == 0:
            print("\nEvaluation:")
            step, dev_summaries, loss, accuracy = sess.run([alexNet.global_step, dev_summary_merged, alexNet.loss, alexNet.accuracy],
                                                           feed_dict={alexNet.x_input: mnist.validation.images,
                                                                      alexNet.y_input: mnist.validation.labels,
                                                                      alexNet.keep_dropout: 1.0}
                                                           )
            dev_summary_writer.add_summary(dev_summaries, step)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            print("")

        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))


