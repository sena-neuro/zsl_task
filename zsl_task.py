import tensorflow as tf
import scipy.io as sio

def main(_):
    # Import data
    sio
    # Create the model
    X_tr = tf.placeholder(tf.float32, [None, 2048])
    W = tf.Variable(tf.float32, [2048, 100])  # init with Xavier

    # Define loss
    S_guess = tf.matmul(X_tr, W)          # Attributes [None, 100]
    S_gt = tf.placeholder(tf.float32, [100, 150])
    S_corr = tf.matmul(S_guess, S_gt)
    L_tr_oh = tf.placeholder(tf.float32, [None, 150])
    L_tr_temp = tf.multiply(L_tr_oh, S_corr)
    L_tr = tf.reduce_sum(L_tr_temp, 1, keep_dims=True)
    loss_temp = tf.subtract(S_corr, L_tr)
    L_tr_oh_dual = tf.ones([None, 150]) - L_tr_oh
    loss = tf.add(loss_temp, L_tr_oh_dual)

    # Define optimizer
    learning_rate = 0.001     # TODO: decaying learning rate
    optimizer = tf.train.AdamOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    with tf.InteractiveSession() as sess:
        tf.global_variables_initializer().run()

        # TODO: feed_dict
        feed_dict = {}
        for _ in range(1000):
            # TODO: import data
            batch_X_tr, batch_S_gt, batch_L_tr_oh = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={X_tr: batch_X_tr:, S_gt: batch_S_gt:, L_tr_oh: batch_L_tr_oh})





