import tensorflow as tf
import load_data_test as ld
import numpy as np

def main(_):
    # Import data
    dataset = ld.get_training_data('gbu_CUB_data.mat')

    # Create the model
    m = tf.placeholder(tf.float32)
    X_tr = tf.placeholder(tf.float32, [m, 2048])
    W = tf.Variable(tf.float32, [2048, 100])  # init with Xavier

    # Define loss
    S_guess = tf.matmul(X_tr, W)          # Attributes [None, 100]
    S_gt = tf.placeholder(tf.float32, [100, 312])
    S_corr = tf.matmul(S_guess, S_gt)
    L_tr_oh = tf.placeholder(tf.float32, [m, 100])
    L_tr_temp = tf.multiply(L_tr_oh, S_corr)
    L_tr = tf.reduce_sum(L_tr_temp, 1, keep_dims=True)
    loss_temp = tf.subtract(S_corr, L_tr)
    L_tr_oh_dual = tf.ones([m, 100]) - L_tr_oh
    loss = tf.add(loss_temp, L_tr_oh_dual)

    # Define optimizer
    learning_rate = 0.001     # TODO: decaying learning rate
    optimizer = tf.train.AdamOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    with tf.InteractiveSession() as sess:
        tf.global_variables_initializer().run()

        X_tr = dataset['Xtr']
        S_gt = dataset['Str_gt']
        L_tr_oh = dataset['Ltr_oh']

        # TODO: feed_dict
        feed_dict = {}
        for _ in range(1000):
            # TODO: import data
            batch_X_tr = ld.getBAATCHED('Xtr')
            batch_S_gt = ld.getBAATCHED('Str_gt')
            batch_L_tr_oh = ld.getBAATCHED('Ltr_oh')
            sess.run(train_step, feed_dict={X_tr: batch_X_tr:, S_gt: batch_S_gt:, L_tr_oh: batch_L_tr_oh})





