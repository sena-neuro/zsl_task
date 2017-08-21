import tensorflow as tf
import utils as ut
import numpy as np

def main():
    # Import data
    dataset = ut.get_training_data('gbu_CUB_data.mat')

    # Create the model
    m = 100
    X_tr = tf.placeholder(tf.float32, [m, 2048])
    W = tf.Variable(tf.zeros([2048, 312]))  # init with Xavier

    # Define loss
    S_guess = tf.matmul(X_tr, W)          # Attributes [None, 100]
    S_gt = tf.placeholder(tf.float32, [100, 312])
    S_corr = tf.matmul(S_guess, tf.transpose(S_gt))
    L_tr_oh = tf.placeholder(tf.float32, [m, 100])
    L_tr_temp = tf.multiply(L_tr_oh, S_corr)
    L_tr = tf.reduce_sum(L_tr_temp, 1, keep_dims=True)
    loss_temp = tf.subtract(S_corr, L_tr)
    L_tr_oh_dual = tf.ones([m, 100]) - L_tr_oh
    loss = tf.add(loss_temp, L_tr_oh_dual)

    # Accuracy # TODO: Compute accuracy


    # Define optimizer

    #learning_rate = 0.001     # TODO: decaying learning rate
    # Decaying learning rate but might not be necessary
    global_step = tf.Variable(0, name='global_step', trainable=False)
    starter_learning_rate = 0.1
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               1000, 0.96, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        feed_dict = {}
        for i in range(10000):
            batch_X_tr, batch_L_tr_oh = ut.next_batch(100, dataset['Xtr'], dataset['Ltr_oh'])
            Str_gt = dataset['Str_gt']

            # to get loss i changed below line
            _, loss_cur = sess.run([train_op, loss], feed_dict={X_tr: batch_X_tr, S_gt: Str_gt, L_tr_oh: batch_L_tr_oh})
            if i % 100 == 0:
                print 'current loss = %s' % loss_cur


main()

