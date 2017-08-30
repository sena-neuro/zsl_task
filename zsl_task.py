import os
import tensorflow as tf
import utils as ut
import numpy as np


def calculate_accuracy(preds, labels):
    correct_predictions = tf.equal(tf.argmax(labels, 1), tf.argmax(preds, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy

def main():

    # Import data
    # relative dataset paths should be supported
    if os.uname()[1] == 'zen': dset_path = "/home/mbs/Datasets/zsl/gbu_CUB_data.mat"
    else: dset_path = "gbu_CUB_data.mat"
    dset = ut.get_data(dset_path)

    # Create the model
    # Bulent: Its good habit to separate placeholders from the rest of the model
    #         Since you can always watch out for feeds.
    #         Also consider naming placeholders, it helps solving certain bugs.
    X = tf.placeholder(tf.float32, [None, 2048], name='features')
    L_oh = tf.placeholder(tf.float32, [None, None], name='one-hot_labels')                         # Labels one hot encoded
    S_gt = tf.placeholder(tf.float32, [None, 312], name='class_embeddings')                          # Ground Truth for test attributes
    reg_constant = tf.placeholder(tf.float32, [], name='regularization_constant')

    W = tf.Variable(tf.zeros([2048, 312]))

    # Define loss
    S_guess = tf.matmul(X, W)                                               # Attributes [None, 312]
    S_corr = tf.matmul(S_guess, tf.transpose(S_gt))
    Corr_oh = tf.reduce_sum(tf.multiply(L_oh, S_corr), 1, keep_dims=True)   # Correlations one hot encoded
    L_oh_dual = tf.ones_like(L_oh) - L_oh
    loss_matrix = tf.add(tf.subtract(S_corr, Corr_oh), L_oh_dual)
    max_scores = tf.maximum(loss_matrix, tf.zeros_like(loss_matrix))
    unregularized_loss = tf.reduce_mean(tf.reduce_sum(max_scores, 1))

    # Regularization
    # Bulent: typical regularization constant ranges between [1e-3, 1e-6], 
    #         it should be tuned on validation set
    l2_loss = reg_constant * tf.nn.l2_loss(W)

    # Loss
    loss = tf.add(unregularized_loss, l2_loss)

    # Define optimizer
    # Bulent: Ive never seen a learning rate that decreases too rapid,
    #         Tune deceleration on validation set
    global_step = tf.Variable(0, name='global_step', trainable=False)
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               500, 0.996, staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    accuracy = calculate_accuracy(S_corr, L_oh)

    # Bulent: Measuring loss of a batch of samples right after processing them does not
    #         reflect the truth. Its good practice to evaluate the model after predefined time
    #         intervals, i.e. epochs.
    n_iters = 100000

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for it in xrange(1, n_iters + 1):
            batch_X_tr, batch_L_tr_oh = ut.next_batch(100, dset['Xtr'], dset['Ltr_oh'])
            sess.run(
                train_op,
                {
                    X: batch_X_tr, 
                    L_oh: batch_L_tr_oh, 
                    S_gt: dset['Str_gt'],
                    reg_constant: 1e-4
                }
            )

            # evaluate model once in a while
            if it % 1000 == 0:
                tr_acc, _unregLoss = sess.run(
                    [accuracy, unregularized_loss],
                    {
                        X: dset['Xtr'],
                        L_oh: dset['Ltr_oh'],
                        S_gt: dset['Str_gt']
                    }
                )

                va_acc = sess.run(
                    accuracy,
                    {
                        X: dset['Xva'],
                        L_oh: dset['Lva_oh'],
                        S_gt: dset['Sva_gt']
                    }
                )

                print 'Iter: {0:05}, loss = {1:09.5f}, acc_tr = {2:0.4f}, acc_va = {2:0.4f}' .format(
                    it, _unregLoss, tr_acc, va_acc)

main()

