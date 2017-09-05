import os
import tensorflow as tf
import utils as ut
import numpy as np


def main(reg_const, learning_rates, n_iters):

    # Get current directory
    cwd = os.getcwd()

    # Directory for summaries that Tensorboard uses
    summaries_dir = cwd + "/summaries"

    # Import data
    if os.uname()[1] == 'zen': dset_path = "/home/mbs/Datasets/zsl/gbu_CUB_data.mat"
    else: dset_path = "gbu_CUB_data.mat"
    dset = ut.get_data(dset_path)

    # Create the model
    with tf.name_scope("Inputs"):
        X = tf.placeholder(tf.float32, [None, 2048], name='features')
        L_oh = tf.placeholder(tf.float32, [None, None], name='one-hot_labels')
        S_gt = tf.placeholder(tf.float32, [None, 312], name='class_embeddings')
        reg_constant = tf.placeholder(tf.float32, [], name='regularization_constant')
        starter_learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')

    with tf.name_scope("Model"):
        W = tf.Variable(tf.zeros([2048, 312]), name="weight")
        S_guess = tf.matmul(X, W, name="guessed_Attributes")
        S_corr = tf.matmul(S_guess, tf.transpose(S_gt), name="correlations_with_GT_attributes")

        # Add Weights to summary
        tf.summary.histogram("Weights", W)

    # Define loss
    with tf.name_scope("Calculate_Loss"):
        Corr_oh = tf.reduce_sum(tf.multiply(L_oh, S_corr), 1, keep_dims=True, name="correlations_of_only_correct_classes")
        L_oh_dual = tf.subtract(tf.ones_like(L_oh), L_oh, name="dual_of_one_hot_labels")
        loss_matrix = tf.add(tf.subtract(S_corr, Corr_oh), L_oh_dual, name="Matrix_of_losses")
        unregularized_loss = tf.reduce_mean(tf.reduce_sum(tf.maximum(loss_matrix, tf.zeros_like(loss_matrix)), 1),name="unregularized_loss")

        # Add loss to summary
        tf.summary.scalar("loss", unregularized_loss)

        # Regularization
        l2_loss = reg_constant * tf.nn.l2_loss(W)
        loss = tf.add(unregularized_loss, l2_loss, name="loss")

    # Define optimizer
    with tf.name_scope("Optimizer"):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   500, 0.996, staircase=True, name="learning_rate")
        optimizer = tf.train.AdamOptimizer(learning_rate,name="Adam_optimizer")

    # Training step
    with tf.name_scope("train"):
        train_op = optimizer.minimize(loss, global_step=global_step, name="train_step")

    # Accuracy
    with tf.name_scope("accuracy"):
        with tf.name_scope("correct_predictions"):
            correct_predictions = tf.equal(tf.argmax(S_corr, 1), tf.argmax(L_oh, 1), name="correct_predictions")
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="top_one_accuracy")
    tf.summary.scalar("accuracy", accuracy)

    # Mergin all summaries
    summary_op = tf.summary.merge_all()
    with tf.Session() as sess:

        # Results will be here
        results = []
        tf.global_variables_initializer().run()

        # Writers for tensorboard
        train_writer = tf.summary.FileWriter(summaries_dir + '/test', graph=tf.get_default_graph())
        val_writer = tf.summary.FileWriter(summaries_dir + '/val', graph=tf.get_default_graph())

        for i in xrange(learning_rates.size):
            for it in xrange(1, n_iters + 1):
                batch_X, batch_L_oh = ut.next_batch(100, dset['Xtr'], dset['Ltr_oh'])
                sess.run(
                    [train_op],
                    {
                        X: batch_X,
                        L_oh: batch_L_oh,
                        S_gt: dset['Str_gt'],
                        reg_constant: reg_const[i],
                        starter_learning_rate: learning_rates[i]
                    }
                )

                # evaluate model once in a while
                if it % 1000 == 0:
                    tr_acc, _unregLoss, summ_train = sess.run(
                        [accuracy, unregularized_loss,summary_op],
                        {
                            X: dset['Xtr'],
                            L_oh: dset['Ltr_oh'],
                            S_gt: dset['Str_gt'],
                            starter_learning_rate: learning_rates[i]
                        }
                    )
                    va_acc, summ_val = sess.run(
                        [accuracy, summary_op],
                        {
                            X: dset['Xva'],
                            L_oh: dset['Lva_oh'],
                            S_gt: dset['Sva_gt'],
                            starter_learning_rate: learning_rates[i]
                        }
                    )
                    print 'Iter: {0:05}, loss = {1:09.5f}, acc_tr = {2:0.4f}, acc_va = {3:0.4f}' .format(
                        it, _unregLoss, tr_acc, va_acc)
                    train_writer.add_summary(summ_train,global_step=it)
                    val_writer.add_summary(summ_val,global_step=it)

            # Final validation accuracy calculation
            va_acc, loss_val = sess.run(
                [accuracy, unregularized_loss],
                {
                    X: dset['Xva'],
                    L_oh: dset['Lva_oh'],
                    S_gt: dset['Sva_gt'],
                    starter_learning_rate: learning_rates[i]
                }
            )
            
            # Append resulted validation information to results
            results.append([learning_rates[i],reg_const[i],va_acc,loss_val])
        train_writer.close()
        val_writer.close()
        return results

main(np.asarray([5e-3]), np.asarray([1.38e-5]), 100000)