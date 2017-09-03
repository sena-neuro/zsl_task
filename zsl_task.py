import os
import tensorflow as tf
import utils as ut
import numpy as np
def main(reg_const, learning_rates,n_iters):

    # directory for summaries
    summaries_dir = "/home/huseyin/Work/Ml/Projects/zsl_task/summaries"

    # Import data
    if os.uname()[1] == 'zen': dset_path = "/home/mbs/Datasets/zsl/gbu_CUB_data.mat"
    else: dset_path = "gbu_CUB_data.mat"
    dset = ut.get_data(dset_path)

    # Create the model
    # Bulent: Its good habit to separate placeholders from the rest of the model
    #         Since you can always watch out for feeds.
    #         Also consider naming placeholders, it helps solving certain bugs.
    with tf.name_scope("Inputs"):
        X = tf.placeholder(tf.float32, [None, 2048], name='features')
        L_oh = tf.placeholder(tf.float32, [None, None], name='one-hot_labels')                         # Labels one hot encoded
        S_gt = tf.placeholder(tf.float32, [None, 312], name='class_embeddings')                          # Ground Truth for test attributes
        reg_constant = tf.placeholder(tf.float32, [], name='regularization_constant')
        starter_learning_rate = tf.placeholder(tf.float32,[],name= 'learning_rate')

    with tf.name_scope("Model"):
        """
        More Layers will be added here
        When doing BN, the BN'd layer will not have a bias
        """
        W = tf.Variable(tf.zeros([2048, 312]))
        S_guess = tf.matmul(X, W)
        S_corr = tf.matmul(S_guess, tf.transpose(S_gt))

        # Add Weights to summary
        tf.summary.histogram("Weights", W)

    # Define loss
    with tf.name_scope("Calculate_Loss"):
        Corr_oh = tf.reduce_sum(tf.multiply(L_oh, S_corr), 1, keep_dims=True)   # Correlations one hot encoded
        L_oh_dual = tf.ones_like(L_oh) - L_oh
        loss_matrix = tf.add(tf.subtract(S_corr, Corr_oh), L_oh_dual)
        max_scores = tf.maximum(loss_matrix, tf.zeros_like(loss_matrix))
        unregularized_loss = tf.reduce_mean(tf.reduce_sum(max_scores, 1))

        # Add loss to summary
        tf.summary.scalar("loss", unregularized_loss)

        # Regularization
        # Bulent: typical regularization constant ranges between [1e-3, 1e-6],
        #         it should be tuned on validation set
        l2_loss = reg_constant * tf.nn.l2_loss(W)
        loss = tf.add(unregularized_loss, l2_loss)

    # Define optimizer
    # Bulent: Ive never seen a learning rate that decreases too rapid,
    #         Tune deceleration on validation set
    with tf.name_scope("Optimizer"):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   500, 0.996, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)

    # Training step
    with tf.name_scope("train"):
        train_op = optimizer.minimize(loss, global_step=global_step)

    # Accuracy
    with tf.name_scope("accuracy"):
        with tf.name_scope("correct_predictions"):
            correct_predictions = tf.equal(tf.argmax(S_corr, 1), tf.argmax(L_oh, 1))
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

    # Bulent: Measuring loss of a batch of samples right after processing them does not
    #         reflect the truth. Its good practice to evaluate the model after predefined time
    #         intervals, i.e. epochs.

    summary_op = tf.summary.merge_all()
    with tf.Session() as sess:

        # Experiments results will be here
        results = []
        tf.global_variables_initializer().run()

        # Writers for tensorboard
        train_writer = tf.summary.FileWriter(summaries_dir + '/test', graph=tf.get_default_graph())
        val_writer = tf.summary.FileWriter(summaries_dir + '/val', graph=tf.get_default_graph())

        for i in xrange(learning_rates.size):
            for it in xrange(1, n_iters + 1):
                batch_X_tr, batch_L_tr_oh = ut.next_batch(100, dset['Xtr'], dset['Ltr_oh'])
                sess.run(
                    [train_op],
                    {
                        X: batch_X_tr,
                        L_oh: batch_L_tr_oh,
                        S_gt: dset['Str_gt'],
                        reg_constant: reg_const[i],
                        starter_learning_rate: learning_rates[i]
                    }
                )

                # evaluate model once in a while
                if it % 1000 == 0:
                    tr_acc, _unregLoss,summ_train = sess.run(
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
