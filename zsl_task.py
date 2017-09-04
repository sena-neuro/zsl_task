import os
import tensorflow as tf
import utils as ut
import numpy as np
def main(reg_consts,learning_rates,n_iters,drop_outs):

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
        starting_learning_rate = tf.placeholder(tf.float32,name = "starting_learning_rate")
        drop_out_constant = tf.placeholder(tf.float32,[], name = "dropOut_constant")

    with tf.name_scope("Variables"):
        '''Variables'''
        # Hidden RELU layer 1
        bias_1 = tf.Variable(tf.zeros([1024]))
        weights_1 = tf.get_variable("weights_1", shape=([2048, 1024]),
                                    initializer=tf.contrib.layers.xavier_initializer())
        # Hidden RELU layer 1
        bias_2 = tf.Variable(tf.zeros([512]))
        weights_2 = tf.get_variable("weights_2", shape=([1024, 512]),
                                    initializer=tf.contrib.layers.xavier_initializer())

        # Hidden RELU layer 1
        bias_3 = tf.Variable(tf.zeros([312]))
        weights_3 = tf.get_variable("weights_3", shape=([512, 312]),
                                    initializer=tf.contrib.layers.xavier_initializer())

    with tf.name_scope("Model"):

        # Hidden RELU layer 1
        logits_1 = tf.matmul(X, weights_1) + bias_1
        hidden_layer_1 = tf.nn.relu(logits_1)
        hidden_layer_1_dropout = tf.nn.dropout(hidden_layer_1, drop_out_constant,)

        # Hidden RELU layer 2
        logits_2 = tf.matmul(hidden_layer_1_dropout, weights_2) + bias_2
        hidden_layer_2 = tf.nn.relu(logits_2)
        hidden_layer_2_dropout = tf.nn.dropout(hidden_layer_2, drop_out_constant)

        # Output layer
        S_guess = tf.matmul(hidden_layer_2_dropout, weights_3) + bias_3  # should i not relu
        S_corr = tf.matmul(S_guess, tf.transpose(S_gt))

        # Add Weights to summary
        ut.variable_summaries(weights_1)
        ut.variable_summaries(weights_2)
        ut.variable_summaries(weights_3)

    # Define loss
    with tf.name_scope("Calculate_Loss"):

        # Define loss
        Corr_oh = tf.reduce_sum(tf.multiply(L_oh, S_corr), 1, keep_dims=True)  # Correlations one hot encoded
        L_oh_dual = tf.ones_like(L_oh) - L_oh
        loss_matrix = tf.add(tf.subtract(S_corr, Corr_oh), L_oh_dual)
        max_scores = tf.maximum(loss_matrix, tf.zeros_like(loss_matrix))
        unregularized_loss = tf.reduce_mean(tf.reduce_sum(max_scores, 1))
        regularizers = reg_constant * tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) + tf.nn.l2_loss(weights_3)

        # Loss
        loss = tf.add(unregularized_loss, regularizers)
        tf.summary.scalar("loss", unregularized_loss)

    # Define optimizer
    # Bulent: Ive never seen a learning rate that decreases too rapid,
    #         Tune deceleration on validation set
    with tf.name_scope("Optimizer"):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(starting_learning_rate, global_step,
                                                   1000, 0.996, staircase=True)
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
        tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter(summaries_dir + '/test', graph=tf.get_default_graph())
        val_writer = tf.summary.FileWriter(summaries_dir + '/val', graph=tf.get_default_graph())
        results = []
        for i in range(learning_rates.size):
            for it in xrange(1, n_iters + 1):
                batch_X_tr, batch_L_tr_oh = ut.next_batch(100, dset['Xtr'], dset['Ltr_oh'])
                sess.run(
                    [train_op],
                    {
                        X: batch_X_tr,
                        L_oh: batch_L_tr_oh,
                        S_gt: dset['Str_gt'],
                        reg_constant: reg_consts[i],
                        starting_learning_rate: learning_rates[i],
                        drop_out_constant:drop_outs[i]
                    }
                )
                # evaluate model once in a while
                if it % 1000 == 0:
                    tr_acc, _unreg_train_loss,summ_train = sess.run(
                        [accuracy, unregularized_loss,summary_op],
                        {
                            X: dset['Xtr'],
                            L_oh: dset['Ltr_oh'],
                            S_gt: dset['Str_gt'],
                            starting_learning_rate: learning_rates[i],
                            drop_out_constant: 1
                        }
                    )
                    va_acc,_unreg_val_Loss, summ_val = sess.run(
                        [accuracy, unregularized_loss,summary_op],
                        {
                            X: dset['Xva'],
                            L_oh: dset['Lva_oh'],
                            S_gt: dset['Sva_gt'],
                            starting_learning_rate: learning_rates[i],
                            drop_out_constant: 1
                        }
                    )
                    print 'Iter: {0:05}, loss_tr = {1:09.5f}, acc_tr = {2:0.4f}, loss_va = {3:09.5f}, acc_va = {4:0.4f}' \
                        .format(it, _unreg_train_loss, tr_acc, _unreg_val_Loss, va_acc)

                    train_writer.add_summary(summ_train,global_step=it)
                    val_writer.add_summary(summ_val,global_step=it)
                train_writer.close()
                val_writer.close()
            # Final accuracy for this setting
            va_acc, _unreg_val_Loss = sess.run(
                [accuracy, unregularized_loss],
                {
                    X: dset['Xva'],
                    L_oh: dset['Lva_oh'],
                    S_gt: dset['Sva_gt'],
                    drop_out_constant: 1
                }
            )
            results.append([learning_rates[i],reg_consts[i],drop_outs[i],va_acc,_unreg_val_Loss])
        return results