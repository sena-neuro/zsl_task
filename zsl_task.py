import tensorflow as tf
import utils as ut
import numpy as np

# train()

def calculate_accuracy(preds, labels):
    correct_predictions = tf.equal(tf.argmax(labels, 1), tf.argmax(preds, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy

def main():
    # Import data
    dataset = ut.get_data('gbu_CUB_data.mat')

    # Create the model
    X = tf.placeholder(tf.float32, [None, 2048])
    W = tf.Variable(tf.zeros([2048, 312]))

    # Define loss
    S_guess = tf.matmul(X, W)                                               # Attributes [None, 312]
    S_gt = tf.placeholder(tf.float32, [None, 312])                          # Ground Truth for test attributes
    S_corr = tf.matmul(S_guess, tf.transpose(S_gt))
    L_oh = tf.placeholder(tf.float32, [None, None])                         # Labels one hot encoded
    Corr_oh = tf.reduce_sum(tf.multiply(L_oh, S_corr), 1, keep_dims=True)   # Correlations one hot encoded
    L_oh_dual = tf.ones_like(L_oh) - L_oh
    loss_matrix = tf.add(tf.subtract(S_corr, Corr_oh), L_oh_dual)
    max_scores = tf.maximum(loss_matrix, tf.zeros_like(loss_matrix))
    unregularized_loss = tf.reduce_mean(tf.reduce_sum(max_scores))


    # Regularization
    reg_constant = tf.placeholder(tf.float32)
    l2_loss = reg_constant * tf.nn.l2_loss(W)

    # Loss
    loss = tf.add(unregularized_loss,l2_loss)

    # Define optimizer

    global_step = tf.Variable(0, name='global_step', trainable=False)
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               500, 0.96, staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    accuracy = calculate_accuracy(S_corr, L_oh)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(10000):
            batch_X_tr, batch_L_tr_oh = ut.next_batch(100, dataset['Xtr'], dataset['Ltr_oh'])

            # to get loss i changed below line
            _, loss_cur, train_acc,L2loss = sess.run([train_op, loss, accuracy, l2_loss],
                                              feed_dict={X: batch_X_tr, S_gt: dataset['Str_gt'],
                                                         L_oh: batch_L_tr_oh, reg_constant: 3})

            val_acc = accuracy.eval(
                feed_dict={X: dataset['Xva'], L_oh: dataset['Lva_oh'], S_gt: dataset['Sva_gt']})
            if i % 10 == 0:
                print 'Batch no = {0}, current loss = {1}, Current training accuracy = {2},' \
                      ' current validation accuracy = {3}, reg loss: {4}'.format(i,loss_cur,train_acc,val_acc, L2loss)

        fin_val_acc = accuracy.eval(
            feed_dict={X: dataset['Xva'], L_oh: dataset['Lva_oh'], S_gt: dataset['Sva_gt']})

        print fin_val_acc
main()

