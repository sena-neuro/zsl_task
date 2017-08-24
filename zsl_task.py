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
    num_of_tr_samples = 100
    X_tr = tf.placeholder(tf.float32, [num_of_tr_samples, 2048])
    W = tf.Variable(tf.zeros([2048, 312]))  # init with Xavier

    # Define loss
    S_guess = tf.matmul(X_tr, W)          # Attributes [None, 100]
    S_gt = tf.placeholder(tf.float32, [100, 312])
    S_corr = tf.matmul(S_guess, tf.transpose(S_gt))
    L_tr_oh = tf.placeholder(tf.float32, [num_of_tr_samples, 100])
    L_tr_temp = tf.multiply(L_tr_oh, S_corr)
    L_tr_corr = tf.reduce_sum(L_tr_temp, 1, keep_dims=True)
    loss_temp = tf.subtract(S_corr, L_tr_corr)
    L_tr_oh_dual = tf.ones([num_of_tr_samples, 100]) - L_tr_oh
    loss_matrix = tf.add(loss_temp, L_tr_oh_dual)
    max_scores = tf.maximum(loss_matrix, tf.zeros([num_of_tr_samples, 100]))
    loss = tf.reduce_mean(tf.reduce_sum(max_scores))
    # ???
    # Define optimizer

    # learning_rate = 0.001     # TODO: decaying learning rate
    # Decaying learning rate but might not be necessary
    global_step = tf.Variable(0, name='global_step', trainable=False)
    starter_learning_rate = 0.1
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               1000, 0.96, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    num_of_val_samples = 2946
    X_va = tf.placeholder(tf.float32, [num_of_val_samples, 2048])
    L_va_oh = tf.placeholder(tf.float32, [num_of_val_samples, 50])
    S_va_gt = tf.placeholder(tf.float32, [50, 312])
    S_va_guess = tf.matmul(X_va, W)
    S_va_corr = tf.matmul(S_va_guess, tf.transpose(S_va_gt))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(10000):
            batch_X_tr, batch_L_tr_oh = ut.next_batch(100, dataset['Xtr'], dataset['Ltr_oh'])

            # to get loss i changed below line
            _, loss_cur = sess.run([train_op, loss], feed_dict={X_tr: batch_X_tr, S_gt: dataset['Str_gt'], L_tr_oh: batch_L_tr_oh})
            if i % 10 == 0:
                print 'current loss = %s' % loss_cur

        accuracy = calculate_accuracy(S_va_corr, L_va_oh)

        print accuracy.eval(feed_dict={X_va: dataset['Xva'], L_va_oh: dataset['Lva_oh'], S_va_gt: dataset['Sva_gt'] })

main()

