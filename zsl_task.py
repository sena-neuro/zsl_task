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
    #num_of_tr_samples = 100
    X_tr = tf.placeholder(tf.float32, [None, 2048])
    W = tf.Variable(tf.zeros([2048, 312]))  # init with Xavier

    # Define loss
    S_guess = tf.matmul(X_tr, W)                                   # Attributes [None, 100]
    S_gt = tf.placeholder(tf.float32, [None, 312])                  # Ground Truth for test attributes
    S_corr = tf.matmul(S_guess, tf.transpose(S_gt))
    L_tr_oh = tf.placeholder(tf.float32, [None, None]) # Training labels one hot encoded
    L_tr_temp = tf.multiply(L_tr_oh, S_corr)
    L_tr_corr = tf.reduce_sum(L_tr_temp, 1, keep_dims=True)
    loss_temp = tf.subtract(S_corr, L_tr_corr)
    L_tr_oh_dual = tf.ones_like(L_tr_oh) - L_tr_oh
    loss_matrix = tf.add(loss_temp, L_tr_oh_dual)
    max_scores = tf.maximum(loss_matrix,tf.zeros_like(loss_matrix))
    loss = tf.reduce_mean(tf.reduce_sum(max_scores))
    # Define optimizer

    # Decaying learning rate but might not be necessary
    global_step = tf.Variable(0, name='global_step', trainable=False)
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               500, 0.98, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)


    accuracy = calculate_accuracy(S_corr, L_tr_oh)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(10000):
            batch_X_tr, batch_L_tr_oh = ut.next_batch(100, dataset['Xtr'], dataset['Ltr_oh'])

            # to get loss i changed below line
            _, loss_cur, train_acc,lr = sess.run([train_op, loss, accuracy, learning_rate],
                                              feed_dict={X_tr: batch_X_tr, S_gt: dataset['Str_gt'],
                                                         L_tr_oh: batch_L_tr_oh})

            val_acc = accuracy.eval(
                feed_dict={X_tr: dataset['Xva'], L_tr_oh: dataset['Lva_oh'], S_gt: dataset['Sva_gt']})
            if i % 10 == 0:
                print 'Batch no = {0}, current loss = {1}, Current training accuracy = {2},' \
                      ' current validation accuracy = {3}'.format(i,loss_cur,train_acc,val_acc)

main()

