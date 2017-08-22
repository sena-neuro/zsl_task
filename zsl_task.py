import tensorflow as tf
import utils as ut


def main():
    # Import data
    dataset = ut.get_data('gbu_CUB_data.mat')

    # Create the model
    # Varibles
    """ In order to use this model for all data without writing more, i wanted to make the numbers that varies 
    between datasets into variables
    """
    num_of_samples = tf.placeholder(tf.float32)
    num_of_classes = tf.placeholder(tf.float32)
    num_of_attributes = tf.placeholder(tf.float32)

    X_tr = tf.placeholder(tf.float32, [num_of_samples, 2048])
    W = tf.Variable(tf.zeros([2048, num_of_attributes]))  # init with Xavier

    # Define loss
    S_guess = tf.matmul(X_tr, W)          # Attributes [None, 100]
    S_gt = tf.placeholder(tf.float32, [num_of_classes, num_of_attributes])
    S_corr = tf.matmul(S_guess, tf.transpose(S_gt))
    L_tr_oh = tf.placeholder(tf.float32, [num_of_samples, num_of_classes])
    L_tr_temp = tf.multiply(L_tr_oh, S_corr)
    L_tr_corr = tf.reduce_sum(L_tr_temp, 1, keep_dims=True)
    loss_temp = tf.subtract(S_corr, L_tr_corr)
    L_tr_oh_dual = tf.ones([num_of_samples, num_of_classes]) - L_tr_oh
    loss_matrix = tf.add(loss_temp, L_tr_oh_dual)
    max_scores = tf.maximum(loss_matrix, tf.zeros([num_of_samples, num_of_classes]))
    loss = tf.reduce_mean(max_scores)                                                                         # ???


    # Decaying learning rate but might not be necessary
    global_step = tf.Variable(0, name='global_step', trainable=False)
    starter_learning_rate = 0.1
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               1000, 0.96, staircase=True)

    # Define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    # Accuracy calculation

    correct_predictions = tf.equal(tf.argmax(L_tr_oh, 1), tf.argmax(S_corr, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        feed_dict = {}
        for i in range(10000):

            batch_X_tr, batch_L_tr_oh = ut.next_batch(100, dataset['Xtr'], dataset['Ltr_oh'])
            # to get loss and accuracy i changed below line
            batch_no, loss_cur, cur_test_accuracy = sess.run([train_op, loss, accuracy], feed_dict={
                X_tr: batch_X_tr, S_gt: dataset['Str_gt'], L_tr_oh: batch_L_tr_oh,
                num_of_samples: 100 , num_of_classes: batch_L_tr_oh.shape[1] ,
                num_of_attributes: dataset['Str_gt'].shape[1]
            })

            if i % 10 == 0:
                # changed formatting
                print 'On batch no {0}: current loss = {1}, current training accuracy = {2} '.format(batch_no, loss_cur, cur_test_accuracy)

    # Can we print accuracy on evaluation dataset on every 10 batches too, meaning that while it is training

    print accuracy.eval(feed_dict={
        X_tr: dataset['Xva'], L_tr_oh:dataset['Lva_oh'], S_gt: dataset['Sva_gt'],
        num_of_samples:dataset['Xva'].shape[0], num_of_classes: dataset['Lva_oh'].shape[1],
        num_of_attributes: dataset['Sva_gt'].shape[1]
    })

main()

