# pylint: disable=invalid-name,redefined-outer-name

import tensorflow as tf

from dataset import get_split, build_dataset, build_aug_pipeline
from model import build_fcn_graph

if __name__ == '__main__':

    height, width = 384, 1280
    batch_size = 4

    # Create dataset
    train_files, validation_files = get_split('data_road/training', (200, 89), seed=1)

    train_dataset = build_dataset(train_files, size=(height, width))
    validation_dataset = build_dataset(validation_files, size=(height, width))

    # TODO
    # train_dataset = build_aug_pipeline(train_dataset)

    train_dataset = train_dataset.shuffle(32).batch(batch_size).prefetch(batch_size * 2)
    validation_dataset = validation_dataset.batch(batch_size).prefetch(batch_size * 2)

    train_iter = train_dataset.make_initializable_iterator()
    validation_iter = validation_dataset.make_initializable_iterator()

    train_next = train_iter.get_next()
    validation_next = validation_iter.get_next()


    # Define model graph
    num_classes = 2

    x = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='x')
    y = tf.placeholder(tf.uint8, shape=[None, None, None], name='y')

    logits = build_fcn_graph(x, 'fcn-8s', num_classes=num_classes)

    mask = tf.not_equal(y, 255)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        tf.boolean_mask(tf.one_hot(y, num_classes), mask),
        tf.boolean_mask(logits, mask)))
    loss = tf.add(loss, tf.losses.get_regularization_loss(), name='total_loss')

    global_step = tf.Variable(1, trainable=False)
    learning_rate = tf.train.exponential_decay(
        1e-4, global_step, 1 * len(train_files), 0.99, staircase=True)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(loss, global_step=global_step)

    tf.summary.scalar('learning_rate', learning_rate)


    # Metrics
    c_matrix = tf.math.confusion_matrix(
        tf.boolean_mask(y, mask),
        tf.boolean_mask(tf.argmax(logits, axis=3, name='y_pred'), mask),
        num_classes=num_classes)

    tp = tf.diag_part(c_matrix)
    fp = tf.reduce_sum(c_matrix, axis=0) - tp
    fn = tf.reduce_sum(c_matrix, axis=1) - tp

    pixel_acc = tf.reduce_sum(tp) / tf.reduce_sum(c_matrix)
    mean_iou = tf.reduce_mean(tp / (tp + fp + fn))

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('pixel_accuracy', pixel_acc)
    tf.summary.scalar('mean_iou', mean_iou)
    summaries = tf.summary.merge_all()


    # Tensorboard and checkpointing
    saver = tf.train.Saver()

    import datetime
    timenow = datetime.datetime.now()
    writer_train = tf.summary.FileWriter(f'train/tensorboard/{timenow}/train',
                                         tf.get_default_graph())
    writer_val = tf.summary.FileWriter(f'train/tensorboard/{timenow}/validation')


    def sample(sess, next_op):
        ''' Returns a mininatch from an iterator '''
        while True:
            try:
                yield sess.run(next_op)
            except tf.errors.OutOfRangeError:
                break


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        max_epochs = 500

        for epoch in range(max_epochs):

            sess.run(train_iter.initializer)
            for batch_x, batch_y in sample(sess, train_next):
                _, summary = sess.run(
                    [optimizer, summaries], feed_dict={x: batch_x, y: batch_y})
                writer_train.add_summary(summary, epoch)

            sess.run(validation_iter.initializer)
            for batch_x, batch_y in sample(sess, validation_next):
                summary = sess.run(
                    summaries, feed_dict={x: batch_x, y: batch_y})
                writer_val.add_summary(summary, epoch)

            # saver.save(sess, 'train/checkpoints/model.ckpt')
