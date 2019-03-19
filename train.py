# pylint: disable=invalid-name,redefined-outer-name

import tensorflow as tf

from dataset import build_dataset, augmentation_pipeline
from model import fcn_model  

if __name__ == '__main__':

    height, width = 384, 1280
    batch_size = 4

    # Create dataset
    train_dataset = build_dataset('data_road/training', size=(height, width))
    train_dataset, validation_dataset = train_dataset.take(200), train_dataset.skip(200)

    train_dataset = train_dataset.map(augmentation_pipeline)
    train_dataset = train_dataset.shuffle(batch_size * 8).batch(batch_size)
    train_dataset = train_dataset.apply(tf.data.experimental.prefetch_to_device('/gpu:0'))

    validation_dataset = validation_dataset.batch(batch_size)
    validation_dataset = validation_dataset.apply(tf.data.experimental.prefetch_to_device('/gpu:0'))

    # TODO remove in tf2 migration
    train_iter = train_dataset.make_initializable_iterator()
    validation_iter = validation_dataset.make_initializable_iterator()

    train_next = train_iter.get_next()
    validation_next = validation_iter.get_next()


    # Define model graph
    num_classes = 2

    x = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='x')
    y = tf.placeholder(tf.uint8, shape=[None, None, None], name='y')

    logits = fcn_model(x, 'fcn-8s', num_classes=num_classes)
    y_pred = tf.argmax(logits, axis=3, name='y_pred')

    mask = tf.not_equal(y, 255)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        tf.boolean_mask(tf.one_hot(y, num_classes), mask),
        tf.boolean_mask(logits, mask)))
    loss = tf.add(loss, tf.losses.get_regularization_loss(), name='total_loss')

    global_step = tf.Variable(1, trainable=False)
    learning_rate = tf.train.exponential_decay(
        1e-4, global_step, 289 / batch_size, 0.993, staircase=True)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(loss, global_step=global_step)

    lr_summary = tf.summary.scalar('learning_rate', learning_rate)

    # Metrics
    c_matrix = tf.math.confusion_matrix(
        tf.boolean_mask(y, mask),
        tf.boolean_mask(y_pred, mask),
        num_classes=num_classes)

    tp = tf.diag_part(c_matrix)
    fp = tf.reduce_sum(c_matrix, axis=0) - tp
    fn = tf.reduce_sum(c_matrix, axis=1) - tp

    pixel_acc = tf.reduce_sum(tp) / tf.reduce_sum(c_matrix)
    mean_iou = tf.reduce_mean(tp / (tp + fp + fn))

    summaries = tf.summary.merge([
        tf.summary.scalar('loss', loss),
        tf.summary.scalar('pixel_accuracy', pixel_acc),
        tf.summary.scalar('mean_iou', mean_iou)
    ])


    # Tensorboard and checkpointing
    saver = tf.train.Saver()

    import datetime
    timenow = datetime.datetime.now()
    writer_train = tf.summary.FileWriter(f'train/tensorboard/{timenow}/train',
                                         tf.get_default_graph())
    writer_val = tf.summary.FileWriter(f'train/tensorboard/{timenow}/validation')


    # TODO remove in tf2 migration
    def sample(sess, next_op):
        ''' Returns a mininatch from an iterator '''
        while True:
            try:
                yield sess.run(next_op)
            except tf.errors.OutOfRangeError:
                break


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        max_epochs = 100

        for epoch in range(max_epochs):

            writer_train.add_summary(sess.run(lr_summary), epoch)

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

            saver.save(sess, 'train/checkpoints/model.ckpt')
