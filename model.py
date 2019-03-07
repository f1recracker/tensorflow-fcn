
import tensorflow as tf

def build_fcn_graph(x, variant, num_classes=10):
    variants = ['fcn-32s', 'fcn-16s', 'fcn-8s']
    assert variant in variants, f'Unknown variant! Supported variants: {variants}'

    relu = tf.nn.relu

    with tf.variable_scope(
            'vgg16',
            initializer=tf.initializers.he_normal(),
            regularizer=tf.keras.regularizers.l2(l=1e-3)):

        conv1 = tf.layers.Conv2D(64, 3, activation=relu, padding='same')(x)
        conv1 = tf.layers.Conv2D(64, 3, activation=relu, padding='same')(conv1)
        pool1 = tf.layers.MaxPooling2D(2, 2)(conv1)

        conv2 = tf.layers.Conv2D(128, 3, activation=relu, padding='same')(pool1)
        conv2 = tf.layers.Conv2D(128, 3, activation=relu, padding='same')(conv2)
        pool2 = tf.layers.MaxPooling2D(2, 2)(conv2)

        conv3 = tf.layers.Conv2D(256, 3, activation=relu, padding='same')(pool2)
        conv3 = tf.layers.Conv2D(256, 3, activation=relu, padding='same')(conv3)
        conv3 = tf.layers.Conv2D(256, 3, activation=relu, padding='same')(conv3)
        pool3 = tf.layers.MaxPooling2D(2, 2)(conv3)

        conv4 = tf.layers.Conv2D(512, 3, activation=relu, padding='same')(pool3)
        conv4 = tf.layers.Conv2D(512, 3, activation=relu, padding='same')(conv4)
        conv4 = tf.layers.Conv2D(512, 3, activation=relu, padding='same')(conv4)
        pool4 = tf.layers.MaxPooling2D(2, 2)(conv4)

        conv5 = tf.layers.Conv2D(512, 3, activation=relu, padding='same')(pool4)
        conv5 = tf.layers.Conv2D(512, 3, activation=relu, padding='same')(conv5)
        conv5 = tf.layers.Conv2D(512, 3, activation=relu, padding='same')(conv5)
        pool5 = tf.layers.MaxPooling2D(2, 2)(conv5)

        # Replace FC layers
        conv6 = tf.layers.Conv2D(4096, 7, activation=relu, padding='same')(pool5)
        conv7 = tf.layers.Conv2D(4096, 1, activation=relu, padding='same')(conv6)

    with tf.variable_scope('decoder'):
        if variant == 'fcn-32s':
            score_conv7 = tf.layers.Conv2D(num_classes, 1, padding='same')(conv7)
            logits = tf.layers.Conv2DTranspose(
                num_classes, 64, strides=32, padding='same', name='logits')(score_conv7)

        elif variant == 'fcn-16s':
            score_conv7 = tf.layers.Conv2D(num_classes, 1, padding='same')(conv7)
            score_conv7 = tf.layers.Conv2DTranspose(
                num_classes, 4, strides=2, padding='same')(score_conv7)

            score_pool4 = tf.layers.Conv2D(num_classes, 1, padding='same')(pool4)
            logits = tf.layers.Conv2DTranspose(
                num_classes, 32, strides=16, padding='same', name='logits')(
                    score_conv7 + score_pool4)

        elif variant == 'fcn-8s':
            score_conv7 = tf.layers.Conv2D(num_classes, 1, padding='same')(conv7)
            score_conv7 = tf.layers.Conv2DTranspose(
                num_classes, 8, strides=4, padding='same')(score_conv7)

            score_pool4 = tf.layers.Conv2D(num_classes, 1, padding='same')(pool4)
            score_pool4 = tf.layers.Conv2DTranspose(
                num_classes, 4, strides=2, padding='same')(score_pool4)

            score_pool3 = tf.layers.Conv2D(num_classes, 1, padding='same')(pool3)
            logits = tf.layers.Conv2DTranspose(
                num_classes, 16, strides=8, padding='same', name='logits')(
                    score_conv7 + score_pool4 + score_pool3)
        
        else: raise NotImplementedError(
            f'Unknown variant: {variant} ! Supported variants: {variants}')

    return logits
