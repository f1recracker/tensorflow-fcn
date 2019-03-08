
''' KITTI Road dataset loader and utils '''

import functools
import os
import random
import re

import tensorflow as tf

def _listdir(path, filter_re=re.compile(r'.*')):
    ''' Enumerates full paths of files in a directory matching a filter '''
    return [os.path.join(path, f) for f in os.listdir(path) if filter_re.match(f)]

def _list_files(root, labels=True):
    ''' '''
    file_re = re.compile(r'(uu|um|umm)(?:_road)?_(\d+)\.png')
    images = sorted(_listdir(os.path.join(root, 'image_2'), file_re))

    if not labels or not os.path.exists(os.path.join(root, 'gt_image_2')):
        raise NotImplementedError('Support for unlabelled dataset pending')

    labels = sorted(_listdir(os.path.join(root, 'gt_image_2'), file_re))
    dataset = list(zip(images, labels))

    def get_id(filepath):
        match = file_re.match(os.path.basename(filepath))
        return (match.group(1), match.group(2))

    for x, y in dataset:
        assert get_id(x) == get_id(y)
    return dataset


def get_split(train_root, splits, seed=1):
    files = _list_files(train_root)
    assert sum(splits) == len(files)

    random.seed(seed)
    random.shuffle(files)

    results = tuple()
    for size in splits:
        results = results + (files[:size],)
        files = files[size:]
    return results


def build_dataset(files, size=(384, 1280), normalize=True,
                  normalize_mean=[0.0, 0.0, 0.0],
                  normalize_std=[1.0, 1.0, 1.0]):
    ''' Converts a list of files into a tf.data.Dataset '''
    images, labels = zip(*files)

    to_uint8 = lambda x: tf.cast(x, tf.uint8)

    def to_tensor(image_file, mode):
        assert mode in ['image', 'label'], f'Unsupported mode {mode}'
        image = tf.read_file(image_file)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize_images(image, size)
        if mode == 'image':
            image = image / 255.0
        if mode == 'label':
            image = to_uint8(image)
        return image

    images_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(images))
    labels_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(labels))

    images_dataset = images_dataset.map(functools.partial(to_tensor, mode='image'))
    labels_dataset = labels_dataset.map(functools.partial(to_tensor, mode='label'))

    def normalize_fn(image):
        return (image - tf.constant(normalize_mean)) / tf.constant(normalize_std)

    if normalize:
        images_dataset = images_dataset.map(normalize_fn)

    def color_to_label(label_image):
        r, _, b = tf.split(to_uint8(label_image), 3, axis=2)
        ignore_mask = to_uint8(tf.equal(r, 0))
        road_mask = to_uint8(tf.math.logical_and(tf.equal(r, 255), tf.equal(b, 255)))
        other_mask = to_uint8(tf.math.logical_and(tf.equal(r, 255), tf.equal(b, 0)))
        return tf.squeeze(0 * road_mask + 1 * other_mask + 255 * ignore_mask)

    labels_dataset = labels_dataset.map(color_to_label)
    dataset = tf.data.Dataset.zip((images_dataset, labels_dataset))
    return dataset


def build_aug_pipeline(dataset):
    raise NotImplementedError('Augmentation pipeline not implemented')


if __name__ == "__main__":

    import numpy as np
    import cv2

    train, _ = get_split('data_road/training', (289, 0))
    train = train[:20]

    train_dataset = build_dataset(train, size=(384, 1280))
    # train_dataset = build_aug_pipeline(train_dataset)

    train_dataset = train_dataset.shuffle(64).batch(1).prefetch(1 * 2)

    train_next = train_dataset.make_initializable_iterator()

    def sample(sess, iterator):
        sess.run([iterator.initializer])
        while True:
            try:
                yield sess.run(iterator.get_next())
            except tf.errors.OutOfRangeError:
                break

    with tf.Session() as sess:

        for mb_x, mb_y in sample(sess, train_next):

            mb_x = np.array(mb_x[0])
            mb_y = np.array(mb_y[0])

            print(np.unique(mb_y))
            print(mb_y.shape)

            cv2.imshow('mb_x', mb_x)
            cv2.imshow('mb_y', mb_y)
            cv2.waitKey(0)
