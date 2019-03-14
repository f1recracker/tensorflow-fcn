
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
        if mode == 'image':
            image = tf.image.resize_images(image, size)
            image = image / 255.0
        if mode == 'label':
            image = tf.image.resize_images(
                image, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
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


def augmentation_pipeline(image, label, batched_input=False,
                          hflip_prob=0.5, max_crop_ratio=0.8,
                          brightness_delta=0.1, hue_delta=0.07,
                          contrast_ratio=1.25, saturation_ratio=1.25):
    if not batched_input:
        image = tf.expand_dims(image, 0)
        label = tf.expand_dims(tf.expand_dims(label, 0), 3)

    batch_size, height, width, _ = [d.value for d in image.shape]

    # Horizontal flip
    flip = tf.random_uniform((batch_size,)) < hflip_prob
    image = tf.where(flip, tf.image.flip_left_right(image), image)
    label = tf.where(flip, tf.image.flip_left_right(label), label)

    # Crop and resize image
    cut_ratio = (1 - max_crop_ratio) / 2
    boxes_1 = tf.random.uniform([batch_size, 2], 0, cut_ratio)
    boxes_2 = tf.random.uniform([batch_size, 2], 1 - cut_ratio, 1)
    boxes = tf.concat([boxes_1, boxes_2], 1)
    image = tf.image.crop_and_resize(
        image, boxes, tf.range(batch_size), tf.constant([height, width]))
    label = tf.image.crop_and_resize(
        label, boxes, tf.range(batch_size), tf.constant([height, width]), method="nearest")

    # Add random brightness, hue, contrast, and saturation
    image = tf.image.random_brightness(image, brightness_delta)
    image = tf.image.random_hue(image, hue_delta)
    image = tf.image.random_contrast(image, 1 / contrast_ratio, contrast_ratio)
    image = tf.image.random_saturation(image, 1 / saturation_ratio, saturation_ratio)

    if not batched_input:
        image = tf.squeeze(image, [0])
        label = tf.squeeze(label, [0, 3])

    return (image, label)


if __name__ == "__main__":

    import numpy as np
    import cv2

    train, _ = get_split('data_road/training', (289, 0))
    train = train[10:12]

    train_dataset = build_dataset(train, size=(384, 1280))
    train_dataset = train_dataset.map(augmentation_pipeline)

    train_dataset = train_dataset.repeat(20).batch(1).prefetch(1 * 2)

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

            cv2.imshow('mb_x', cv2.cvtColor(mb_x, cv2.COLOR_BGR2RGB))
            # cv2.imshow('mb_y', mb_y)

            cv2.imshow('mb_y_0', (mb_y == 0).astype(float))
            cv2.imshow('mb_y_1', (mb_y == 1).astype(float))
            cv2.imshow('mb_y_255', (mb_y == 255).astype(float))

            cv2.waitKey(0)
