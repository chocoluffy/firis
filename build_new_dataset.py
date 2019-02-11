"""
ref: https://github.com/tensorflow/models/blob/master/research/deeplab/datasets/build_ade20k_data.py

Converts image\masks data to TFRecord file format.
"""

import math
import os
import random
import sys
import build_data
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'train_image_folder',
    './dataset/source_image/train',
    'Folder containing training images')
tf.app.flags.DEFINE_string(
    'train_image_label_folder',
    './dataset/segmentation_image/train',
    'Folder containing annotations for trainng images')

tf.app.flags.DEFINE_string(
    'val_image_folder',
    './dataset/source_image/val',
    'Folder containing validation images')

tf.app.flags.DEFINE_string(
    'val_image_label_folder',
    './dataset/segmentation_image/val',
    'Folder containing annotations for validation')

tf.app.flags.DEFINE_string(
    'output_dir', './tfrecord',
    'Path to save converted tfrecord of Tensorflow example')

_NUM_SHARDS = 4


def _convert_dataset(dataset_split, dataset_dir, dataset_label_dir):
  """Converts Firis dataset into into tfrecord format.
  Args:
    dataset_split: Dataset split (e.g., train, val).
    dataset_dir: Dir in which the dataset locates.
    dataset_label_dir: Dir in which the annotations locates.
  Raises:
    RuntimeError: If loaded image and label have different shape.
  """

  img_names = tf.gfile.Glob(os.path.join(dataset_dir, '*.jpg'))
  random.shuffle(img_names)
  seg_names = []
  for f in img_names:
    # get the filename without the extension
    basename = os.path.basename(f).split('.')[0]
    # cover its corresponding *_seg.png
    seg = os.path.join(dataset_label_dir, basename+'.png')
    seg_names.append(seg)

  num_images = len(img_names)
  num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

  image_reader = build_data.ImageReader('jpeg', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)

  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(
        FLAGS.output_dir,
        '%s-%05d-of-%05d.tfrecord' % (dataset_split, shard_id, _NUM_SHARDS))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, num_images, shard_id))
        sys.stdout.flush()
        # Read the image.
        image_filename = img_names[i]
        image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
        height, width = image_reader.read_image_dims(image_data)
        # Read the semantic segmentation annotation.
        seg_filename = seg_names[i]
        seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()
        seg_height, seg_width = label_reader.read_image_dims(seg_data)
        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')
        # Convert to tf example.
        example = build_data.image_seg_to_tfexample(
            image_data, img_names[i], height, width, seg_data)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def main(unused_argv):
  tf.gfile.MakeDirs(FLAGS.output_dir)
  _convert_dataset(
      'train', FLAGS.train_image_folder, FLAGS.train_image_label_folder)
  _convert_dataset('val', FLAGS.val_image_folder, FLAGS.val_image_label_folder)


if __name__ == '__main__':
  tf.app.run()