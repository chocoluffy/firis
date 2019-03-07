
# coding: utf-8

# # DeepLab Demo
# 
# This demo will demostrate the steps to run deeplab semantic segmentation model on sample input images.

# In[74]:


"""
ref: https://github.com/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb
"""

#@title Imports

import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf


# In[88]:


#@title Helper methods

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self):
    """Creates and loads pretrained deeplab model."""
    frozen_graph_pb = './model/19-2-12-v3/frozen_inference_graph.pb'
    with tf.gfile.FastGFile(frozen_graph_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def, name="")
    self.graph = g_in
    self.sess = tf.Session(graph=g_in)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def create_act_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  num_classes = 2
  colormap = np.zeros((num_classes, 3), dtype=int)
  ind = np.arange(num_classes, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_act_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

    
  print("label", label)
  print("colormap", colormap)
    
  return colormap[label]


def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
    
#   print(np.max(seg_map))
#   print(np.min(seg_map))
    
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  print(len(unique_labels))  
    
    
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  plt.show()


LABEL_NAMES = np.asarray([
    'background', 'person'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


# In[89]:


MODEL = DeepLabModel()
print('model loaded successfully!')


# ## Run on sample images
# 
# Select one of sample images (leave `IMAGE_URL` empty) or feed any internet image
# url for inference.
# 
# Note that we are using single scale inference in the demo for fast computation,
# so the results may slightly differ from the visualizations in
# [README](https://github.com/tensorflow/models/blob/master/research/deeplab/README.md),
# which uses multi-scale and left-right flipped inputs.

# In[90]:


#@title Run on sample images {display-mode: "form"}

SAMPLE_IMAGE = 'image1'  # @param ['image1', 'image2', 'image3']
IMAGE_URL = ''  #@param {type:"string"}

_SAMPLE_URL = ('https://1.bp.blogspot.com/-ygj8Bi4JgM4/W6fGAk9hioI/AAAAAAAAPkY/xBvh6RFoA3AOgXKLsgkW2F4q-CPWfNIFgCLcBGAs/s1600/Dilireba%2BDolce%2BGabbana%2B5.jpg')


def run_visualization(url):
  """Inferences DeepLab model and visualizes result."""
  try:
    f = urllib.request.urlopen(url)
    jpeg_str = f.read()
    original_im = Image.open(BytesIO(jpeg_str))
  except IOError:
    print('Cannot retrieve image. Please check url: ' + url)
    return

  print('running deeplab on image %s...' % url)
  resized_im, seg_map = MODEL.run(original_im)

  vis_segmentation(resized_im, seg_map)


image_url = _SAMPLE_URL
run_visualization(image_url)


# In[87]:


image_url = "https://d2gg9evh47fn9z.cloudfront.net/800px_COLOURBOX1790514.jpg"
f = urllib.request.urlopen(image_url)
jpeg_str = f.read()
original_im = Image.open(BytesIO(jpeg_str))

print('running deeplab on image %s...' % image_url)
resized_im, seg_map = MODEL.run(original_im)

np.unique(seg_map)
run_visualization(image_url)

