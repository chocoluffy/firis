###  COPY ALL THE CODE INTO A JYPYTER NOTEBOOK  ### 
###  THE JYPYTER NOTEBOOK NEEDS TO BE IN 'tensorflow\models\research\deeplab'  ### 

## Imports

import collections
import os
import io
import sys
import tarfile
import tempfile
import urllib

from IPython import display
from ipywidgets import interact
from ipywidgets import interactive
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2
# import skvideo.io

import tensorflow as tf


# Needed to show segmentation colormap labels
sys.path.append('utils')

import get_dataset_colormap



## Load model in TensorFlow

_FROZEN_GRAPH_NAME = 'frozen_inference_graph'


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""
    
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513

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

model = DeepLabModel()


## Webcam demo

cap = cv2.VideoCapture(0)

# Next line may need adjusting depending on webcam resolution
final = np.zeros((1, 384, 1026, 3))
while True:
    ret, frame = cap.read()
    
    # From cv2 to PIL
    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    
    # Run model
    resized_im, seg_map = model.run(pil_im)
    
    # Adjust color of mask
    seg_image = get_dataset_colormap.label_to_color_image(
        seg_map, get_dataset_colormap.get_pascal_name()).astype(np.uint8)
    
    # Convert PIL image back to cv2 and resize
    frame = np.array(pil_im)
    r = seg_image.shape[1] / frame.shape[1]
    dim = (int(frame.shape[0] * r), seg_image.shape[1])[::-1]
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    
    # Stack horizontally color frame and mask
    color_and_mask = np.hstack((resized, seg_image))

    cv2.imshow('frame', color_and_mask)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

    
###  UNCOMMENT NEXT LINES TO SAVE THE VIDEO  ###
#    output = np.expand_dims(both, axis=0)
#    final = np.append(final, output, 0)
#skvideo.io.vwrite("outputvideo111.mp4", final)