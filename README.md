`ACT/parse-data.ipynb`: parse supervisely dataset.  
    - draw segmentation polygon from opencv detection points. 
    - save masks. 

`ACT/convert_dataset.sh`: convert dataset into TF-Records, similar to processing PASCAL VOC 2012 dataset. 

## Tips

- Run python2 on the scripts, to avoid type conversion bugs!
- `tensorboard --logdir=${PATH_TO_LOG_DIRECTORY}` to visualize network architecture.
- Model backbone: DeepLab V3, reference on "Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs".