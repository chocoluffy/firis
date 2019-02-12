`parse-data.ipynb`: parse supervisely dataset.
    - draw segmentation polygon from opencv detection points. 
    - save masks. 

`convert_dataset.sh`: convert dataset into TF-Records. 

- run python2 on the script! to avoid type conversion bugs!
- `tensorboard --logdir=${PATH_TO_LOG_DIRECTORY}` to visualize network architecture.