`train_act.sh`:  
    - `TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ACT_FOLDER}/${EXP_FOLDER}/train` 19-2-11 更新
    - `TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ACT_FOLDER}/${EXP_FOLDER}/train-19-2-12"` 19-2-12 更新


`ACT/parse-data.ipynb`: parse supervisely dataset.  
    - draw segmentation polygon from opencv detection points.  
    - save masks. 

`ACT/convert_dataset.sh`: convert dataset into TF-Records, similar to processing PASCAL VOC 2012 dataset. 

## Tips

- Run python2 on the scripts, to avoid type conversion bugs!
- `tensorboard --logdir=${PATH_TO_LOG_DIRECTORY} --host=127.0.0.1` to visualize network architecture.
- Model backbone: DeepLab V3, reference on "Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs".