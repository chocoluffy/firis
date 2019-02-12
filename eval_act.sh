# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
DATASET_DIR="datasets"

# Set up the working directories.
ACT_FOLDER="ACT"
EXP_FOLDER="exp/train_on_trainval_set"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${ACT_FOLDER}/${EXP_FOLDER}/init_models"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ACT_FOLDER}/${EXP_FOLDER}/eval"
DATASET="${WORK_DIR}/${DATASET_DIR}/${ACT_FOLDER}/tfrecord"

MODEL_CHECKPOINT=

# From tensorflow/models/research/ 
python deeplab/eval.py \     
--logtostderr \     
--eval_split="val" \     
--model_variant="mobilenet_v2" \         
--output_stride=16 \     
--decoder_output_stride=4 \     
--eval_crop_size=513 \     
--eval_crop_size=513 \     
--dataset="act" \     
--checkpoint_dir=${MODEL_CHECKPOINT} \     
--eval_logdir=${EVAL_LOGDIR} \     
--dataset_dir=${DATASET} 