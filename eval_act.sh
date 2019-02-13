cd ..
# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
DATASET_DIR="datasets"

# Set up the working directories.
ACT_FOLDER="ACT"
EXP_FOLDER="exp/train_on_trainval_set"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${ACT_FOLDER}/${EXP_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ACT_FOLDER}/${EXP_FOLDER}/train-19-2-12-v3/model.ckpt-30"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ACT_FOLDER}/${EXP_FOLDER}/eval"
DATASET="${WORK_DIR}/${DATASET_DIR}/${ACT_FOLDER}/tfrecord"

mkdir -p "${EVAL_LOGDIR}"

# From tensorflow/models/research/ 
python "${WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="mobilenet_v2" \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --eval_crop_size=3761 \
  --eval_crop_size=3761 \
  --dataset="act" \
  --checkpoint_dir=${TRAIN_LOGDIR} \
  --eval_logdir=${EVAL_LOGDIR} \
  --dataset_dir=${DATASET} \
  --max_number_of_evaluations=1

# k * stride + 1, k = 235, thus vis_crop_size = 3761