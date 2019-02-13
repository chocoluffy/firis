cd ..
# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
DATASET_DIR="datasets"

# Set up the working directories.
ACT_FOLDER="ACT"
EXP_FOLDER="exp/train_on_trainval_set"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${ACT_FOLDER}/${EXP_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ACT_FOLDER}/${EXP_FOLDER}/train/model.ckpt-20000"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ACT_FOLDER}/${EXP_FOLDER}/vis"
DATASET="${WORK_DIR}/${DATASET_DIR}/${ACT_FOLDER}/tfrecord"

mkdir -p "${WORK_DIR}/${DATASET_DIR}/${ACT_FOLDER}/exp"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${INIT_FOLDER}"

# Copy locally the trained checkpoint as the initial checkpoint.
# TF_INIT_ROOT="http://download.tensorflow.org/models"
# CKPT_NAME="deeplabv3_mnv2_pascal_train_aug"
# TF_INIT_CKPT="${CKPT_NAME}_2018_01_29.tar.gz"
# cd "${INIT_FOLDER}"
# wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
# tar -xf "${TF_INIT_CKPT}"
# cd "${CURRENT_DIR}"

# From tensorflow/models/research/ 
python "${WORK_DIR}"/vis.py \
    --logtostderr \
    --vis_split="val" \
    --model_variant="mobilenet_v2" \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --vis_crop_size=3761 \
    --vis_crop_size=3761 \
    --dataset="act" \
    --checkpoint_dir="${TRAIN_LOGDIR}" \
    --vis_logdir="${VIS_LOGDIR}" \
    --dataset_dir="${DATASET}"

# k * stride + 1, k = 235, thus vis_crop_size = 3761