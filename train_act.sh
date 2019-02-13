cd ..
# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
DATASET_DIR="datasets"

# Set up the working directories.
ACT_FOLDER="ACT"
EXP_FOLDER="exp/train_on_trainval_set"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${ACT_FOLDER}/${EXP_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${ACT_FOLDER}/${EXP_FOLDER}/train-19-2-12-v3"
DATASET="${WORK_DIR}/${DATASET_DIR}/${ACT_FOLDER}/tfrecord"

mkdir -p "${WORK_DIR}/${DATASET_DIR}/${ACT_FOLDER}/exp"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${INIT_FOLDER}"

# Copy locally the trained checkpoint as the initial checkpoint.
# TF_INIT_ROOT="https://storage.googleapis.com/mobilenet_v2/checkpoints"
# CKPT_NAME="mobilenet_v2"
# TF_INIT_CKPT="${CKPT_NAME}_1.4_224.tgz"
# cd "${INIT_FOLDER}"
# wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
# tar -xf "${TF_INIT_CKPT}"
# cd "${CURRENT_DIR}"

# Copy locally the trained checkpoint as the initial checkpoint.
TF_INIT_ROOT="http://download.tensorflow.org/models"
CKPT_NAME="deeplabv3_mnv2_pascal_train_aug"
TF_INIT_CKPT="${CKPT_NAME}_2018_01_29.tar.gz"
cd "${INIT_FOLDER}"
wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"

NUM_ITERATIONS=30000
python "${WORK_DIR}"/train.py \
  --logtostderr \
  --dataset="act" \
  --train_split="train" \
  --model_variant="mobilenet_v2" \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size=513 \
  --train_crop_size=513 \
  --train_batch_size=4 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=True \
  --initialize_last_layer=False \
  --last_layers_contain_logits_only=False \
  --tf_initial_checkpoint="${INIT_FOLDER}/${CKPT_NAME}/model.ckpt-30000" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${DATASET}"

# mobilenet_v2 should not use atrous_rates.
# run eval\vis: https://beerensahu.wordpress.com/2018/04/17/guide-for-using-deeplab-in-tensorflow/

# key:
# If you have your own dataset but want to reuse the pre-trained feature encoder (also called backbone)
#   --initialize_last_layer=False \
#   --last_layers_contain_logits_only=False \