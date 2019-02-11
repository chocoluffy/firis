WORK_DIR="."

# Build TFRecords of the dataset.
OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"
echo "Converting Firis dataset..."
python ./build_new_dataset.py