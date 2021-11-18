# Dataset path.
DATASET_PATH="dataset_path"

# Log path.
LOG_PATH="log_path"

# Number of output classes.
NUM_CLASSES=2

# Loss function type.
LOSS_FUNCTION_TYPE="CrossEntropyLoss"

# Train batch size.
TRAIN_BATCH_SIZE=64

# Val batch size.
VAL_BATCH_SIZE=64

# Epochs
EPOCHS=12

# Learning rate
LR=0.0001

# Seed number
SEED_NUMBER=11182021

# Model names array.
MODEL_NAMES_ARRAY=("resnet18" "resnet50" "mobilenet_v2" "mobilenet_v3_small" "mobilenet_v3_large")

# ----------------------------------------------------------------------------------------------------------------------
# Fine tune all model parameters.
# ----------------------------------------------------------------------------------------------------------------------

for MODEL_NAME in "${MODEL_NAMES_ARRAY[@]}"
do

  # Current log path.
  CURRENT_LOG_PATH="$LOG_PATH/train_epochs_$EPOCHS/fine_tune_all_parameters/$LOSS_FUNCTION_TYPE/$MODEL_NAME"

  # Train model.
  python train_model.py "$MODEL_NAME" \
  --dataset-path "$DATASET_PATH" \
  --log-path "$CURRENT_LOG_PATH" \
  --num-classes $NUM_CLASSES \
  --loss-function-type "$LOSS_FUNCTION_TYPE" \
  --train-batch-size $TRAIN_BATCH_SIZE \
  --val-batch-size $VAL_BATCH_SIZE \
  --epochs $EPOCHS \
  --learning-rate $LR \
  --seed-number $SEED_NUMBER \
  --enable-training \
  --verbose

done

# ----------------------------------------------------------------------------------------------------------------------
# Optimize only the parameters in the last layer of the neural network.
# ----------------------------------------------------------------------------------------------------------------------

for MODEL_NAME in "${MODEL_NAMES_ARRAY[@]}"
do

  # Current log path.
  CURRENT_LOG_PATH="$LOG_PATH/train_epochs_$EPOCHS/train_last_layer_parameters/$LOSS_FUNCTION_TYPE/$MODEL_NAME"

  # Train model.
  python train_model.py "$MODEL_NAME" \
  --dataset-path "$DATASET_PATH" \
  --log-path "$CURRENT_LOG_PATH" \
  --num-classes $NUM_CLASSES \
  --loss-function-type "$LOSS_FUNCTION_TYPE" \
  --train-batch-size $TRAIN_BATCH_SIZE \
  --val-batch-size $VAL_BATCH_SIZE \
  --epochs $EPOCHS \
  --learning-rate $LR \
  --seed-number $SEED_NUMBER \
  --enable-training \
  --train-last-layer \
  --verbose

done
