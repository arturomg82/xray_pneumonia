# Detection of Pneumonia in x-ray images.

This code shows the concept of transfer learning, using a variety of pre-trained CNNs (e.g., Resnet-18, Resnet-50, 
Mobilenet-V2, Mobilenet-V3) in the ImageNet dataset. The target task is the classification 
of normal vs. abnormal x-ray images, where an abnormal image corresponds to a case of pneumonia.

## 1. Dataset.

The dataset used to train and evaluate the CNN models implemented in this code is publicly available in:
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

## 2. Dependencies.

This code has been implemented in Python 3.8 and PyTorch 1.9.1 (CUDA 11.1).

## 3. Model training and evaluation.

To train and evaluate various CNN models on the x-ray dataset, modify the entry values (e.g., `DATASET_PATH`, `LOG_PATH`, etc.) in the bash script `train_model.sh`.

``` 

# Dataset path.
DATASET_PATH="dataset_path"

# Log path to save the results.
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
EPOCHS=10

# Learning rate
LR=0.0001

# Seed number
SEED_NUMBER=18112021

# Model names array.
MODEL_NAMES_ARRAY=("resnet18" "resnet50" "mobilenet_v2" "mobilenet_v3_small" "mobilenet_v3_large")

# ----------------------------------------------------------------------------------------------------------------------
# Fine tune all model parameters.
# ----------------------------------------------------------------------------------------------------------------------

for MODEL_NAME in "${MODEL_NAMES_ARRAY[@]}"
do

  # Current log path.
  CURRENT_LOG_PATH="$LOG_PATH/train_epochs_$EPOCHS/fine_tune_all_parameters/$MODEL_NAME"

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
  CURRENT_LOG_PATH="$LOG_PATH/train_epochs_$EPOCHS/train_last_layer_parameters/$MODEL_NAME"

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

```

Run the bash script:

``` bash train_model.sh ```

## 4. Experimental results.

The model performance is available as Tensorboard and CSV data, which are stored in the specified `LOG_PATH`.