#!/bin/bash

python -m nmt.nmt \
  --export=true \
  --export_dir $EXPORT_DIR \
  --out_dir=$MODEL_DIR \
  --ckpt=$CHECKPOINT_PATH
