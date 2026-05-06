#!/bin/bash
python train.py --train-file "data/train/91-image_x3.h5" \
                --eval-file "data/train/Set5_x3.h5" \
                --outputs-dir "data/train/outputs" \
                --scale 3 \
                --lr 1e-4 \
                --batch-size 16 \
                --num-epochs 400 \
                --num-workers 8 \
                --seed 123
