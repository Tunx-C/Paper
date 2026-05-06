#!/bin/bash
python test.py --weights-file "data/train/outputs/x3/best.pth" \
               --image-file "data/butterfly_GT.bmp" \
               --scale 3
