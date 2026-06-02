#!/bin/bash
python test.py --weights-file "data/train/outputs/x4/best.pth" \
               --image-file "data/butterfly_GT.bmp" \
               --scale 4
