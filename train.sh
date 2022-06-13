#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python train.py --batchsize=8 \
                --epochs=10 \
                --lr=0.05 \
                --datadir='new_data_zwh/' \
                --dividedir='new_data_zwh/divide' \
                --logname='logs/log.txt'