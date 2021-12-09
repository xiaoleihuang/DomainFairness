#!/bin/bash
CUDA_VISIBLE_DEVICES="" python rnn.py --lr=3e-5 --batch_size=64 --max_len=200 --device cpu
CUDA_VISIBLE_DEVICES="" python rnn_blind.py --lr=3e-5 --batch_size=64 --max_len=200 --device cpu