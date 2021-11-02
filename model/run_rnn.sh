#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python domain_rnn.py --lr=3e-5 --batch_size=64 --max_len=200 --device cuda
