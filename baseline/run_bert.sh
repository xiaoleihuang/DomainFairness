#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python bert.py --lr=3e-6 --batch_size=64 --max_len=200 --device cuda
CUDA_VISIBLE_DEVICES=0 python bert_blind.py --lr=3e-6 --batch_size=64 --max_len=200 --device cuda
