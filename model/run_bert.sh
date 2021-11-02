#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python domain_bert.py --lr=3e-6 --batch_size=64 --max_len=200 --device cuda
