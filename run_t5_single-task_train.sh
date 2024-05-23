#!/bin/bash
python t5_single-task_train.py \
	--model_path "google/flan-t5-small" \
	--train_path "data/uspto_mit/mit_separated/train_example.csv" \
        --val_path "data/uspto_mit/mit_separated/val_example.csv" \
	--max_new_tokens 96 \
	--max_length 192 \
        --max_source_len 96 \
	--max_target_len 96 \
	--seed 42 \
	--freeze "none" \
	--tokenization "none" \

