#!/bin/bash
python t5_infer.py \
	--model_path "./output/flant5s-orig-e50-lr003-adamw-linear-b64-prodsep-none-trainexample_20240523_09:38:52.27/" \
	--test_path "data/uspto_mit/mit_separated/test_example.csv" \
	--seed 42 \
	--smiles_check "no" \
	--infer_batch 48 \
	--max_new_tokens 132 \
	--max_length 264 \
	--num_return_sequences 5 \
	--num_beams 5 \
	--num_beam_groups 1 \
	--top_k 10 \
	--top_p 0.8 \
	--penalty_alpha 0.6 \
	--diversity_penalty 1.0 \
	--length_penalty -4 \
	--temperature 1.0 \
	--tokenization "none" \
	--infer_mode "search_beam"
