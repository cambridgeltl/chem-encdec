import os

from datasets import load_dataset
from rdkit import Chem

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import random

from timeit import default_timer as timer
import datetime

import numpy as np
import torch
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          GenerationConfig, set_seed)

from utils_other import generate_batch_custom

from tokenization.tokenization_utils import smi_tokenizer_spaces, simple_spaces
from generation.generation_utils import set_generation_config
from evaluation.evaluation_utils import evaluate_batch

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, required=True, default=None)
    parser.add_argument("--seed",type=int, required=False, default=42)
    parser.add_argument("--smiles_check",type=str, required=False, default="no", choices=['yes', 'no'])

    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=210)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_return_sequences",type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=200, required=False)
    parser.add_argument("--max_length", type=int, default=400, required=True)
    parser.add_argument("--penalty_alpha", type=float, default=0.0, required=False)
    parser.add_argument("--diversity_penalty", type=float, default=1.0, required=False)
    parser.add_argument("--length_penalty", type=float, default=1.0, required=False)
    parser.add_argument("--num_beam_groups", type=int, default=1, required=False)

    parser.add_argument("--infer_batch", type=int, default=1)
    parser.add_argument("--final_n", type=int, default=1)
    parser.add_argument("--load_in_8bit", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--infer_mode", type=str, default="greedy", required=True, choices=['greedy', 'search_beam', 'diverse_beam', 'contrastive', 'sampling_beam', 'nucleus'])

    parser.add_argument("--tokenization", type=str, required=False, default="none", choices=['none','map','shrink', 'map_shrink', 'spaces', 'shrink_spaces', 'simple_spaces'])

    args = parser.parse_args()

    return args


def main():

    # Set arguments
    args = load_arguments()
    print(args)

    # Set the seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    # Set the generation configuration
    generation_config = set_generation_config(args)

    # Load the test dataset now
    dataset = load_dataset("csv", data_files={"test": [args.test_path]})["test"]

    if args.load_in_8bit:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, load_in_8bit=True)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    start_time = timer()
    in_gold_mol_pairs = []
    
    model.eval()

    spaces=False

    for i in range(len(dataset)):

        orig_input, orig_output = dataset[i]["Input"], dataset[i]["Output"]
        if args.tokenization == "none":
            tokenized_input, tokenized_output = orig_input, orig_output
        elif args.tokenization == "spaces":
            tokenized_input = smi_tokenizer_spaces(orig_input)
            tokenized_output = orig_output
            spaces = True
        elif args.tokenization == "simple_spaces":
            tokenized_input = simple_spaces(orig_input)
            tokenized_output = orig_output
            spaces = True
        else:
            print("ERROR: Unsupported preprocessing tokenization!")
            sys.exit(-1)

        in_gold_mol_pair = (tokenized_input, tokenized_output)
        in_gold_mol_pairs.append(in_gold_mol_pair)

    in_gold_mol_batches = generate_batch_custom(in_gold_mol_pairs, args.infer_batch)

    all_predictions = []
    all_golds = []

    for in_gold_mol_batch in in_gold_mol_batches:
        batch_inputs = [x[0] for x in in_gold_mol_batch]
        batch_golds = [x[1] for x in in_gold_mol_batch]

        inputs = tokenizer(batch_inputs, return_tensors="pt", max_length=args.max_length, padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config)
            output_ids = outputs.sequences
            batch_pred_mols = tokenizer.batch_decode(output_ids, skip_special_tokens = True)

        ex_batch_pred_mols = [batch_pred_mols[j:j+args.num_return_sequences] for j in range(0, len(batch_pred_mols), args.num_return_sequences)]

        all_predictions.append(ex_batch_pred_mols)
        all_golds.append(batch_golds)

    # Finally, run evaluation over different K-s

    for K in [1,2,3,5]:
        total = 0
        correct = 0
        for batch_golds, ex_batch_pred_mols in zip(all_golds, all_predictions):
            correct_batch, total_batch = evaluate_batch(batch_golds, ex_batch_pred_mols, K, spaces)
            correct += correct_batch
            total += total_batch

        acc = float(correct)/total
        print("K="+str(K), "|||", "Correct / Total:", correct, "/", total, "["+str(round(acc*100,2))+"%]")

    end_time = timer()
    print("Processing time:" + str(datetime.timedelta(seconds=(end_time-start_time))))

            
if __name__ == "__main__":
    main()
