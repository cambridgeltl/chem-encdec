# To silence TensorFlow
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import os
import random
import sys
from datetime import datetime
from random import randrange

import evaluate
import nltk
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from nltk.tokenize import sent_tokenize
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, T5ForConditionalGeneration,
                          T5Tokenizer, set_seed, GenerationConfig)

import argparse

from tokenization.tokenization_utils import smi_tokenizer_spaces

device = "cuda" if torch.cuda.is_available() else "cpu"
template = "{sentence}"
portion="train"

def load_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)

    parser.add_argument("--train_path", type=str, required=True, default=None)
    parser.add_argument("--val_path", type=str, required=True, default=None)

    parser.add_argument("--seed",type=int, required=False, default=42)

    parser.add_argument("--freeze",type=str, required=False, default="none", choices=['none', 'encoder'])

    parser.add_argument("--max_new_tokens", type=int, default=96, required=False)
    parser.add_argument("--max_length", type=int, default=192, required=False)
    parser.add_argument("--max_source_len", type=int, default=96, required=False)
    parser.add_argument("--max_target_len", type=int, default=192, required=False)

    parser.add_argument("--tokenization", type=str, required=False, default="none", choices=['none','map','shrink', 'map_shrink', 'spaces', 'shrink_spaces'])

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

    # Set the model and its tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(args.model_path,use_cache=True)

    # Now also set the generation config (non-mandatory)
    generation_config = GenerationConfig.from_pretrained(
        args.model_path,
        do_sample=False,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )

    # Freeze some parameters or not: this code supports only freezing of the encoder at the moment
    modules_to_freeze = []
    if args.freeze == "encoder":
        print("Freezing all encoder layers...")
        modules_to_freeze = [model.encoder.block[i] for i in range(len(model.encoder.block))]
    for module in modules_to_freeze:
        for param in module.parameters():
            param.requires_grad = False 

    # Load the dataset (via HuggingFace functions)
    max_source_len, max_target_len = args.max_source_len, args.max_target_len
    dataset = load_dataset("csv", data_files={"train": [args.train_path], "val": [args.val_path]})

    # Prepare the dataset for training and evaluation: key functionality
    def preprocess_function(sample, padding="max_length"):
        inputs = []
        outputs = []
        
        for in_mol, out_mol in zip(sample["Input"], sample["Output"]):
            
            # Set inputs
            if args.tokenization == "spaces":
                in_mol = smi_tokenizer_spaces(in_mol)
            else:
                in_mol = in_mol

            final_input = template.replace("{sentence}", in_mol)

            # Set outputs
            if args.tokenization == "spaces":
                out_mol = smi_tokenizer_spaces(out_mol)
            else:
                out_mol = out_mol

            final_output = out_mol

            inputs.append(final_input)
            outputs.append(final_output)
        
        model_inputs = tokenizer(inputs, max_length=max_source_len, padding=padding, truncation=True)
        print("Total items:", len(outputs))
        labels = tokenizer(text_target=outputs, max_length=max_target_len, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["Input", "Output"])
    print(f"Keys of tokenized dataset: {list(tokenized_dataset[portion].features)}")
    print(tokenized_dataset["val"][0])
 
    em = evaluate.load("exact_match")

    def compute_eval_metrics_training(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds > 0, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        avg_gold_len = 0
        avg_pred_len = 0

        for item in decoded_labels:
            avg_gold_len += len(item)

        for item in decoded_preds:
            avg_pred_len += len(item)

        decoded_labels_f = [[item] for item in decoded_labels]

        result_em = em.compute(predictions=decoded_preds, references=decoded_labels)
        result_returned = {}
        result_returned["exact_match"] = round(100*float(result_em["exact_match"]),2)
        result_returned["avg_gold_len"] = avg_gold_len/float(len(decoded_labels))
        result_returned["avg_pred_len"] = avg_pred_len/float(len(decoded_preds))

        return result_returned

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
    )

    # Define training args
    timestamp = datetime.now().strftime('%Y%m%d_%H:%M:%S.%f')[:-4]
    experiment_name = "flant5s-orig-e50-lr003-adamw-wlinear-b64-prodsep-none"
    output_folder = f"./output/{experiment_name}_{timestamp}"
    print(output_folder)

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{output_folder}",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        generation_config=generation_config,
        generation_max_length=args.max_target_len,
        predict_with_generate=True,
        fp16=False, # Overflows with fp16
        learning_rate=0.003,
        num_train_epochs=50,
        #max_steps=100000,
        # logging & evaluation strategies
        logging_dir=f"{output_folder}/logs",
        logging_strategy="steps",
        logging_steps=1000,
        gradient_checkpointing=False,
        evaluation_strategy="steps",
        #optim="adafactor",
        eval_steps=10000,
        save_strategy="steps",
        save_steps=10000,
        warmup_steps=5000,
        weight_decay=0.01,
        #lr_scheduler_type="inverse_sqrt",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        compute_metrics=compute_eval_metrics_training,
    )
    #... and train
    trainer.train()
    trainer.save_model(output_folder)


if __name__ == "__main__":
    main()
