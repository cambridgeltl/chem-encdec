import sys
import evaluate
import numpy as np
from transformers import AutoTokenizer


def evaluate_batch(golds, lists_of_preds, K, spaces=False):
    correct = 0
    total = 0
    for gold, pred_list in zip(golds, lists_of_preds):
        glued_pred_list = []
        for item in pred_list:
            glued_item = "".join(item.strip().split(" "))
            if spaces:
                glued_pred_list.append(glued_item)
            else:
                glued_pred_list.append(item)
        if gold in glued_pred_list[:K]:
            correct += 1
        
        total +=1

    return correct, total


# helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels

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
    result_returned["exact_match"] = result_em["exact_match"]
    result_returned["avg_gold_len"] = avg_gold_len/float(len(decoded_labels))
    result_returned["avg_pred_len"] = avg_pred_len/float(len(decoded_preds))

    return result_returned





