from prettytable import PrettyTable
import torch
import re
import json
from secrets import token_hex
from datetime import datetime

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    total_params = round(total_params/1000000, 1)
    print(f"Total Trainable Params: {total_params}M")
    return total_params


def generate_batch_custom(lst, batch_size):
    """  Yields batch of specified size """
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]

def print_predictions(all_inputs, all_golds, all_predictions, spaces):
    individual_inputs = []
    individual_golds = []
    individual_pred_lists = []
    for batch_input, batch_gold, batch_pred_list in zip(all_inputs, all_golds, all_predictions):
        for individual_input, individual_gold, individual_pred_list in zip(batch_input, batch_gold, batch_pred_list):
            if spaces:
                glued_individual_input = "".join(individual_input.strip().split(" "))
                glued_individual_gold = "".join(individual_gold.strip().split(" "))
                glued_individual_pred_list = []
                for item in individual_pred_list:
                    glued_item = "".join(item.strip().split(" "))
                    glued_individual_pred_list.append(glued_item)
            else:
                glued_individual_input = individual_input
                glued_individual_gold = individual_gold
                glued_individual_pred_list = individual_pred_list

            individual_inputs.append(glued_individual_input)
            individual_golds.append(glued_individual_gold)
            individual_pred_lists.append(glued_individual_pred_list)

    outputs = []
    for x_input, x_gold, x_pred_list in zip(individual_inputs, individual_golds, individual_pred_lists):
        output_dic = {}
        output_dic["id"] = "id_" + str(token_hex(16))
        output_dic["input"] = x_input
        output_dic["gold"] = x_gold
        output_dic["predictions"] = x_pred_list
        outputs.append(output_dic)

    # Print this to the final JSON-formatted file
    json_output = json.dumps(outputs, indent=4)
    timestamp = datetime.now().strftime('%Y%m%d_%H:%M:%S.%f')[:-4]
    json_filename = "joutputs_" + str(timestamp) + ".json"
    with open(json_filename, 'w') as outfile:
        outfile.write(json_output
