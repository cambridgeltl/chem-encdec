from prettytable import PrettyTable
import torch
import re

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
