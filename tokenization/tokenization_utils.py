import re
import csv

pattern_smi = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex_smi = re.compile(pattern_smi)


def smi_tokenizer(smi):
    tokens = [token for token in regex_smi.findall(smi)]
    assert smi == ''.join(tokens)
    return tokens

def smi_tokenizer_spaces(smi):
    tokens = [token for token in regex_smi.findall(smi)]
    return " ".join(tokens)

def simple_spaces(smi):
    tokens = [token for token in smi]
    return " ".join(tokens)

