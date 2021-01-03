from pli_parser import parse
from ga_framework import Encoder
from random import random, randint


def add_nrn(chrom):
    # Add to syn_matrix
    syn_matrix = chrom.get_syn_matrix()
    new_row, new_col = [], []
    for index in range(len(syn_matrix)+1):
        if random() < 0.2:
            new_row.append(1)
        else:
            new_row.append(0)

    output_nrns = chrom.get_output_nrns()
    for index in range(len(syn_matrix)+1):
        if index not in output_nrns and random() < 0.2:
            new_col.append(1)
        else:
            new_col.append(0)
    
    syn_matrix.append(new_row)
    for index in range(len(syn_matrix)):
        # row.append(col[index])
        syn_matrix[index].append(new_col[index])

    chrom.set_syn_matrix(syn_matrix)

    # Add to number of neurons
    chrom.set_num_neurons(chrom.get_num_neurons() + 1)

    # Add initial spikes [0, 5]
    chrom.set_spikes(chrom.get_spikes() + [randint(0, 5)])

    # Add rules [1, 5]
    rules = []
    rule_pool_copy = rule_pool.copy()
    num_rules = randint(1, 5)
    for _ in range(num_rules):
        rule_index = randint(0, len(rule_pool_copy) - 1)
        rules.append(rule_pool_copy[rule_index])
        rule_pool_copy = rule_pool_copy[:rule_index] + rule_pool_copy[rule_index+1:]
    chrom.set_rules(chrom.get_rules() + [rules])

    return chrom


def generate_adv_pli(filename, directory="init_pli/sm_pli/"):
    parsed_pli = parse(filename=directory + "/" + filename + ".pli")
    chrom = Encoder.parsed_pli_to_chrom(parsed_pli=parsed_pli)
    rng = chrom.get_num_neurons() - len(chrom.get_output_nrns())
    for _ in range(rng):
        chrom = add_nrn(chrom)
    Encoder.chrom_to_pli(chrom=chrom, directory="init_pli/adv_pli/", filename=filename, extension="adv")


rule_pool = \
    [{'x': 5, 'y': 0, 'c': 1, 'p': 1, 'delay': 0},
     {'x': 4, 'y': 0, 'c': 1, 'p': 1, 'delay': 0},
     {'x': 3, 'y': 0, 'c': 1, 'p': 1, 'delay': 0},
     {'x': 2, 'y': 0, 'c': 1, 'p': 1, 'delay': 0},
     {'x': 1, 'y': 0, 'c': 1, 'p': 1, 'delay': 0},
     {'x': 5, 'y': 0, 'c': 1, 'p': 0, 'delay': 0},
     {'x': 4, 'y': 0, 'c': 1, 'p': 0, 'delay': 0},
     {'x': 3, 'y': 0, 'c': 1, 'p': 0, 'delay': 0},
     {'x': 2, 'y': 0, 'c': 1, 'p': 0, 'delay': 0},
     {'x': 1, 'y': 0, 'c': 1, 'p': 0, 'delay': 0}]
if __name__ == "__main__":
    generate_adv_pli(filename="add-sm")
    generate_adv_pli(filename="sub-sm")
