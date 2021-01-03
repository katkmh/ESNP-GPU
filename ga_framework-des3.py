from simulator import simulate
from pli_parser import parse
from fitness_cases import fit_cases
from initial_population import init_population
from random import random, randint, shuffle
from operator import methodcaller, itemgetter
from time import time
from datetime import datetime
from graphviz import Digraph
from getopt import getopt, GetoptError
import sys
import os
import math


class Chromosome:
    def __init__(self, num_nrns):
        self.num_nrns = num_nrns
        self.precision = 0
        self.fitness = 0
        self.spikes = [0] * num_nrns
        self.rules = [list() for _ in range(num_nrns)]
        self.input_nrns = []
        self.output_nrns = []
        self.syn_matrix = [[0] * num_nrns for _ in range(num_nrns)]

        # These will be generated when Chromosome.set_syn_matrix() is called
        self.active_nrns = []
        self.connected_nrns = []
        self.connected_to_input_nrns = []
        self.connected_to_output_nrns = []

    def set_num_nrns(self, num_nrns):
        self.num_nrns = num_nrns
        
    def set_precision(self, precision):
        self.precision = precision

    def set_fitness(self, fitness):
        self.fitness = fitness

    def set_spikes(self, spikes):
        self.spikes = spikes
        
    def set_rules(self, rules):
        self.rules = rules
        
    def set_input_nrns(self, input_nrns):
        self.input_nrns = input_nrns
        
    def set_output_nrns(self, output_nrns):
        self.output_nrns = output_nrns
        
    def set_syn_matrix(self, syn_matrix):
        self.syn_matrix = syn_matrix
        self.connected_to_input_nrns = self.generate_connected_to_input_nrns()
        self.connected_to_output_nrns = self.generate_connected_to_output_nrns()
        self.active_nrns = self.generate_active_nrns()
        self.connected_nrns = self.generate_connected_nrns()

    def set_active_nrns(self, active_nrns):
        self.active_nrns = active_nrns

    def set_connected_nrns(self, connected_nrns):
        self.connected_nrns = connected_nrns

    def set_connected_to_input_nrns(self, connected_to_input_nrns):
        self.connected_to_input_nrns = connected_to_input_nrns
        
    def set_connected_to_output_nrns(self, connected_to_output_nrns):
        self.connected_to_output_nrns = connected_to_output_nrns

    def get_num_nrns(self):
        return self.num_nrns

    def get_precision(self):
        return self.precision

    def get_fitness(self):
        return self.fitness
    
    def get_spikes(self):
        return self.spikes.copy()
    
    def get_rules(self):
        new_rules = []
        for row in self.rules:
            new_rules.append(row.copy())
            
        return new_rules

    def get_input_nrns(self):
        return self.input_nrns.copy()

    def get_output_nrns(self):
        return self.output_nrns.copy()

    def get_syn_matrix(self):
        new_matrix = []
        for row in self.syn_matrix:
            new_matrix.append(row.copy())

        return new_matrix

    def get_active_nrns(self):
        return self.active_nrns.copy()

    def get_connected_nrns(self):
        return self.connected_nrns.copy()

    def get_connected_to_input_nrns(self):
        return self.connected_to_input_nrns.copy()

    def get_connected_to_output_nrns(self):
        return self.connected_to_output_nrns.copy()

    def get_copy(self):
        new_chrom = Chromosome(self.get_num_nrns())
        new_chrom.set_spikes(self.get_spikes())
        new_chrom.set_rules(self.get_rules())
        new_chrom.set_input_nrns(self.get_input_nrns())
        new_chrom.set_output_nrns(self.get_output_nrns())
        new_chrom.set_syn_matrix(self.get_syn_matrix())
        new_chrom.set_precision(self.get_precision())
        new_chrom.set_fitness(self.get_fitness())
    
        return new_chrom

    def generate_connected_to_input_nrns(self):
        syn_matrix = self.get_syn_matrix()
        input_nrns = self.get_input_nrns()
        num_nrns = self.get_num_nrns()

        a_nrns = input_nrns.copy()
        connected_to_input_nrns = input_nrns.copy()
        b_nrns = [x for x in range(num_nrns) if x not in connected_to_input_nrns]

        while True:
            do_break = True
            new_a = []
            for a in a_nrns:
                for b in b_nrns:
                    if syn_matrix[a][b]:
                        do_break = False
                        connected_to_input_nrns.append(b)
                        new_a.append(b)
            if do_break:
                break                
            a_nrns = list(set(new_a))
            b_nrns = [x for x in range(num_nrns) if x not in connected_to_input_nrns]
            
        return list(set(connected_to_input_nrns))
    
    def generate_connected_to_output_nrns(self):
        syn_matrix = self.get_syn_matrix()
        output_nrns = self.get_output_nrns()
        num_nrns = self.get_num_nrns()

        a_nrns = output_nrns.copy()
        connected_to_output_nrns = output_nrns.copy()
        b_nrns = [x for x in range(num_nrns) if x not in connected_to_output_nrns]

        while True:
            do_break = True
            new_a = []
            for a in a_nrns:
                for b in b_nrns:
                    if syn_matrix[b][a]:
                        do_break = False
                        connected_to_output_nrns.append(b)
                        new_a.append(b)
            if do_break:
                break                
            a_nrns = list(set(new_a))
            b_nrns = [x for x in range(num_nrns) if x not in connected_to_output_nrns]
            
        return list(set(connected_to_output_nrns))

    def generate_active_nrns(self):
        # Active neurons = reachable from at least one input neuron AND to the output neuron (env)
        # or if it can spike (meaning it contains at least one spike in the beginning and a rule cna be used)
        num_nrns = self.get_num_nrns()
        active_nrn_indices = self.input_nrns + self.output_nrns
        syn_matrix = self.get_syn_matrix()
        connected_to_input_nrns = self.get_connected_to_input_nrns()
        connected_to_output_nrns = self.get_connected_to_output_nrns()

        a_nrns = [x for x in range(num_nrns) if x not in active_nrn_indices]
        for a in a_nrns:
            if a in connected_to_output_nrns:
                if a in connected_to_input_nrns:
                    active_nrn_indices.append(a)
                elif self.spikes[a] > 0:
                    is_will_spike = False

                    for rule in self.rules[a]:
                        if rule['y'] == 0:
                            if self.spikes[a] - rule['x'] == 0 and self.spikes[a] >= rule['c']:
                                is_will_spike = True
                                break
                        else:
                            if (self.spikes[a] - rule['x']) % rule['y'] == 0 and self.spikes[a] >= rule['c']:
                                is_will_spike = True
                                break

                    if is_will_spike:
                        active_nrn_indices.append(a)

        # Children of active neurons are also active
        a_nrns = active_nrn_indices.copy()
        b_nrns = [x for x in range(num_nrns) if x not in active_nrn_indices]

        while True:
            do_break = True
            new_a = []
            for a in a_nrns:
                for b in b_nrns:
                    if syn_matrix[a][b]:
                        do_break = False
                        active_nrn_indices.append(b)
                        new_a.append(b)
            if do_break:
                break
            a_nrns = list(set(new_a))
            b_nrns = [x for x in range(num_nrns) if x not in active_nrn_indices]

        return list(set(active_nrn_indices))

    def generate_connected_nrns(self):
        # Connected neurons = reachable from an input neuron OR to an output neuron
        num_nrns = self.get_num_nrns()
        connected_nrn_indices = self.input_nrns + self.output_nrns
        connected_to_input_nrns = self.get_connected_to_input_nrns()
        connected_to_output_nrns = self.get_connected_to_output_nrns()

        a_nrns = [x for x in range(num_nrns) if x not in connected_nrn_indices]
        for a in a_nrns:
            if a in connected_to_input_nrns:
                connected_nrn_indices.append(a)
            elif a in connected_to_output_nrns:
                connected_nrn_indices.append(a)

        return connected_nrn_indices

    def delete_invalid_nrns(self, valid="active"):
        chrom = self.get_copy()
        num_nrns = chrom.get_num_nrns()
        if valid is "active":
            valid_nrns = chrom.get_active_nrns()
        else:
            valid_nrns = chrom.get_connected_nrns()

        invalid_nrns = [x for x in range(num_nrns) if x not in valid_nrns]

        if invalid_nrns:
            spikes = chrom.get_spikes()
            rules = chrom.get_rules()
            syn_matrix = chrom.get_syn_matrix()
            input_nrns = chrom.get_input_nrns()
            output_nrns = chrom.get_output_nrns()

            for index in sorted(invalid_nrns, reverse=True):
                spikes = spikes[:index] + spikes[index + 1:]
                rules = rules[:index] + rules[index + 1:]
                syn_matrix = syn_matrix[:index] + syn_matrix[index + 1:]
                for i in range(len(syn_matrix)):
                    syn_matrix[i] = syn_matrix[i][:index] + syn_matrix[i][index + 1:]
                for i in range(len(input_nrns)):
                    if input_nrns[i] > index:
                        input_nrns[i] -= 1
                for i in range(len(output_nrns)):
                    if output_nrns[i] > index:
                        output_nrns[i] -= 1
            chrom.set_num_nrns(len(valid_nrns))
            chrom.set_spikes(spikes)
            chrom.set_rules(rules)
            chrom.set_input_nrns(input_nrns)
            chrom.set_output_nrns(output_nrns)
            chrom.set_syn_matrix(syn_matrix)

        return chrom, invalid_nrns

    def is_exists_path_to_output_nrn(self):
        # Returns true if there is at least one path from an input neuron to an output neuron
        # There could actually be only output neurons -> env lol
        connected_to_input_nrns = self.get_connected_to_input_nrns()
        output_nrns = self.get_output_nrns()

        for o in output_nrns:
            if o in connected_to_input_nrns:
                return True
        return False
    
    def get_size(self):
        # Returns size of chromosome without inactive neurons for correct comparison
        active_nrns = self.get_active_nrns()
        num_nrns = len(active_nrns)
        num_nrns -= len(self.get_output_nrns())
        
        num_init_spikes = 0
        for index in active_nrns:
            num_init_spikes += self.get_spikes()[index]
        
        num_rules = 0
        rules = self.get_rules()
        for index in active_nrns:
            num_rules += len(rules[index])
            
        num_synapses = 0
        syn_matrix = self.get_syn_matrix()
        for row_index in active_nrns:
            for col_index in active_nrns:
                if syn_matrix[row_index][col_index] == 1:
                    num_synapses += 1
                    
        overall_size = num_nrns + num_init_spikes + num_rules + num_synapses
        
        return overall_size, num_nrns, num_init_spikes, num_rules, num_synapses, active_nrns

    def save_graph(self, name, directory):
        graph = Digraph(name=name, directory=directory, format="pdf")
        rules = self.get_rules()
        num_nrns = self.get_num_nrns()
        input_nrns = self.get_input_nrns()
        output_nrns = self.get_output_nrns()
        syn_matrix = self.get_syn_matrix()
        spikes = self.get_spikes()
        for i in range(num_nrns):
            label = ""
            if spikes[i] > 0:
                label += "a"
                if spikes[i] > 1:
                    label += "^" + str(spikes[i])
                label += "\n"
            for rule in rules[i]:
                if rule['y']:
                    label += "a^" + str(rule['x']) + "(a^" + str(rule['y']) + ")*/"
                elif not rule['y'] and rule['x'] != rule['c']:
                    label += "a^" + str(rule['x']) + "/"

                label += "a^" + str(rule['c']) + " -> a^" + str(rule['p'])

                if rule['delay']:
                    label += "; " + str(rule['delay'])
                label += " "
                label += "\n"

            if i in input_nrns:
                graph.attr('node', shape='parallelogram')
                graph.node(str(i), label=label, xlabel="in" + str(input_nrns.index(i)))
                graph.attr('node', shape='oval')
            elif i in output_nrns:
                graph.attr('node', shape='doublecircle')
                graph.node(str(i), label='<env>')
                graph.attr('node', shape='oval')
            else:
                graph.node(str(i), label=label, xlabel=str(i))

        for row in range(num_nrns):
            for col in range(num_nrns):
                if syn_matrix[row][col]:
                    graph.edge(str(row), str(col))

        try:
            graph.render(cleanup=True)
        except:
            pass

    def remove_self_loop(self):
        syn_matrix = self.get_syn_matrix()
        input_nrns = self.get_input_nrns()
        output_nrns = self.get_output_nrns()
        spikes = self.get_spikes()
        rules = self.get_rules()

        orig_num_nrns = self.get_num_nrns()
        added_nrns = []
        for nrn_id in range(orig_num_nrns):
            if syn_matrix[nrn_id][nrn_id] == 1 and nrn_id not in output_nrns:
                # New neuron id
                new_nrn_id = orig_num_nrns + len(added_nrns)
                added_nrns.append(new_nrn_id)

                if nrn_id in input_nrns:
                    input_nrns.append(new_nrn_id)

                # Remove self-loop
                syn_matrix[nrn_id][nrn_id] = 0

                # Create duplicate neuron
                spikes.append(spikes[nrn_id])
                rules.append(rules[nrn_id])
                curr_num_nrns = orig_num_nrns + len(added_nrns)
                # Add outgoing synapse from new duplicate neuron to original neuron
                new_row = [0 for _ in range(curr_num_nrns)]
                new_row[nrn_id] = 1
                syn_matrix.append(new_row)
                # Add ingoing synapses to new duplicate neuron from 1. the original neuron, and
                # 2. presynaptic neurons of ingoing synapses to the original neuron
                # Note: No need to append to new row
                for row_id in range(len(syn_matrix) - 1):
                    if syn_matrix[row_id][nrn_id] == 1 or row_id == nrn_id:
                        syn_matrix[row_id].append(1)
                    else:
                        syn_matrix[row_id].append(0)

        self.set_input_nrns(input_nrns)
        self.set_spikes(spikes)
        self.set_rules(rules)
        self.set_num_nrns(orig_num_nrns + len(added_nrns))
        self.set_syn_matrix(syn_matrix)


class Mutation:
    def __init__(self, population_size, mutation_rate, orig_chrom):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.orig_chrom = orig_chrom
        # TODO: Weighted operators?
        self.mutate_ops = [self.del_syn, self.add_syn, self.disc_nrn]
        self.order = [i for i in range(len(self.mutate_ops))]

    def del_syn(self, chrom):
        # Delete synapse
        num_nrns = chrom.get_num_nrns()
        output_nrns = chrom.get_output_nrns()
        syn_matrix = chrom.get_syn_matrix()

        choices = []
        for row_index in range(num_nrns):
            for col_index in range(num_nrns):
                if syn_matrix[row_index][col_index] == 1:
                    choices.append((row_index, col_index))

        shuffle(choices)

        success = False
        for index_pair in choices:
            syn_matrix[index_pair[0]][index_pair[1]] = 0
            chrom.set_syn_matrix(syn_matrix)
            if not syn_matrix == self.orig_chrom.get_syn_matrix() and chrom.is_exists_path_to_output_nrn():
                success = True
                break
            else:
                syn_matrix[index_pair[0]][index_pair[1]] = 1
                chrom.set_syn_matrix(syn_matrix)

        return chrom, success

    def add_syn(self, chrom):
        # Add synapse
        num_nrns = chrom.get_num_nrns()
        output_nrns = chrom.get_output_nrns()
        syn_matrix = chrom.get_syn_matrix()

        choices = []
        for row_index in range(num_nrns):
            if row_index not in output_nrns:
                for col_index in range(num_nrns):
                    if col_index not in output_nrns and syn_matrix[row_index][col_index] == 0:
                        choices.append((row_index, col_index))

        shuffle(choices)

        success = False
        for index_pair in choices:
            syn_matrix[index_pair[0]][index_pair[1]] = 1
            if not syn_matrix == self.orig_chrom.get_syn_matrix():
                chrom.set_syn_matrix(syn_matrix)
                success = True
                break
            else:
                syn_matrix[index_pair[0]][index_pair[1]] = 0

        return chrom, success

    @staticmethod
    def disc_nrn(chrom):
        # Delete neuron
        num_nrns = chrom.get_num_nrns()
        input_nrns = chrom.get_input_nrns()
        output_nrns = chrom.get_output_nrns()
        syn_matrix = chrom.get_syn_matrix()
        connected_nrns = chrom.get_connected_nrns()
        disconnected_nrns = [x for x in range(num_nrns) if x not in connected_nrns]

        choices = []
        for i in range(num_nrns):
            if i not in input_nrns and i not in output_nrns:
                choices.append(i)

        success = False
        while choices:
            # Input, output neurons cannot be disconnected
            # Already disconnected neurons can also not be disconnected
            index = randint(0, len(choices) - 1)
            nrn = choices[index]
            if not (nrn in output_nrns or nrn in input_nrns or nrn in disconnected_nrns):
                pre_syn_nrns = []
                post_syn_nrns = []

                for i in range(num_nrns):
                    if syn_matrix[i][nrn]:
                        pre_syn_nrns.append(i)
                    if syn_matrix[nrn][i]:
                        post_syn_nrns.append(i)

                # 'nrn' is disconnected by adding synapses from all pre-synaptic neurons of in-going synapses
                # to all post-synaptic neurons of outgoing synapses, and then deleting all in- and out-going synapses
                new_syn_matrix = []
                for row in syn_matrix:
                    new_syn_matrix.append(row.copy())

                for i in pre_syn_nrns:
                    new_syn_matrix[i][nrn] = 0
                    for j in post_syn_nrns:
                        new_syn_matrix[i][j] = 1

                for i in post_syn_nrns:
                    new_syn_matrix[nrn][i] = 0

                chrom.set_syn_matrix(new_syn_matrix)

                if chrom.is_exists_path_to_output_nrn():
                    success = True
                    break
                else:
                    chrom.set_syn_matrix(syn_matrix)

            del choices[index]

        return chrom, success

    def get_mutated_chrom(self, chrom):
        success = False
        shuffle(self.order)
        for index in self.order:
            chrom, success = self.mutate_ops[index](chrom)
            if success:
                break

        return chrom, success

    def do_mutate(self, population):
        mutated_population = []

        if len(population) == 1:
            while len(mutated_population) < self.population_size:
                new_chrom, success = self.get_mutated_chrom(population[0].get_copy())
                if not success:
                    raise ValueError("Original chromosome can't be mutated")

                mutated_population.append(new_chrom)

        else:
            # Randomly mutate all chromosomes in population
            for chrom in population:
                new_chrom = chrom.get_copy()
                if random() < self.mutation_rate:
                    new_chrom = self.get_mutated_chrom(new_chrom)[0]

                # Append new_chrom to mutated_population whether it was mutated or not
                mutated_population.append(new_chrom)

            # Mutate random chromosomes to fill up the population
            while len(mutated_population) < self.population_size:
                if population:
                    # Select a random chromosome in the population and mutate it
                    chrom = population[randint(0, len(population) - 1)]
                    new_chrom, success = self.get_mutated_chrom(chrom.get_copy())

                    if not success:
                        # Delete chrom from population so it can't be chosen next time
                        del population[population.index(chrom)]
                    else:
                        mutated_population.append(new_chrom)
                else:
                    break

        return mutated_population


class Selection:
    def __init__(self, population_size, fit_cases_key, selection_rate, orig_chrom=None):
        self.population_size = population_size
        self.best_chrom = orig_chrom  # best_chrom -> chromosome w highest precision
        self.best_gen = 0       # Generation that the best_chrom comes from
        self.orig_chrom = orig_chrom
        self.highest_precision = 0
        if self.orig_chrom:
            self.highest_precision = self.orig_chrom.get_precision()
        self.halt_evolution = False
        self.fit_cases_key = fit_cases_key
        self.num_evol_leaps = -1
        self.start_new_outer_loop = False
        self.selection_rate = selection_rate

    def is_halt_evolution(self):
        return self.halt_evolution

    def is_start_new_outer_loop(self):
        return self.start_new_outer_loop

    def get_best_chrom(self):
        return self.best_chrom

    def get_best_gen(self):
        return self.best_gen

    def get_num_evol_leaps(self):
        return self.num_evol_leaps

    def get_highest_precision(self):
        return self.highest_precision

    @staticmethod
    def calc_lcs_rate(actual_spt_copy, ideal_spt):
        # Todo: use generalized suffix tree
        # To get the precision, we must get the longest common substring between the actual spike train
        # outputted by the evolved SN P system and the ideal spike train outputted  by the
        # original SN P system
        longest_string = ''
        actual_spt = actual_spt_copy
        while len(actual_spt) > len(longest_string):
            sub_spt = actual_spt

            if len(longest_string) < 3:
                string = sub_spt[:3]
                sub_spt = sub_spt[3:]
            else:
                string = sub_spt[:len(longest_string)]
                sub_spt = sub_spt[len(longest_string):]

            while string in ideal_spt and sub_spt:
                string += sub_spt[0]
                sub_spt = sub_spt[1:]
            if string not in ideal_spt:
                # sub_spt = string[-1] + sub_spt
                string = string[:-1]

            if len(string) > len(longest_string):
                longest_string = string

            actual_spt = actual_spt[1:]

        lcs_rate = len(longest_string) / len(ideal_spt)
        return lcs_rate

    def set_precision_fitness(self, population, gen=0):
        # Set precision of each chromosome in the population
        avg_precision = 0
        highest_precision = self.highest_precision
        for chrom in population:
            precision = 0
            for case in fit_cases[self.fit_cases_key]:
                out_spt = simulate(in_spt=case['in'], parsed=Encoder.chrom_to_parsed_pli(chrom))
                precision += self.calc_lcs_rate(out_spt, case['out'])
            precision /= len(fit_cases[self.fit_cases_key])
            avg_precision += precision

            chrom.set_precision(precision)

            if precision > highest_precision \
                    or (precision == highest_precision and chrom.get_size() < self.best_chrom.get_size()):
                highest_precision = precision
                self.best_chrom = chrom
                self.best_gen = gen

        if highest_precision > self.highest_precision:
            self.num_evol_leaps += 1
            self.highest_precision = highest_precision

        # Set fitness of each chromosome in the population
        avg_precision /= len(population)
        for chrom in population:
            fitness = chrom.get_precision() / avg_precision
            chrom.set_fitness(fitness)

        # If ff conditions are met, halt current outer_loop and start a new one with self.best_chrom as new
        #   self.orig_chrom
        if self.orig_chrom \
                and self.highest_precision >= self.orig_chrom.get_precision() \
                and len(self.best_chrom.get_active_nrns()) < len(self.orig_chrom.get_active_nrns()):
            self.halt_evolution = True
            self.start_new_outer_loop = True

    def do_select(self, population):
        # Remove lowest ranking chromosomes from population (David et al.)
        ranked_population = sorted(population, key=methodcaller('get_fitness'), reverse=True)

        return ranked_population[:math.ceil(self.population_size*self.selection_rate)]


class Crossover:
    def __init__(self, crossover_rate, orig_chrom, population_size):
        self.crossover_rate = crossover_rate
        self.orig_chrom = orig_chrom
        self.population_size = population_size

    @staticmethod
    def get_parents(population):
        if len(population) > 1:
            index1 = randint(0, len(population) - 1)
            parent1 = population[index1]
            index2 = randint(0, len(population) - 1)
            while index1 == index2:
                index2 = randint(0, len(population) - 1)
            parent2 = population[index2]

            temp_population = {}
            for i in range(len(population)):
                temp_population[i] = population[i]
            del temp_population[index1]

            while temp_population and parent1.get_syn_matrix() == parent2.get_syn_matrix():
                del temp_population[index2]
                if temp_population:
                    index2 = list(temp_population.keys())[randint(0, len(temp_population) - 1)]
                    parent2 = temp_population[index2]

            if temp_population:
                return parent1, parent2, index1, index2
            return None

        return None

    @staticmethod
    def get_row_index(parent1, parent2):
        disconnected_nrns = [x for x in range(parent1.get_num_nrns()) if x not in parent1.get_connected_nrns()]
        disconnected_nrns += [x for x in range(parent1.get_num_nrns()) if x not in parent2.get_connected_nrns()]

        row_index = None
        is_row_index_valid = False
        choices = [num for num in range(len(parent1.get_syn_matrix()))]
        while choices:
            indx = randint(0, len(choices) - 1)
            row_index = choices[indx]
            del choices[indx]

            if not (row_index in parent1.get_output_nrns() or row_index in parent2.get_output_nrns() or
                    parent1.get_syn_matrix()[row_index] == parent2.get_syn_matrix()[row_index] or
                    row_index in disconnected_nrns):
                is_row_index_valid = True
                break

        if is_row_index_valid:
            return row_index
        return None

    @staticmethod
    def exchange_rows(parent1, parent2, row_index):
        # Get copy of synapse matrices of parents
        syn_matrix1 = parent1.get_syn_matrix()
        syn_matrix2 = parent2.get_syn_matrix()

        temp = syn_matrix1[row_index]
        syn_matrix1[row_index] = syn_matrix2[row_index]
        syn_matrix2[row_index] = temp

        # Disconnected neurons should stay disconnected
        disconnected_nrns1 = [x for x in range(parent1.get_num_nrns()) if x not in parent1.get_connected_nrns()]
        disconnected_nrns2 = [x for x in range(parent1.get_num_nrns()) if x not in parent2.get_connected_nrns()]
        if disconnected_nrns1 or disconnected_nrns2:
            for indx in range(len(syn_matrix1[row_index])):
                if indx in disconnected_nrns1:
                    syn_matrix1[row_index][indx] = 0
                if indx in disconnected_nrns2:
                    syn_matrix2[row_index][indx] = 0

        return syn_matrix1, syn_matrix2

    def do_crossover(self, population):
        # Crossover by swapping corresponding rows (Miller et al.)
        next_population = []

        if len(population) < self.population_size:
            for chrom in population:
                next_population.append(chrom)

            limit = len(population) * (len(population) - 1)
            misses = 0
            while len(next_population) + misses < self.population_size:
                misses += 2
                for _ in range(limit):
                    do_break = False
                    ret_val = self.get_parents(population)
                    if ret_val:
                        parent1, parent2 = ret_val[:2]
                        row_index = self.get_row_index(parent1, parent2)

                        if row_index:
                            syn_matrix1, syn_matrix2 = self.exchange_rows(parent1, parent2, row_index)

                            if not (syn_matrix1 == self.orig_chrom.get_syn_matrix() or
                                    syn_matrix2 == self.orig_chrom.get_syn_matrix()):
                                # Create children
                                child1 = parent1.get_copy()
                                child2 = parent2.get_copy()
                                child1.set_syn_matrix(syn_matrix1)
                                child2.set_syn_matrix(syn_matrix2)

                                if child1.is_exists_path_to_output_nrn():
                                    next_population.append(child1)
                                    misses -= 1
                                    do_break = True

                                if child2.is_exists_path_to_output_nrn():
                                    next_population.append(child2)
                                    misses -= 1
                                    do_break = True

                                if do_break:
                                    break

        else:
            while population:
                is_crossover_performed = False
                if len(population) > 1 and random() < self.crossover_rate:
                    ret_val = self.get_parents(population)
                    if ret_val:
                        parent1, parent2, index1, index2 = ret_val
                        row_index = self.get_row_index(parent1, parent2)

                        if row_index:
                            syn_matrix1, syn_matrix2 = self.exchange_rows(parent1, parent2, row_index)

                            if not (syn_matrix1 == self.orig_chrom.get_syn_matrix() or
                                    syn_matrix2 == self.orig_chrom.get_syn_matrix()):
                                # Create children
                                child1 = parent1.get_copy()
                                child2 = parent2.get_copy()
                                child1.set_syn_matrix(syn_matrix1)
                                child2.set_syn_matrix(syn_matrix2)

                                if child1.is_exists_path_to_output_nrn():
                                    next_population.append(child1)
                                    # Remove parent1
                                    population = population[:index1] + population[index1 + 1:]
                                    is_crossover_performed = True

                                if child2.is_exists_path_to_output_nrn():
                                    next_population.append(child2)
                                    # Remove parent2
                                    population = population[:index2] + population[index2 + 1:]
                                    is_crossover_performed = True

                if not is_crossover_performed:
                    # Choose random chromosome to place in next population
                    index = randint(0, len(population) - 1)
                    chrom = population[index]
                    population = population[:index] + population[index+1:]
                    next_population.append(chrom.get_copy())

        return next_population


class Encoder:
    @staticmethod
    def parsed_pli_to_chrom(parsed_pli):
        # Direct encoding
        nrn_names, spikes_copy, synapses, in_nrn_names, out_nrn_names, rules_copy = parsed_pli
        chrom = Chromosome(len(nrn_names))

        id_ = 0
        spikes = [0]*len(nrn_names)
        rules = [list() for _ in range(len(nrn_names))]
        syn_matrix = [[0] * len(nrn_names) for _ in range(len(nrn_names))]
        for name in nrn_names:
            # Encode spikes
            for key, value in spikes_copy.items():
                if key == name:
                    spikes[id_] += value
            chrom.set_spikes(spikes)

            # Change rule, input neuron, output neuron, and synapse names to corresponding id_ value
            for key, value in rules_copy.items():
                if key == name:
                    rules[id_] = value
            chrom.set_rules(rules)

            for name_index in range(len(in_nrn_names)):
                if in_nrn_names[name_index] == name:
                    in_nrn_names[name_index] = id_
            chrom.set_input_nrns(in_nrn_names)

            for name_index in range(len(out_nrn_names)):
                if out_nrn_names[name_index] == name:
                    out_nrn_names[name_index] = id_
            chrom.set_output_nrns(out_nrn_names)

            for syn_index in range(len(synapses)):
                if synapses[syn_index][0] == name:
                    syn = (id_, synapses[syn_index][1])
                    synapses[syn_index] = syn
                if synapses[syn_index][1] == name:
                    syn = (synapses[syn_index][0], id_)
                    synapses[syn_index] = syn

            id_ += 1

        # Encode synapses
        for syn in synapses:
            syn_matrix[syn[0]][syn[1]] = 1
        chrom.set_syn_matrix(syn_matrix)

        return chrom

    @staticmethod
    def chrom_to_parsed_pli(chrom):
        num_nrns = chrom.get_num_nrns()
        spikes_copy = chrom.get_spikes()
        spikes = {}
        rules_copy = chrom.get_rules()
        rules = {}
        input_nrns = chrom.get_input_nrns()
        output_nrns = chrom.get_output_nrns()
        syn_matrix = chrom.get_syn_matrix()

        nrn_names = [0] * num_nrns
        # input_nrns
        for index in range(len(input_nrns)):
            name = 'in' + str(index + 1)
            nrn_names[input_nrns[index]] = name

            if spikes_copy[input_nrns[index]]:
                spikes[name] = spikes_copy[input_nrns[index]]

            if rules_copy[input_nrns[index]]:
                rules[name] = rules_copy[input_nrns[index]]

            input_nrns[index] = name

        # output_nrns
        for index in range(len(output_nrns)):
            name = 'out' + str(index + 1)
            nrn_names[output_nrns[index]] = name

            if spikes_copy[output_nrns[index]]:
                spikes[name] = spikes_copy[output_nrns[index]]

            if rules_copy[output_nrns[index]]:
                rules[name] = rules_copy[output_nrns[index]]

            output_nrns[index] = name

        # rest of the nrns
        counter = 1
        for index in range(num_nrns):
            if nrn_names[index] == 0:
                name = str(counter)
                nrn_names[index] = name

                if spikes_copy[index]:
                    spikes[name] = spikes_copy[index]

                if rules_copy[index]:
                    rules[name] = rules_copy[index]

                counter += 1

        # synapses
        synapses = []
        for row in range(num_nrns):
            for col in range(num_nrns):
                if syn_matrix[row][col]:
                    synapses.append((nrn_names[row], nrn_names[col]))

        parsed_pli = nrn_names, spikes, synapses, input_nrns, output_nrns, rules
        return parsed_pli

    @staticmethod
    def chrom_to_pli(chrom, gen=-1, num_evol_leaps=0, runtime=0.0, directory='/', filename='', extension=''):
        precision = chrom.get_precision()
        parsed_pli = Encoder.chrom_to_parsed_pli(chrom=chrom)
        nrn_names, spikes, synapses, in_nrn_names, out_nrn_names, rules = parsed_pli
        overall_size, num_nrns, num_init_spikes, num_rules, num_synapses, active_nrns = \
            chrom.get_size()

        pli = "precision: " + str(precision)

        if not extension or extension == 'final':
            pli += "\ngeneration: " + str(gen)
            if num_evol_leaps:
                pli += "\nevolution leaps: " + str(num_evol_leaps)
            if runtime:
                pli += "\nruntime (s): " + str(runtime)

        pli += "\noverall_size = " + str(overall_size)
        pli += "\nnum_nrns = " + str(num_nrns)
        pli += "\nnum_init_spikes = " + str(num_init_spikes)
        pli += "\nnum_rules = " + str(num_rules)
        pli += "\nnum_synapses = " + str(num_synapses)

        pli += "\n\n@mu = " + ''.join(str(nrn_names)[1:-1].split("'")) + "\n"
        for name, spike in spikes.items():
            if spike == 1:
                pli += "@ms(" + name + ") = a" + "\n"
            else:
                pli += "@ms(" + name + ") = a*" + str(spike) + "\n"
        for syn in synapses:
            pli += "@marcs = " + ''.join(str(syn).split("'")) + "\n"
        pli += "@min = " + ''.join(str(in_nrn_names)[1:-1].split("'")) + "\n"
        pli += "@mout = " + ''.join(str(out_nrn_names)[1:-1].split("'")) + "\n"
        for name, rule_list in rules.items():
            for rule in rule_list:
                if rule['c'] == 1:
                    pli += "[a --> "
                else:
                    pli += "[a*" + str(rule['c']) + " --> "

                if rule['p']:
                    if rule['p'] == 1:
                        pli += "a]"
                    else:
                        pli += "a*" + str(rule['p']) + "]"
                else:
                    pli += '#]'

                pli += "'" + name

                if rule['x'] and rule['x'] != rule['c']:
                    if rule['x'] == 1:
                        pli += ' "a" '
                    else:
                        pli += ' "a*' + str(rule['x']) + '" '

                if rule['delay']:
                    pli += " :: " + str(rule['delay'])

                pli += "\n"

        if filename:
            if not os.path.isdir(directory):
                os.mkdir(directory)
            if not extension:
                f = open(directory + '/' + filename + '-' + datetime.now().strftime("%y%m%d-%H%M%S-%f") + '.pli', 'w')
            else:
                f = open(directory + '/' + filename + '-' + extension + '.pli', 'w')

            f.write(pli)
            f.close()

        return pli


def my_write(file, string):
    f = open(file, 'a')
    f.write(string + "\n")
    f.close()
    print(string)
    

if __name__ == "__main__":
    ##########################################################################################
    #                                   Default Parameters                                   #
    ##########################################################################################
    """
        - mutation_rate: probability that a chromosome will be mutated
        - crossover_rate: probability that child chromosome/s will be created through crossover
    """
    num_runs = 5
    num_inner_loops = 10
    max_gen = 75
    population_size = 80

    mutation_rate = 0.5
    crossover_rate = 0.3
    selection_rate = 0.6

    precision_threshold = 0

    same_init_pop = True
    ##########################################################################################
    ##########################################################################################

    try:
        opts, args = getopt(sys.argv[1:], 'hd:f:c:r:p:i::')
    except GetoptError:
        print("Error! Use format: ga_framework.py -f <filename> -d init_pli/<directory-of-pli-file> -c "
              "<fit_cases key> [-r <num_runs, int, default: 5>, -p <population_size, int, default: 80>, -i to generate init pop]")
        sys.exit(2)

    if not opts:
        print("Error! Use format: ga_framework.py -f <filename> -d init_pli/<directory-of-pli-file> -c "
              "<fit_cases key> [-r <num_runs, int, default: 5>, -p <population_size, int, default: 80>, -i to generate init pop]")
        sys.exit(2)

    init_pli_file, init_pli_dir, fit_cases_key = None, None, None
    for opt, arg in opts:
        if opt == '-h':
            print("ga_framework.py -f <filename> -d init_pli/<directory-of-pli-file> -c "
                  "<fit_cases key> [-r <num_runs, int, default: 5>, -p <population_size, int, default: 80>, -i to generate init pop]")
        elif opt == '-d':
            init_pli_dir = 'init_pli/' + arg
        elif opt == '-f':
            init_pli_file = arg
        elif opt == '-c':
            fit_cases_key = arg
        elif opt == '-r':
            try:
                num_runs = int(arg)
                if num_runs < 1:
                    print("ValueError: argument -r must be int > 0")
                    sys.exit(2)
            except ValueError:
                print("ValueError: argument -r must be int > 0")
                sys.exit(2)
        elif opt == '-p':
            try:
                population_size = int(arg)
                if population_size < 1:
                    print("ValueError: argument -p must be int > 0")
                    sys.exit(2)
            except ValueError:
                print("ValueError: argument -p must be int > 0")
                sys.exit(2)
        elif opt == '-i':
            same_init_pop = False

    # Encode input SN P sytem to chrom
    parsed = parse(filename=init_pli_dir + '/' + init_pli_file + '.pli')

    base_output_dir = "outputs-des3/"
    if not os.path.isdir(base_output_dir):
        os.mkdir(base_output_dir)
    base_output_dir += "outputs_" + init_pli_file + "/"
    if not os.path.isdir(base_output_dir):
        os.mkdir(base_output_dir)
    num_runs_num_digits = len(str(num_runs))
    total_time = 0
    for run in range(num_runs):
        missing_zeroes = ''
        for _ in range(num_runs_num_digits - len(str(run))):
            missing_zeroes += '0'
        output_dir = base_output_dir + '/' + init_pli_file + '-' + missing_zeroes + str(run+1) + '/'
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        valid = "active"

        inner_loop = 0
        outer_loop = -1
        selection = None
        outer_start_time = time()
        overall_best_chrom = None
        start_new_outer_loop = True
        while start_new_outer_loop:
            outer_loop += 1
            if overall_best_chrom:
                chrom, deleted_nrns = overall_best_chrom.delete_invalid_nrns(valid=valid)
                while deleted_nrns:
                    chrom, deleted_nrns = chrom.delete_invalid_nrns(valid=valid)
            else:
                chrom = Encoder.parsed_pli_to_chrom(parsed)

            start_new_outer_loop = False

            orig_chrom = chrom.get_copy()
            s = Selection(population_size=population_size,
                          fit_cases_key=fit_cases_key, selection_rate=selection_rate)
            s.set_precision_fitness([orig_chrom])

            # Create outer loop directory
            outer_loop_dir = output_dir + '/' + init_pli_file + "-" + datetime.now().strftime("%y%m%d-%H%M%S-%f") + "/"
            os.mkdir(outer_loop_dir)

            if outer_loop == 0:
                Encoder.chrom_to_pli(chrom=orig_chrom, directory=output_dir, filename=init_pli_file, extension="init")
                orig_chrom.save_graph(name=init_pli_file + '-init', directory=output_dir)
            Encoder.chrom_to_pli(chrom=orig_chrom, directory=outer_loop_dir, filename=init_pli_file, extension="init")
            orig_chrom.save_graph(name=init_pli_file + '-init', directory=outer_loop_dir)

            f = open(outer_loop_dir + '_info.py', 'w')
            f.write('"""')
            f.write("\n    ~ GA Design ~")
            f.write("\n    Encoding: Direct encoding [Miller et al., 1989]")
            f.write("\n    Precision (evaluation) function: lcs rate between binary strings")
            f.write("\n    Fitness function: precision/avg_precision (unused) [Whitley, 1994]")
            f.write("\n    Halting: highest_precision > precision threshold; max number of generations is reached")
            f.write("\n    Selection: Selects " + str(population_size) + " higher ranking chromosomes (no effect)")
            f.write("\n    Mutation: ")
            f.write("\n        First mutation produces the initial population with size " + str(population_size))
            f.write("\n        Equal probability of choosing mutation operators")
            f.write("\n        del_syn -> Ingoing synapses to output neurons cannot be deleted")
            f.write("\n        add_syn -> Output neurons can't be pre-synaptic neurons")
            f.write("\n        disc_nrn -> Input and output neurons cannot be disconnected")
            f.write("\n    Crossover: Switching rows between two random parents")
            f.write('\n"""')

            f.write("\n\n# GA Parameters")
            f.write("\npopulation_size = " + str(population_size))
            f.write("\nmax_gen = " + str(max_gen))
            f.write("\nprecision_threshold = " + str(precision_threshold))
            f.write("\nmutation_rate = " + str(mutation_rate))
            f.write("\ncrossover_rate = " + str(crossover_rate))
            f.close()

            overall_best_chrom = orig_chrom
            avg_runtime = 0
            avg_size_of_best_chroms = 0

            # Performance evaluation measures
            Lopt = f_bar = Lel = 0
            G_best = {}  # Self-created??? G_best(k) is the likelihood that the best chrom is from the kth generation
            Lhalt = {}   # Self-created??? Lhalt(k) is the likelihood of halting at the kth generation in n inner_loops
            for inner_loop in range(num_inner_loops):
                # for operation specific time
                gen_op_stats = "= Gen 0" + " - "
                gen_op_start = 0
                # Create inner loop directory
                inner_loop_dir = outer_loop_dir + init_pli_file + "-" + \
                                 datetime.now().strftime("%y%m%d-%H%M%S-%f") + "/"
                os.mkdir(inner_loop_dir)

                # Initialize classes
                selection = Selection(population_size=population_size,
                                      fit_cases_key=fit_cases_key, selection_rate=selection_rate, orig_chrom=orig_chrom)
                crossover = Crossover(crossover_rate=crossover_rate, orig_chrom=orig_chrom,
                                      population_size=population_size)
                mutation = Mutation(population_size=population_size, mutation_rate=mutation_rate, orig_chrom=orig_chrom)

                ###############################################
                # Getting corresponding initial population
                # Added by Rogelio Gungon and Katreen Hernandez
                ###############################################
                if same_init_pop == True and outer_loop == 0 and inner_loop == 0:
                    print("Getting initial population...")
                    population = []
                    for i in init_population[init_pli_file][0]:
                        population.append(Encoder.parsed_pli_to_chrom(i[0]))
                    gen_op_stats += "M: "+str(init_population[init_pli_file][1]-init_population[init_pli_file][2])+"; "
                else:
                    try:
                        print("Mutation process...")
                        gen_op_start = time()
                        population = mutation.do_mutate(population=[orig_chrom])
                        gen_op_stats += "M: "+str(time()-gen_op_start)+"; "
                    except ValueError:
                        f = open(outer_loop_dir + '_info.py', 'a')
                        f.write('\n\n"""')
                        f.write("\n    Can't mutate original chromosome.")
                        f.write('\n"""\n')
                        f.close()
                        raise

                # Start evolution process
                gen = 0
                start_time = time()
                do_print = False
                for gen in range(max_gen):
                    # Create GA loop directory
                    ga_loop_dir = inner_loop_dir + "generation-" + str(gen+1) + "/"
                    os.mkdir(ga_loop_dir)

                    for chrom in population:
                        extension = datetime.now().strftime("%y%m%d-%H%M%S-%f")
                        Encoder.chrom_to_pli(chrom=chrom, gen=gen+1, directory=ga_loop_dir, filename=init_pli_file,
                                             extension=extension)
                        chrom.save_graph(name=init_pli_file + '-' + extension, directory=ga_loop_dir)
                    print("before setting fitness")

                    gen_op_start = time()
                    # Calculate precision and fitness
                    selection.set_precision_fitness(population, gen+1)
                    gen_op_stats += "SIM: "+str(time()-gen_op_start)+"; "

                    print("before convergence check")

                    # Check for convergence; halt if max_gen is reached
                    if selection.is_halt_evolution() or gen+1 == max_gen:
                        print("Generation " + str(gen+1) + " halted")
                        # print(len(population))
                        my_write(base_output_dir + "/" + init_pli_file + ".out",
                            gen_op_stats+"halt")
                        break
                    print("before selection")

                    # Perform selection
                    gen_op_start = time()
                    population = selection.do_select(population)
                    gen_op_stats += "SEL: "+str(time()-gen_op_start)+"; "
                    print("before crossover")

                    # Perform crossover
                    gen_op_start = time()
                    population = crossover.do_crossover(population)
                    gen_op_stats += "C: "+str(time()-gen_op_start)
                    print("before mutation")

                    my_write(base_output_dir + "/" + init_pli_file + ".out",
                        gen_op_stats)

                    gen_op_stats = "= Gen "+str(gen+1)+ " - "
                    # Perform mutation
                    gen_op_start = time()
                    population = mutation.do_mutate(population)
                    gen_op_stats += "M: "+str(time()-gen_op_start)+"; "
                    if len(population) < population_size:
                        do_print = True
                    print("after mutation")

                my_write(base_output_dir + "/" + init_pli_file + ".out",
                         "Outer loop " + str(outer_loop+1) + " - Inner loop " + str(inner_loop+1) + " - Generation " +
                         str(gen+1) + " - " + str(time()-start_time) + "s")
                if do_print:
                    my_write(base_output_dir + "/" + init_pli_file + ".out",
                             "len(population) < population_size from mutation")

                Encoder.chrom_to_pli(chrom=selection.get_best_chrom(), gen=selection.get_best_gen(),
                                     num_evol_leaps=selection.get_num_evol_leaps(), runtime=time()-start_time,
                                     directory=inner_loop_dir, filename=init_pli_file, extension="final")
                selection.get_best_chrom().save_graph(name=init_pli_file + '-final', directory=inner_loop_dir)

                avg_runtime += time()-start_time
                avg_size_of_best_chroms += selection.get_best_chrom().get_size()[0]

                if overall_best_chrom is None \
                        or overall_best_chrom.get_precision() < selection.get_best_chrom().get_precision() \
                        or (overall_best_chrom.get_precision() == selection.get_best_chrom().get_precision()
                            and overall_best_chrom.get_size() > selection.get_best_chrom().get_size()):
                    overall_best_chrom = selection.get_best_chrom().get_copy()

                if selection.get_best_gen() in G_best.keys():
                    G_best[selection.get_best_gen()] += 1
                else:
                    G_best[selection.get_best_gen()] = 1

                if gen+1 in Lhalt.keys():
                    Lhalt[gen+1] += 1
                else:
                    Lhalt[gen+1] = 1

                f_bar += selection.get_highest_precision()
                Lel += selection.get_num_evol_leaps()
                if precision_threshold and selection.get_highest_precision() >= precision_threshold:
                    Lopt += 1
                elif not precision_threshold and selection.get_highest_precision() == 1:
                    Lopt += 1

                if selection.is_start_new_outer_loop() \
                        or overall_best_chrom.get_precision() > orig_chrom.get_precision() \
                        or (overall_best_chrom.get_precision() == orig_chrom.get_precision()
                            and overall_best_chrom.get_size() < orig_chrom.get_size()):
                    start_new_outer_loop = True
                    break
                elif selection.get_highest_precision() == 1:
                    break

            Encoder.chrom_to_pli(chrom=overall_best_chrom, gen=selection.get_best_gen(),
                                 num_evol_leaps=selection.get_num_evol_leaps(), directory=outer_loop_dir,
                                 filename=init_pli_file, extension="final")
            overall_best_chrom.save_graph(name=init_pli_file + '-final', directory=outer_loop_dir)

            f = open(outer_loop_dir + '_info.py', 'a')

            inner_loops = num_inner_loops
            if start_new_outer_loop:
                if selection.get_highest_precision() == 1:
                    valid = "active"
                f.write("\n\n# NEW EXPERIMENT STARTED. Last inner_loop #: " + str(inner_loop+1))
                inner_loops = inner_loop+1

            f.write("\nnum_inner_loops = " + str(num_inner_loops))
            f.write("\nactual_inner_loops = " + str(inner_loops))
            f.write("\n\n# Performance Evaluation Measures")
            f.write("\n# G_best[k]: Likelihood of best chrom coming from the kth generation")
            f.write("\nG_best = dict()")
            for key, value in sorted(G_best.items(), key=itemgetter(1), reverse=True):
                f.write("\nG_best[" + str(key) + "] = " + str(value/inner_loops))

            f.write("\n# Lhalt[k]: Likelihood of halting in k generations")
            f.write("\nLhalt = dict()")
            for key, value in sorted(Lhalt.items(), key=itemgetter(1), reverse=True):
                f.write("\nLhalt[" + str(key) + "] = " + str(value/inner_loops))

            f.write("\n# Lopt[k]: Likelihood of optimality with k = max_gen")
            f.write("\nLopt = dict()")
            f.write("\nLopt[" + str(max_gen) + "] = " + str(Lopt/inner_loops))

            f.write("\n# f_bar[k]: Average precision (fitness) value with k = max_gen")
            f.write("\nf_bar = dict()")
            f.write("\nf_bar[" + str(max_gen) + "] = " + str(f_bar/inner_loops))

            f.write("\n# Lel[k]: Likelihood of evolution leap with k = max_gen")
            f.write("\nLel = dict()")
            f.write("\nLel[" + str(max_gen) + "] = " + str(Lel/inner_loops))

            f.write("\n\n# Overall best chromosome in the outer_loop")
            f.write("\n# Size")
            overall_size, num_nrns, num_init_spikes, num_rules, num_synapses = \
                overall_best_chrom.get_size()[:5]
            f.write("\noverall_size_best = " + str(overall_size))
            f.write("\nnum_nrns_best = " + str(num_nrns))
            f.write("\nnum_init_spikes_best = " + str(num_init_spikes))
            f.write("\nnum_rules_best = " + str(num_rules))
            f.write("\nnum_synapses_best = " + str(num_synapses))
            f.write("\n# Precision")
            f.write("\nprecision_best = " + str(overall_best_chrom.get_precision()))

            f.write("\n\n# Original chromosome")
            f.write("\n# Size")
            overall_size, num_nrns, num_init_spikes, num_rules, num_synapses = orig_chrom.get_size()[:5]
            f.write("\noverall_size_orig = " + str(overall_size))
            f.write("\nnum_nrns_orig = " + str(num_nrns))
            f.write("\nnum_init_spikes_orig = " + str(num_init_spikes))
            f.write("\nnum_rules_orig = " + str(num_rules))
            f.write("\nnum_synapses_orig = " + str(num_synapses))
            f.write("\n# Precision")
            f.write("\nprecision_orig = " + str(orig_chrom.get_precision()))

            f.write("\n\n# Average runtime in seconds")
            f.write("\navg_runtime = " + str(avg_runtime/inner_loops))
            f.write("\n# Average size of best chromosomes")
            f.write("\navg_size_of_best_chroms = " + str(avg_size_of_best_chroms/inner_loops))

            f.write("\n")

            f.close()

        overall_best_chrom.remove_self_loop()
        Encoder.chrom_to_pli(chrom=overall_best_chrom, gen=selection.get_best_gen(),
                             num_evol_leaps=selection.get_num_evol_leaps(), directory=output_dir,
                             filename=init_pli_file, extension="final")
        overall_best_chrom.save_graph(name=init_pli_file + '-final', directory=output_dir)
        outer_loop_time = time() - outer_start_time
        total_time += outer_loop_time
        my_write(base_output_dir + "/" + init_pli_file + ".out",
                 "Run #" + str(run) + " total time: " + str(outer_loop_time))
    my_write(base_output_dir + "/" + init_pli_file + ".out", "Total time: " + str(total_time))
