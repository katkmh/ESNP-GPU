################################
# Evolving SNP System with GA algorithm
# GA Framework
# Mutation, Simulator, and Crossover in GPU
#   - Evolution taken from Casauay et al
#   - Simulator on GPU taken from Aboy et al,
#       modified with Casauay et al to use spike trains
#
# Last modified by Rogelio Gungon and Katreen Hernandez
# on January 3, 2021
################################

from pli_parser import parse
from fitness_cases import fit_cases
from initial_population import init_population
from operator import methodcaller, itemgetter
from time import time
from datetime import datetime
from graphviz import Digraph
from getopt import getopt, GetoptError
import sys
import os
import math, random

import numpy
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import driver, compiler, gpuarray, tools

from string import Template
import skcuda.misc as misc


class Chromosome:
    """
    Represents an SNP System recognizable by the GA Function
    """
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

    def get_swappable_nrns(self):
        "All neurons sans output and disconnected neurons"
        disconnected_nrns = [x for x in range(self.get_num_nrns()) if x not in self.get_connected_nrns()]
        swappable = [x for x in range(self.get_num_nrns()) if x not in disconnected_nrns and x not in self.get_output_nrns()]

        return swappable

    def get_data(self):
        output = ""
        output += "Num of neurons: "+ str(self.get_num_nrns()) + "\n"
        output += "Precision: " + str(self.get_precision()) + "\n"
        output += "Fitness: "+ str(self.get_fitness()) + "\n"
        output += "Spikes: "+ str(self.get_spikes()) + "\n"

        output += "Rules:\n"
        rules = self.get_rules()
        ctr = 0
        for nrn in rules:
            output += "\tNeuron " + str(ctr) + "\n"
            for rule in nrn:
                output += "\t\t" + str(rule) + "\n"
            ctr+=1
        
        output += "Input neurons: "+ str(self.get_input_nrns()) + "\n"
        output += "Output neurons: "+ str(self.get_output_nrns()) + "\n"

        output += "Syn Matrix:\n"
        syn_matrix = self.get_syn_matrix()
        for syn in syn_matrix:
            output += "\t\t" + str(syn) + "\n"

        output += "Active neurons: "+ str(self.get_active_nrns()) + "\n"
        output += "Connected neurons: "+ str(self.get_connected_nrns()) + "\n"
        output += "Connected to input nrns: "+ str(self.get_connected_to_input_nrns()) + "\n"
        output += "Connected to output nrns: "+ str(self.get_connected_to_output_nrns()) + "\n"
        return output

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

class Nrn:
    """
    Represents a neuron of an SNP System,
    taken from *.pli files
    """
    def __init__(self, idn, name, a=0):
        self.idn = idn
        self.name = name
        self.a = a
        self.rules = []
        self.distrib = {'delay': -1, 'a': 0, 'receivers': []}

    def enter_spikes(self, num):
        if self.distrib['delay'] <= 0:
            self.a += num

    def add_rule(self, E, c, p, delay=0):
        self.rules.append(Rules(E, c, p, self, delay))

class Rules:
    """
    Represents the rules of an SNP System,
    taken from *.pli files
    """
    def __init__(self, E, c, p, nrn, delay=0):

        self.x = E[0]
        self.y = E[1]
        self.c = c
        self.p = p
        self.nrn = nrn
        self.delay = delay

def get_max_spt(in_spt):
    # Get the length of the longest spike train
    lens = []
    for i in in_spt:
        lens.append(len(i))
    return max(lens)


def is_binary_string(spt):
    if not spt.isdigit():
        return False
    for i in spt:
        if i != '0' and i != '1':
            return False
    return True

class Selection:
    """
    Selects subset of most fit chromosomes
    in the population with regard to the selection rate
    """
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

    def simulate(self, population):
        """
        This method runs partially parallel to simulate
        all chromosomes of the population given all input
        spike trains, preparation of input arrays to be 
        copied to the device are also processed here
        """

        pop_size = len(population)
        
        # Getting parsed data lists from population
        nrn_names = []
        spikes = []
        synapses = []
        in_nrn_names = []
        out_nrn_names = []
        rules = []
        for chrom in population:
            temp = Encoder.chrom_to_parsed_pli(chrom)
            nrn_names.append(temp[0])
            spikes.append(temp[1])
            synapses.append(temp[2])
            in_nrn_names.append(temp[3])
            out_nrn_names.append(temp[4])
            rules.append(temp[5])

        transfer_time_start = 0    # for subtracting time to copy files from host-data v.v.
        transfer_time = 0

        N_list= []      # list number of rules
        M_list = []     # list of number of neurons
        # get list of list of neurons
        neurons = [[] for i in range(pop_size)]
        for pop_ctr in range(pop_size):
            temp_list = []
            temp_N = 0
            for name in nrn_names[pop_ctr]:
                
                try:
                    n = Nrn(idn=len(temp_list), name=name, a=spikes[pop_ctr][name])
                except KeyError:
                    n = Nrn(idn=len(temp_list), name=name)

                try:
                    for rule in rules[pop_ctr][name]:
                        n.add_rule(E=(rule['x'], rule['y']), c=rule['c'], p=rule['p'], delay=rule['delay'])
                        temp_N += 1
                except KeyError:
                    pass
                finally:
                    temp_list.append(n)
            
                neurons[pop_ctr] = temp_list
            M_list.append(len(nrn_names[pop_ctr]))
            N_list.append(temp_N)

        # initialize spike trains
        in_spt = []
        for case in fit_cases[self.fit_cases_key]:
            in_spt.extend([case['in'][0],case['in'][1]])
        max_spt = get_max_spt(in_spt)

        # getting max size to equalize arrays
        N = max(N_list)
        M = max(M_list)
        MAX_SIZE = max([N*3, N*M, max_spt*2, max_spt+M])
        # MAX_SIZE needs to be divisible by 2 because of
        # the nature of the h_in_spt list
        if (MAX_SIZE%2==1):
            MAX_SIZE+=1

        h_in_spt = []
        # Add 0s to spike trains shorter than the maximum length
        for spt_index in range(len(in_spt)):
            add_0 = (MAX_SIZE//2) - len(in_spt[spt_index])
            in_spt[spt_index] += '0'*add_0

        # get 2d array of input spike trains [case,concatenated_spt]
        temp = ''
        i = 1
        while i < len(in_spt):
            h_in_spt.append([int(x) for x in (in_spt[i-1]+in_spt[i])])
            i+=2

        # initialize 2d h_out_spt according to number of cases
        CASES = len(h_in_spt)
        h_in_spt = [h_in_spt]*pop_size
        h_out_spt = numpy.zeros(pop_size * CASES * MAX_SIZE).reshape(pop_size,CASES,MAX_SIZE)
        
        # get synapse matrix
        syn_matrix = []
        for chrom in population:
            syn_matrix.append(chrom.syn_matrix)
        syn_matrix = numpy.array(syn_matrix)
        
        # initializing vectors
        h_Sv = numpy.zeros(pop_size * CASES * MAX_SIZE).reshape(pop_size,CASES,MAX_SIZE)             # (N) spiking vector
        h_St = numpy.ones(pop_size * CASES * MAX_SIZE).reshape(pop_size,CASES,MAX_SIZE)              # (M) status vector
        h_Lv = numpy.zeros(pop_size * CASES * MAX_SIZE).reshape(pop_size,CASES,MAX_SIZE)             # (M) loss vector
        h_Gv = numpy.zeros(pop_size * CASES * MAX_SIZE).reshape(pop_size,CASES,MAX_SIZE)             # (M) gain vector
        h_NG = numpy.zeros(pop_size * CASES * MAX_SIZE).reshape(pop_size,CASES,MAX_SIZE)             # (M) net gain vector
        
        h_Iv = numpy.zeros(pop_size * CASES * MAX_SIZE).reshape(pop_size,CASES,MAX_SIZE)             # (N) indicator vector

        h_rules = [[] for i in range(pop_size)]          # (N*3) rules representation
        h_delays = [[] for i in range(pop_size)]         # (N) delays vector
        h_lhs = [[] for i in range(pop_size)]            # (N) left hand side of rules
        h_TMv = [[] for i in range(pop_size)]            # (N*M) transition matrix vector

        rule_list = [[] for i in range(pop_size)]
        rule_owner_list = [[] for i in range(pop_size)]
        h_in_idx = [[] for i in range(pop_size)]                        # (num_in_nrns)
        num_in_nrns = []
        h_Cv = [[] for i in range(pop_size)]                        # (M) config vector 
        h_out_idx = []                                  # only caters to single output

        for pop_ctr in range(pop_size):
            temp_Cv = []
            temp_in_idx = []
            for nrn in neurons[pop_ctr]:
                # get list of rule owners and list of rules
                if nrn.name not in out_nrn_names[pop_ctr]:
                    for rule in rules[pop_ctr][nrn.name]:
                        rule_owner_list[pop_ctr].append(nrn.idn)
                        rule_list[pop_ctr].append(rule)

                # get indexes of input/output neurons
                if nrn.name in in_nrn_names[pop_ctr]:
                    temp_in_idx.append(nrn.idn)
                if nrn.name in out_nrn_names[pop_ctr]:
                    h_out_idx.append(nrn.idn)
                # get initial config vector
                temp_Cv.append(nrn.a)

            # padding config vector
            temp_Cv.extend([0]*(MAX_SIZE - len(temp_Cv)))
            num_in_nrns.append(len(temp_in_idx))
            temp_in_idx.extend([0]*(MAX_SIZE - len(temp_in_idx)))
            h_in_idx[pop_ctr] = [temp_in_idx]*CASES
            # cloning config vector of population by num of cases
            h_Cv[pop_ctr] = [temp_Cv]*CASES

        # get rules, delays, TMv
        for pop_ctr in range(pop_size):
            temp_rules = numpy.zeros(MAX_SIZE)
            temp_delays = numpy.zeros(MAX_SIZE)
            temp_lhs = numpy.zeros(MAX_SIZE)
            temp_TMv = numpy.zeros(MAX_SIZE)
            for i in range(N_list[pop_ctr]):
                temp_rules[i*3] = rule_owner_list[pop_ctr][i]
                temp_rules[i*3 + 1] = -1
                temp_rules[i*3 + 2] = -rule_list[pop_ctr][i]['c']
                temp_delays[i] = rule_list[pop_ctr][i]['delay']
                temp_lhs[i] = rule_list[pop_ctr][i]['x']
                j = rule_owner_list[pop_ctr][i]
                for k in range(M_list[pop_ctr]):
                    if syn_matrix[pop_ctr][j][k] == 1:
                        temp_TMv[i*M_list[pop_ctr]+k] = rule_list[pop_ctr][i]['p']
            h_rules[pop_ctr] = [temp_rules]*CASES
            h_delays[pop_ctr] = [temp_delays]*CASES
            h_lhs[pop_ctr] = [temp_lhs]*CASES
            h_rules[pop_ctr] = [temp_rules]*CASES
            h_TMv[pop_ctr] = [temp_TMv]*CASES

        transfer_time_start = time()
        # allocate memory
        num_in_nrns = numpy.array(num_in_nrns,numpy.int32)
        d_num_in_nrns = cuda.mem_alloc(num_in_nrns.nbytes)
        cuda.memcpy_htod(d_num_in_nrns, num_in_nrns)


        h_in_spt = numpy.array(h_in_spt,numpy.int32)
        d_in_spt = cuda.mem_alloc(h_in_spt.nbytes)
        cuda.memcpy_htod(d_in_spt, h_in_spt)

        h_out_spt = numpy.array(h_out_spt,numpy.int32)
        d_out_spt = cuda.mem_alloc(h_out_spt.nbytes)
        cuda.memcpy_htod(d_out_spt, h_out_spt)

        h_in_idx = numpy.array(h_in_idx,numpy.int32)
        d_in_idx = cuda.mem_alloc(h_in_idx.nbytes)
        cuda.memcpy_htod(d_in_idx, h_in_idx)

        h_out_idx = numpy.array(h_out_idx,numpy.int32)
        d_out_idx = cuda.mem_alloc(h_out_idx.nbytes)
        cuda.memcpy_htod(d_out_idx, h_out_idx)

        
        N_list = numpy.array(N_list,numpy.int32)
        d_N_list = cuda.mem_alloc(N_list.nbytes)
        cuda.memcpy_htod(d_N_list, N_list)

        M_list = numpy.array(M_list,numpy.int32)
        d_M_list = cuda.mem_alloc(M_list.nbytes)
        cuda.memcpy_htod(d_M_list, M_list)
        
        h_Cv = numpy.array(h_Cv,numpy.int32)
        d_Cv = cuda.mem_alloc(h_Cv.nbytes)
        cuda.memcpy_htod(d_Cv, h_Cv)

        h_Sv = numpy.array(h_Sv,numpy.int32)
        d_Sv = cuda.mem_alloc(h_Sv.nbytes)
        cuda.memcpy_htod(d_Sv, h_Sv)

        h_St = numpy.array(h_St,numpy.int32)
        d_St = cuda.mem_alloc(h_St.nbytes)
        cuda.memcpy_htod(d_St, h_St)

        h_Lv = numpy.array(h_Lv,numpy.int32)
        d_Lv = cuda.mem_alloc(h_Lv.nbytes)
        cuda.memcpy_htod(d_Lv, h_Lv)

        h_Gv = numpy.array(h_Gv,numpy.int32)
        d_Gv = cuda.mem_alloc(h_Gv.nbytes)
        cuda.memcpy_htod(d_Gv, h_Gv)

        h_NG = numpy.array(h_NG,numpy.int32)
        d_NG = cuda.mem_alloc(h_NG.nbytes)
        cuda.memcpy_htod(d_NG, h_NG)

        h_rules = numpy.array(h_rules,numpy.int32)
        d_rules = cuda.mem_alloc(h_rules.nbytes)
        cuda.memcpy_htod(d_rules, h_rules)

        h_delays = numpy.array(h_delays,numpy.int32)
        d_delays = cuda.mem_alloc(h_delays.nbytes)
        cuda.memcpy_htod(d_delays, h_delays)

        h_lhs = numpy.array(h_lhs,numpy.int32)
        d_lhs = cuda.mem_alloc(h_lhs.nbytes)
        cuda.memcpy_htod(d_lhs, h_lhs)

        h_Iv = numpy.array(h_Iv,numpy.int32)
        d_Iv = cuda.mem_alloc(h_Iv.nbytes)
        cuda.memcpy_htod(d_Iv, h_Iv)

        h_TMv = numpy.array(h_TMv,numpy.int32)
        d_TMv = cuda.mem_alloc(h_TMv.nbytes)
        cuda.memcpy_htod(d_TMv, h_TMv)

        transfer_time += time() - transfer_time_start
        
        blockDimX = 2
        gridDimX = MAX_SIZE//2
        gridDimY = CASES*pop_size

        #print(h_in_spt)

        current_step = 0
        while current_step < max_spt + len(max(neurons, key=len)):
            # Get Output Spike train
            GetOutputSpt = mod.get_function("GetOutputSpt")
            GetOutputSpt(numpy.int32(CASES), d_out_idx, d_M_list, numpy.int32(current_step), d_out_spt, d_Cv, block=(blockDimX,1,1), grid=(gridDimX,gridDimY,1))

            # Determine Spiking Vector
            SNPDetermineRules = mod.get_function("SNPDetermineRules")
            SNPDetermineRules(numpy.int32(CASES), d_N_list, d_M_list, d_Cv, d_Sv, d_rules, d_lhs, block=(blockDimX,1,1), grid=(gridDimX,gridDimY,1))

            # Compute Next Step
            SNPSetStates = mod.get_function("SNPSetStates")
            SNPSetStates(numpy.int32(CASES), d_N_list, d_M_list, d_Cv, d_Sv, d_rules, d_delays, d_Lv, d_St, d_Iv, d_TMv, block=(blockDimX,1,1), grid=(gridDimX,gridDimY,1))

            SNPComputeNext = mod.get_function("SNPComputeNext")
            SNPComputeNext(numpy.int32(CASES), d_N_list, d_M_list, d_Cv, d_St, d_Lv, d_Iv, d_TMv, block=(blockDimX,1,1), grid=(gridDimX,gridDimY,1))

            # Counter Reduce
            SNPPostCompute = mod.get_function("SNPPostCompute")
            SNPPostCompute(numpy.int32(CASES), d_N_list, d_M_list, d_rules, d_Iv, d_TMv, d_Sv, block=(blockDimX,1,1), grid=(gridDimX,gridDimY,1))

            # Reset Values
            SNPReset = mod.get_function("SNPReset")
            SNPReset(numpy.int32(CASES), d_N_list, d_M_list, d_Lv, d_Gv, d_NG, block=(blockDimX,1,1), grid=(gridDimX,gridDimY,1))

            # Read Input Spike Train
            ReadInputSpt = mod.get_function("ReadInputSpt")
            ReadInputSpt(numpy.int32(CASES), numpy.int32(MAX_SIZE//2),d_M_list, d_in_spt, d_in_idx, d_Cv, numpy.int32(current_step), d_num_in_nrns, block=(blockDimX,1,1), grid=(gridDimX,gridDimY,1))

            current_step += 1

        transfer_time_start = time()
        h_out_spt = numpy.empty_like(h_out_spt)
        cuda.memcpy_dtoh(h_out_spt, d_out_spt)
        transfer_time += time() - transfer_time_start

        # generating list of dim [pop_size,cases] of string
        # spike trains of outputs per case per chrom
        out_spt = [[] for i in range(pop_size)]
        chrom_ctr = 0
        for chrom in h_out_spt:
            temp_case = []
            for case in chrom:
                temp_spt = ""
                for spike in case:
                    temp_spt += str(spike)
                temp_case.append(temp_spt)
            out_spt[chrom_ctr] = temp_case
            chrom_ctr += 1

        return out_spt, transfer_time

    def set_precision_fitness(self, population, gen=0):
        # Set precision of each chromosome in the population
        avg_precision = 0
        highest_precision = self.highest_precision
        transfer_time = 0

        # simulating, 2d out_spt [pop_size,cases]
        out_spt, gpu_delay = self.simulate(population=population)
        transfer_time += gpu_delay
        chrom_ctr = 0  # for traversing output spike trains per chrom
        for chrom in population:
            # traversing chroms of population
            precision = 0
            ctr = 0
            for case in fit_cases[self.fit_cases_key]:
                precision += self.calc_lcs_rate(out_spt[chrom_ctr][ctr], case['out'])
                ctr+=1
            precision /= len(fit_cases[self.fit_cases_key])
            avg_precision += precision

            chrom.set_precision(precision)

            if precision > highest_precision \
                    or (precision == highest_precision and chrom.get_size() < self.best_chrom.get_size()):
                highest_precision = precision
                self.best_chrom = chrom
                self.best_gen = gen

            chrom_ctr += 1

        if highest_precision > self.highest_precision:
            self.num_evol_leaps += 1
            self.highest_precision = highest_precision

        # Set fitness of each chromosome in the population
        avg_precision /= len(population)
        for chrom in population:
            fitness = chrom.get_precision() / avg_precision
            chrom.set_fitness(fitness)

        # If ff conditions are met, halt current outer_loop and 
        # start a new one with self.best_chrom as new self.orig_chrom
        if self.orig_chrom \
                and self.highest_precision >= self.orig_chrom.get_precision() \
                and len(self.best_chrom.get_active_nrns()) < len(self.orig_chrom.get_active_nrns()):
            self.halt_evolution = True
            self.start_new_outer_loop = True

        return transfer_time

    def do_select(self, population):
        # Remove lowest ranking chromosomes from population (David et al.)
        ranked_population = sorted(population, key=methodcaller('get_fitness'), reverse=True)

        return ranked_population[:math.ceil(self.population_size*self.selection_rate)]

class Encoder:
    """
    For translating chromosome to *.pli files and vice versa
    """
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

class GPU:
    """
    Houses GPU related tasks for Mutation and Crossover
    of the GA function
    """
    def __init__(self, mod, population_size, mutation_rate, 
        crossover_rate, chrom_init, allow_self_loops):
        self.module = mod
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.chrom_init = chrom_init
        self.num_nrns = chrom_init.get_num_nrns()
        self.chrom_size = chrom_init.get_num_nrns() **2
        self.allow_self_loops = allow_self_loops

    def get_block_grid(self, size):
        "For generating block and grid dimensions"
        block_dim_X = self.num_nrns
        block_dim_Y = 1
        grid_dim_X = self.num_nrns
        grid_dim_Y = size 
        return block_dim_X, block_dim_Y, grid_dim_X, grid_dim_Y

    def get_function(self, fxn):
        "Returns function to be run in device"
        return self.module.get_function(fxn)

    def get_population(self, population):
        "Generates population list of chroms from GPU"
        population_chroms = []

        for i in range(len(population)):
            new_chrom = self.chrom_init.get_copy()
            new_chrom.set_syn_matrix(syn_matrix=population[i].tolist())
            population_chroms.append(new_chrom)

        return population_chroms

    def get_syn_matrices(self, population):
        "Generates list of synapse matrices of population"
        syn_matrices = []

        for chrom in population:
            syn_matrices.append(chrom.get_syn_matrix())

        return syn_matrices

    def get_mutate_task(self, chromosome):
        "Generate task to do to mutate chromosome"
        task_order = [i for i in range(3)] # 3 for equal chances
        nrns = chromosome.get_connected_nrns()
        random.shuffle(task_order)
        success = False

        for i in task_order:
            if i == 0:
                # disconnect neuron
                random.shuffle(nrns)
                for n in nrns:
                    if n not in chromosome.get_input_nrns() and n not in chromosome.get_output_nrns():
                        if self.allow_self_loops:
                            # variant 2
                            nrn = n
                            syn = 0
                            success = True
                            return nrn, syn, success
                        else:
                            # variant 1
                            # Check that there is no intersection wiht pre-synapses
                            # and post-synapses to avoid a self loop
                            pre_syns = []
                            post_syns = []
                            syn_mtx = chromosome.get_syn_matrix()
                            for i in range(chromosome.get_num_nrns()):
                                if syn_mtx[i][n]:
                                    pre_syns.append(i)
                                if syn_mtx[n][i]:
                                    post_syns.append(i)
                            intersection = [x for x in pre_syns if x in post_syns]
                            if not intersection:
                                # No intersection therefor no self loops
                                # if neuron n is to be disconnected
                                nrn = n
                                syn = 0
                                success = True
                                return nrn, syn, success
            else:
                # add or delete synapse
                random.shuffle(nrns)
                for n in nrns:
                    if n not in chromosome.get_output_nrns():
                        nrn = -(n+1)
                        syn = numpy.random.randint(chromosome.get_num_nrns())
                        if not allow_self_loops:
                            # variant 1
                            while syn == n:
                                # For add synapse case:
                                # Adding this synapse has possibility of adding
                                # synapse to own neuron, which creates a self-loop
                                # For delete synapse case:
                                # Assuming no self loops, no instance of deleting
                                # a synapse pointed to self
                                syn = numpy.random.randint(chromosome.get_num_nrns())
                        success = True
                        return nrn, syn, success

        return nrn, syn, success

    def get_mutation_params(self, population):
        "Returns 4 arrays: (2) synapse matrices, tasks, and synapses"
        mutate_pop = []     # to be fed to the device, list of syn matrices
        notmutate_pop = []  # list of chroms not to be mutated
        tasks = []
        syns = []
        if len(population) == 1:
            while len(mutate_pop) < self.population_size:
                task, syn, success = self.get_mutate_task(chromosome=population[0])
                if success:
                    mutate_pop.append(population[0].get_syn_matrix())
                    tasks.append(task)
                    syns.append(syn)      
        else:
            # population already exists
            # Pseudorandomly mutate all chromosomes in current population
            for chrom in population:
                if random.random() < self.mutation_rate:
                    # try to mutate chromosome
                    task, syn, success = self.get_mutate_task(chromosome=chrom)
                    if success:
                        mutate_pop.append(chrom.get_syn_matrix())
                        tasks.append(task)
                        syns.append(syn)
                else:
                    # do not mutate chromosome
                    notmutate_pop.append(chrom.get_copy())

            # Add pseudorandom chromosomes to mutate to fill population size      
            nrn_order = [i for i in range(len(population))]
            random.shuffle(nrn_order)
            if (len(mutate_pop)+len(notmutate_pop)) < self.population_size:
                for i in nrn_order:
                    task, syn, success = self.get_mutate_task(population[i])
                    if success:
                        mutate_pop.append(population[i].get_syn_matrix())
                        tasks.append(task)
                        syns.append(syn)
                    if (len(mutate_pop)+len(notmutate_pop)) == self.population_size:
                        break

        return mutate_pop, notmutate_pop, tasks, syns

    def do_mutate(self, population):
        "Mutation operation"
        pop, prepend_pop, tasks, syns = gpu.get_mutation_params(population=population)
        transfer_time_start = 0    # for subtracting time to copy files from host-data v.v.
        transfer_time = 0
        
        if len(pop) > 0:
            # Calculate time to transfer data from host to device
            transfer_time_start = time()

            # There's something to mutate
            # Copying population from host to device
            pop = numpy.array(pop,numpy.int32)
            pop_gpu = cuda.mem_alloc(pop.nbytes)
            cuda.memcpy_htod(pop_gpu,pop)

            # Copying tasks from host to device
            tasks = numpy.array(tasks,numpy.int32)
            tasks_gpu = cuda.mem_alloc(tasks.nbytes)
            cuda.memcpy_htod(tasks_gpu,tasks)

            # Copying synapses from host to device
            syns = numpy.array(syns,numpy.int32)
            syns_gpu = cuda.mem_alloc(syns.nbytes)
            cuda.memcpy_htod(syns_gpu,syns)

            transfer_time += time() - transfer_time_start

            ##### MUTATION PROPER
            # For generating block and grid dimensions
            block_dim_X, block_dim_Y, grid_dim_X, grid_dim_Y = self.get_block_grid(size=len(pop))

            mutate = self.get_function("mutate")
            print("\tblock: ("+str(block_dim_X)+","+str(block_dim_Y)+"), grid: ("+str(grid_dim_X)+","+str(grid_dim_Y)+")")
            mutate(pop_gpu,tasks_gpu,syns_gpu,
                numpy.int32(self.chrom_init.get_num_nrns()),
                block=(block_dim_X,block_dim_Y,1),grid=(grid_dim_X,grid_dim_Y,1))

            # Clearing synapse values of disconnected neurons
            clear_disc = mod.get_function("clear_disc")
            clear_disc(pop_gpu,tasks_gpu,
                numpy.int32(self.chrom_init.get_num_nrns()),
                block=(block_dim_X,block_dim_Y,1),grid=(grid_dim_X,grid_dim_Y,1))

            # Calculate time to transfer data from device to host
            transfer_time_start = time()
            # Copying mutated population from device to host
            pop_out = numpy.empty_like(pop)
            cuda.memcpy_dtoh(pop_out, pop_gpu)
            transfer_time += time() - transfer_time_start
            
            if len(prepend_pop) > 0:
                # Prepend population not empty
                mutated_chroms = self.get_population(population=pop_out)
                for chrom in mutated_chroms:
                    prepend_pop.append(chrom)
                return prepend_pop,transfer_time
            
            return self.get_population(population=pop_out),transfer_time

        # Otherwise, nothing to mutate
        return population,transfer_time


    def get_swappable(self, population, length, crossover_rate=None):
        "Get neuron eligible for swapping for both parents"
        # population - list of chroms of current population
        # length - number of pairs needed
        # crossover_rate - check crossover_rate if exists
        swappable = []

        parent1_chroms = random.sample(population,len(population))
        parent2_chroms = random.sample(population,len(population))

        parent1 = [] # synapse matrices
        parent2 = [] # synapse matrices
        swap = True # relies on crossover_rate if exists
        nrn_exists = False # swappable neuron exists

        for i in range(len(population)):
            # Getting swappable neuron
            swap = True
            if crossover_rate:
                if random.random() >= crossover_rate:
                    swap = False

            if swap == True:
                temp = random.sample(parent1_chroms[i].get_swappable_nrns(),len(parent1_chroms[i].get_swappable_nrns()))
                nrn_exists = False
                for nrn in temp:
                    if nrn in parent2_chroms[i].get_swappable_nrns():
                        if parent1_chroms[i].get_syn_matrix()[nrn] != parent2_chroms[i].get_syn_matrix()[nrn]:
                            swappable.append(nrn)
                            nrn_exists = True
                            break
            else:
                # If crossover_rate exists, it means that curr_pop_size = param
                # therefore, we also add parents to be passed to GPU even when
                # crossover did not happen, which is denoted by -1 value
                swappable.append(-1)

            # Getting synapse matrices of swappable parents
            if nrn_exists or crossover_rate:
                # 1st conditon: for curr_pop < param, add parents to GPU because
                # swappable neuron exists
                # 2nd condition: for curr_pop = param, add parents to GPU
                # regardless if swappable neuron exists or not
                parent1.append(parent1_chroms[i].get_syn_matrix())
                parent2.append(parent2_chroms[i].get_syn_matrix())

            # If pairs reach number of length, halt
            if len(swappable) == length:
                break

        return parent1, parent2, swappable 

    def get_crossover_params(self, population):
        "Returns 3 arrays: base parent, copy parent, swap neuron"
        diff = self.population_size - len(population)
        diff2 = diff*2
        append_to_current = False
        # True - append values of children to current population
        # False - replace current population with children

        # Generating list of synapse matrices of current population
        # for pop in population:
        #     curr_pop.append(pop.get_syn_matrix())

        if diff > 0:
            # Population not filled up 
            parent1, parent2, swappable = self.get_swappable(population=population,
                length=diff)
            append_to_current = True
        else:
            # Population filled up
            parent1, parent2, swappable = self.get_swappable(population=population,
                length=self.population_size,
                crossover_rate=self.crossover_rate)
            append_to_current = False

        return parent1, parent2, swappable, append_to_current

    def generate_children_chroms(self, children1, children2, size):
        "Generate list of chroms of children"
        children_chroms = []

        for i in range(max(len(children1),len(children2))):
            new_chrom1 = self.chrom_init.get_copy()
            new_chrom1.set_syn_matrix(syn_matrix=children1[i].tolist())
            if new_chrom1.is_exists_path_to_output_nrn():
                children_chroms.append(new_chrom1)

            if len(children_chroms) == size:
                break

            new_chrom2 = self.chrom_init.get_copy()
            new_chrom2.set_syn_matrix(syn_matrix=children2[i].tolist())
            if new_chrom2.is_exists_path_to_output_nrn():
                children_chroms.append(new_chrom2)

            if len(children_chroms) == size:
                break

        return children_chroms

    def do_crossover(self, population):
        "Crossover operation"
        #### CROSSOVER PARAMETERS
        parents1, parents2, swap_nrns, append_to_current = gpu.get_crossover_params(population=population)
        crossover_size = len(swap_nrns)
        transfer_time_start = 0    # for subtracting time to copy files from host-data v.v.
        transfer_time = 0

        if len(parents1) != 0:
            # Calculate time to transfer data from host to device
            transfer_time_start = time()
            # If len of parents 1 != 0, parents2 and swap_nrns also are
            # Otherwise, just return same population because crossover 
            # did not happen
            parents1 = numpy.array(parents1,numpy.int32)
            parents1_gpu = cuda.mem_alloc(parents1.nbytes)
            cuda.memcpy_htod(parents1_gpu,parents1)

            parents2 = numpy.array(parents2,numpy.int32)
            parents2_gpu = cuda.mem_alloc(parents2.nbytes)
            cuda.memcpy_htod(parents2_gpu,parents2)

            swap_nrns = numpy.array(swap_nrns,numpy.int32)
            swap_nrns_gpu = cuda.mem_alloc(swap_nrns.nbytes)
            cuda.memcpy_htod(swap_nrns_gpu,swap_nrns)

            children1 = numpy.zeros(parents1.shape, numpy.int32)
            children1_gpu = cuda.mem_alloc(children1.nbytes)
            cuda.memcpy_htod(children1_gpu, children1)

            children2 = numpy.zeros(parents2.shape, numpy.int32)
            children2_gpu = cuda.mem_alloc(children2.nbytes)
            cuda.memcpy_htod(children2_gpu, children2)

            transfer_time += time() - transfer_time_start

            #### CROSSOVER PROPER
            # For generating block and grid dimensions
            block_dim_X, block_dim_Y, grid_dim_X, grid_dim_Y = gpu.get_block_grid(size=crossover_size)

            crossover = self.get_function("crossover")
            print("\tblock: ("+str(block_dim_X)+","+str(block_dim_Y)+"), grid: ("+str(grid_dim_X)+","+str(grid_dim_Y)+")")
            crossover(parents1_gpu, parents2_gpu, swap_nrns_gpu, 
                children1_gpu, children2_gpu, numpy.int32(self.chrom_init.get_num_nrns()),
                block=(block_dim_X,block_dim_Y,1), grid=(grid_dim_X,grid_dim_Y,1))

            # Calculate time to transfer data from device to host
            transfer_time_start = time()
            # Copying children from device to host
            children1_out = numpy.empty_like(children1)
            cuda.memcpy_dtoh(children1_out, children1_gpu)

            children2_out = numpy.empty_like(children2)
            cuda.memcpy_dtoh(children2_out, children2_gpu)
            transfer_time += time() - transfer_time_start

            # Generating list of chroms of children
            children_chroms = gpu.generate_children_chroms(children1=children1_out, 
                    children2=children2_out, size=(population_size-len(population)))
            if append_to_current:
                population.extend(children_chroms)
            else:
                population = children_chroms

        return population, transfer_time

# this paramater contains the kernel functions called
# that are executed on the device
mod = SourceModule("""
    #include <stdio.h>
    #define _SYN (threadIdx.x)
    #define _NRN (blockIdx.x)
    #define _CHROM (blockIdx.y)

    #define _X (threadIdx.x + blockIdx.x * blockDim.x)
    #define _Y (threadIdx.y + blockIdx.y * blockDim.y)
    #define _WIDTH (blockDim.x * gridDim.x)

    #define _INDEX(x,y) (x + y * _WIDTH)
    #define _TRAVERSE(s,n) ((s + n * blockDim.x)+_Y* _WIDTH)
    #define _XTRAVERSE(t) (t+_Y* _WIDTH)

    
    __global__ void mutate(int *pop, int *tasks, int *syns, int nrns) {
        if (tasks[_CHROM] >= 0) {
            // Disconnect Neuron
            if (tasks[_CHROM] != _NRN && _SYN == tasks[_CHROM]) {
                // In possible pre-synapse
                // 1st cond: not chosen neuron
                // 2nd cond: current synapse is pointing to diconnected neuron
                if (pop[_INDEX(_X,_Y)] == 1) {
                    // There exists a synapse to the disconnected neuron, remove synapse
                    pop[_INDEX(_X,_Y)] = 0;
                    // Copy post-synapses of disconnected neuron to current neuron
                    for (int i=0; i<nrns; i++) {
                        if (tasks[_CHROM]!=i && pop[_TRAVERSE(i,tasks[_CHROM])] == 1) {
                            // 1st cond: ensures that it doesn't reconnect neuron, in the even that the neuron to be disconnected has self loop
                            // 2nd cond: There exists a post-synapse to be copied
                            pop[_TRAVERSE(i,_NRN)] = 1;
                        }
                    }
                }
            }
        } else {
            // Delete or add synapse
            int chosen_nrn = (tasks[_CHROM] + 1) * -1;
            if (_NRN == chosen_nrn && _SYN == syns[_CHROM]) {
                // 1st cond: in chosen neuron
                // 2nd cond: current synapse is synapse to act on
                if (pop[_INDEX(_X,_Y)] == 1) {
                    // Delete synapse
                    pop[_INDEX(_X,_Y)] = 0;
                } else {
                    // Add synapse
                    pop[_INDEX(_X,_Y)] = 1;
                }
            }
        }
    }
    __global__ void clear_disc(int *pop, int *tasks, int nrns) {
        if (tasks[_CHROM] >= 0) {
            if (_NRN == tasks[_CHROM]) {
                pop[_INDEX(_X,_Y)] = 0;
            }
        }
    }
    __global__ void crossover(int *parent1, int *parent2, int *swap, int *child1, int *child2, int nrns) {
        if (swap[_CHROM] == _NRN) {
            // Swap post-synapses
            child1[_INDEX(_X,_Y)] = parent2[_INDEX(_X,_Y)];
            child2[_INDEX(_X,_Y)] = parent1[_INDEX(_X,_Y)];
        } else {
            // No swapping, just copy from its parent
            child1[_INDEX(_X,_Y)] = parent1[_INDEX(_X,_Y)];
            child2[_INDEX(_X,_Y)] = parent2[_INDEX(_X,_Y)];
        }
    }

    __global__ void SNPDetermineRules(int num_cases, int* N, int* M, int* Cv, int* Sv, int* rules, int* lhs){//, int * ncount, int * scount){
        // modifies SPIKING VECTOR, 1 if rule is applicable and 0 if otherwise
        int i = _X;
        int curr_pop = _CHROM/num_cases;

        if(i < N[curr_pop]){
            if( (Cv[_XTRAVERSE((int)rules[_XTRAVERSE(i*3)])] == lhs[_INDEX(_X,_Y)] || lhs[_INDEX(_X,_Y)] == 0 ) && Cv[_XTRAVERSE((int)rules[_XTRAVERSE(i*3)])] + rules[_XTRAVERSE(i*3+2)] >= 0 ){
                Sv[_INDEX(_X,_Y)] = 1;
            }
            else{
                Sv[_INDEX(_X,_Y)] = 0;
            }
        }
    }

    __global__ void SNPSetStates(int num_cases, int* N, int* M, int* Cv, int* Sv, int* rules, int* delays, int* Lv, int* St, int* Iv, int* TMv){
        int i = _X;
        int curr_pop = _CHROM/num_cases;

        // i denotes rule in question

        if(i < N[curr_pop]){
        // check with SPIKING VECTOR if rule is applicable
            if(Sv[_INDEX(_X,_Y)] == 1){
                // logging spikes consumed in LOSS VECTOR
                Lv[ _XTRAVERSE((int)rules[_XTRAVERSE(i*3)]) ] = rules[_XTRAVERSE(i*3+2)];
                // logging in delay
                rules[_XTRAVERSE(i*3+1)] = delays[_INDEX(_X,_Y)];
                // closes neuron via STATUS VECTOR
                St[ _XTRAVERSE((int)rules[_XTRAVERSE(i*3)]) ] = 0;
                // check if rule has delay, run if no delay
                if(delays[_INDEX(_X,_Y)] == 0){
                    // rule is set to produce spike via INDICATOR VECTOR
                    Iv[_INDEX(_X,_Y)] = 1;
                    // opens neuron via STATUS VECTOR
                    St[_XTRAVERSE((int)rules[_XTRAVERSE(i*3)])] = 1;
                }
            }
        // check if RULE REPRESENTATION has no delay
            else if(rules[_XTRAVERSE(i*3+1)] == 0){
                Iv[_INDEX(_X,_Y)] = 1;
                St[_XTRAVERSE((int)rules[_XTRAVERSE(i*3)])] = 1;
            }
        }
    }

    __global__ void SNPComputeNext(int num_cases, int* N, int* M, int* C, int* St, int* Lv, int* Iv, int* Tv){
        int i = _X;
        int curr_pop = _CHROM/num_cases;

        int gv_i;   // GAIN VECTOR
        int ng;     // NET GAIN

        if(i < M[curr_pop]){
            gv_i = 0;
            for(int z = 0; z < N[curr_pop]; z++){
                gv_i += Iv[_XTRAVERSE(z)] * Tv[_XTRAVERSE(z * M[curr_pop] + i)]; 
            }
            ng = St[_INDEX(_X,_Y)] * gv_i + Lv[_INDEX(_X,_Y)];
            C[_INDEX(_X,_Y)] += ng;
        }
    }

    __global__ void SNPPostCompute(int num_cases, int* N, int* M, int* rules, int* Iv, int* TMv, int* Sv){
        int i = _X;
        int curr_pop = _CHROM/num_cases;

        if( i < N[curr_pop] ){
            if(rules[_XTRAVERSE(i*3+1)] > -1) rules[_XTRAVERSE(i*3+1)] -= 1;
            Iv[_INDEX(_X,_Y)] = 0;
            Sv[_INDEX(_X,_Y)] = 0;
        }
    }

    __global__ void SNPReset(int num_cases, int* N, int* M, int* Lv, int* Gv, int* NG){
        int i = _X;
        int curr_pop = _CHROM/num_cases;

        if( i < M[curr_pop] ){
            Lv[_INDEX(_X,_Y)] = 0;
            Gv[_INDEX(_X,_Y)] = 0;
            NG[_INDEX(_X,_Y)] = 0;
        }
    }

    __global__ void ReadInputSpt(int num_cases, int max_spt, int* M, int* in_spt, int* in_idx, int* C, int current_step, int* num_in_nrns){
        int i = _X;
        int curr_pop = _CHROM/num_cases;

        if( i < M[curr_pop] ){
            for(int z = 0; z < num_in_nrns[curr_pop]; z++){
                if( i == in_idx[_XTRAVERSE(z)] && current_step < max_spt){
                    C[_INDEX(_X,_Y)] += in_spt[_XTRAVERSE(z * max_spt + current_step)];
                }
            }
        }
    }

    __global__ void GetOutputSpt(int num_cases, int* out_idx, int* M, int current_step, int* out_spt, int* C){
        int i = _X;
        int curr_pop = _CHROM/num_cases;

        if( i < M[curr_pop] ){
            if( i == out_idx[curr_pop]){
                out_spt[_XTRAVERSE(current_step)] = C[_INDEX(_X,_Y)];
                C[_INDEX(_X,_Y)] = 0;
            }
        }
    }

""")

def write_to_file(file, string):
    "Appends string to file"
    f = open(file, 'a')
    f.write(string + "\n")
    f.close()
    print(string)

if __name__ == "__main__":
    ##### Default Params
    num_runs = 5
    num_inner_loops = 10
    max_gen = 75
    population_size = 80

    mutation_rate = 0.5
    crossover_rate = 0.3
    selection_rate = 0.6

    precision_threshold = 0

    same_init_pop = True
    allow_self_loops = False

    # Parsing arguments
    try:
        opts, args = getopt(sys.argv[1:], 'hd:f:c:r:p:i:s:')
    except GetoptError:
        print("Error! Use format: ga_framework.py -f <filename> -d init_pli/<directory-of-pli-file> -c "
              "<fit_cases key> [-r <num_runs, int, default: 5>, -p <population_size, int, default: 80>,"
              "-i to generate init pop, -s to allow self loops]")
        sys.exit(2)

    if not opts:
        print("Error! Use format: ga_framework.py -f <filename> -d init_pli/<directory-of-pli-file> -c "
              "<fit_cases key> [-r <num_runs, int, default: 5>, -p <population_size, int, default: 80>,"
              "-i to generate init pop, -s to allow self loops]")
        sys.exit(2)

    init_pli_file, init_pli_dir, fit_cases_key = None, None, None
    for opt, arg in opts:
        if opt == '-h':
            print("ga_framework.py -f <filename> -d init_pli/<directory-of-pli-file> -c "
                  "<fit_cases key> [-r <num_runs, int, default: 5>, -p <population_size, int, default: 80>,"
                  "-i to generate init pop, -s to allow self loops]")
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
        elif opt == '-s':
            allow_self_loops = True

    # Parsing input
    parsed = parse(filename=init_pli_dir + '/' + init_pli_file + '.pli')

    # Creating directory
    if allow_self_loops:
        base_output_dir = "outputs-gpu2_var3/"
    else:
        base_output_dir = "outputs-gpu2_var1/"
    if not os.path.isdir(base_output_dir):
        os.mkdir(base_output_dir)
    base_output_dir += "outputs-gpu2_" + init_pli_file + "/"
    if not os.path.isdir(base_output_dir):
        os.mkdir(base_output_dir)

    # Creating output directory
    runs_length = len(str(num_runs))    # for prepending 0's when >9 runs
    total_time = 0                      # total runtime of command
    total_delay = 0                     # total delay from copying data between host/device
    for run in range(num_runs):
        # Prepending 0's to num of runs in directory
        prepend_zeros = ''
        for _ in range(runs_length - len(str(run))):
            prepend_zeros += '0'
        output_dir = base_output_dir + '/' + init_pli_file + '-' +\
            prepend_zeros + str(run+1) + '/'
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        valid = "active"

        inner_loop = 0
        outer_loop = -1
        selection = None
        outer_start_time = time()
        outer_delay = 0        # total delay of GPU in outer loop

        ################################
        #       OUTER LOOP START
        ################################
        overall_best_chrom = None
        start_new_outer_loop = True
        while start_new_outer_loop:
            outer_loop += 1
            # Check if chrom_fin exists
            if overall_best_chrom:
                # Set chrom_fin as new chrom_init
                # TODO: Delete inactive neurons
                chrom, deleted_nrns = overall_best_chrom.delete_invalid_nrns(valid=valid)
                while deleted_nrns:
                    chrom, deleted_nrns = chrom.delete_invalid_nrns(valid=valid)
            else:
                # Initial run, use initial input
                chrom = Encoder.parsed_pli_to_chrom(parsed)

            start_new_outer_loop = False

            # Getting precision fitness of chrom_init
            orig_chrom = chrom.get_copy()
            selection = Selection(population_size=population_size,
                    fit_cases_key=fit_cases_key, selection_rate=selection_rate)
            outer_delay += selection.set_precision_fitness([orig_chrom])

            # Create outer loop directory
            outer_loop_dir = output_dir + '/' + init_pli_file +\
                '-O-' + datetime.now().strftime("%y%m%d-%H%M%S-%f") + '/'
            os.mkdir(outer_loop_dir)

            if outer_loop == 0:
                # Saving original chrom_init in output directory
                Encoder.chrom_to_pli(chrom=orig_chrom, directory=output_dir,
                    filename=init_pli_file, extension="init")
                orig_chrom.save_graph(name=init_pli_file + '-init',
                    directory=output_dir)
            # Saving chrom_init in outer loop directory
            Encoder.chrom_to_pli(chrom=orig_chrom, directory=outer_loop_dir,
                filename=init_pli_file, extension="init")
            orig_chrom.save_graph(name=init_pli_file + '-init',
                directory=outer_loop_dir) 

            # WRITING AND SAVING _info FILE FOR GA RUN PARAMS
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

            # Performance evaluation measures from Casauay
            # Performance evaluation measures
            Lopt = f_bar = Lel = 0
            G_best = {}  # Self-created??? G_best(k) is the likelihood that the best chrom is from the kth generation
            Lhalt = {}   # Self-created??? Lhalt(k) is the likelihood of halting at the kth generation in n inner_loops

            ################################
            #        INNER LOOP
            ################################
            for inner_loop in range(num_inner_loops):
                inner_delay = 0
                # for operation specific time
                gen_op_stats = "= Gen 0" + " - "
                gen_op_start = 0

                # Create inner loop directory
                inner_loop_dir = outer_loop_dir + init_pli_file + "-I-" +\
                            datetime.now().strftime("%y%m%d-%H%M%S-%f") + "/"
                os.mkdir(inner_loop_dir)

                # Initializing classes
                gpu = GPU(mod=mod, population_size=population_size,mutation_rate=mutation_rate, 
                    crossover_rate=crossover_rate, chrom_init=chrom, allow_self_loops=allow_self_loops)
                selection = Selection(population_size=population_size, 
                    fit_cases_key=fit_cases_key, selection_rate=selection_rate, 
                    orig_chrom=orig_chrom)

                # Generate initial population
                if same_init_pop == True and outer_loop == 0 and inner_loop == 0:
                    print("Getting initial population...")
                    population = []
                    for i in init_population[init_pli_file][0]:
                        population.append(Encoder.parsed_pli_to_chrom(i[0]))
                    inner_delay += init_population[init_pli_file][2]
                    gen_op_stats += "M: "+str(init_population[init_pli_file][1]-init_population[init_pli_file][2])+"; "
                else:
                    try:
                        print("Mutation process...")
                        gen_op_start = time()
                        population, gpu_delay = gpu.do_mutate(population=[orig_chrom])
                        inner_delay += gpu_delay
                        gen_op_stats += "M: "+str(time()-gen_op_start-gpu_delay)+"; "
                    except ValueError:
                        f = open(outer_loop_dir + '_info.py', 'a')
                        f.write('\n\n"""')
                        f.write("\n    Can't mutate original chromosome.")
                        f.write('\n"""\n')
                        f.close()
                        raise

                ################################
                #  START OF EVOLUTION PROCESS
                ################################
                gen = 0
                start_time = time()
                do_print = False

                for gen in range(max_gen):
                    # Create GA loop directory
                    ga_loop_dir = inner_loop_dir + "generation-" +\
                                    str(gen+1) + "/"
                    os.mkdir(ga_loop_dir)

                    # Making chrom files
                    for chrom in population:
                        extension = datetime.now().strftime("%y%m%d-%H%M%S-%f")
                        Encoder.chrom_to_pli(chrom=chrom, gen=gen+1, 
                            directory=ga_loop_dir, filename=init_pli_file,
                            extension=extension)
                        chrom.save_graph(name=init_pli_file + '-' + extension, 
                            directory=ga_loop_dir)

                    print("Set fitness...")
                    gen_op_start = time()
                    # Calculate precision and fitness
                    gpu_delay = selection.set_precision_fitness(population,gen+1)
                    inner_delay += gpu_delay
                    gen_op_stats += "SIM: "+str(time()-gen_op_start-gpu_delay)+"; "

                    print("Check for convergence...")
                    # Check for convergence, halt if max_gen is reached
                    if selection.is_halt_evolution() or gen+1 == max_gen:
                        print("Generation " + str(gen+1) + " halted")
                        # print(len(population))
                        write_to_file(base_output_dir + "/" + init_pli_file + ".out",
                            gen_op_stats+"halt")
                        break
                    # print(len(population))

                    print("Selection process...")
                    gen_op_start = time()
                    population = selection.do_select(population)
                    # print(len(population))
                    gen_op_stats += "SEL: "+str(time()-gen_op_start)+"; "

                    print("Crossover process...")
                    gen_op_start = time()
                    population, gpu_delay = gpu.do_crossover(population=population)
                    inner_delay += gpu_delay
                    gen_op_stats += "C: "+str(time()-gen_op_start-gpu_delay)

                    write_to_file(base_output_dir + "/" + init_pli_file + ".out",
                        gen_op_stats)

                    gen_op_stats = "= Gen "+str(gen+1)+ " - "
                    print("Mutation process...")
                    gen_op_start = time()
                    population, gpu_delay = gpu.do_mutate(population=population)
                    inner_delay += gpu_delay
                    gen_op_stats += "M: "+str(time()-gen_op_start-gpu_delay)+"; "
                    if len(population) < population_size:
                        do_print = True

                ##### END OF EVOLUTION PROCESS #####
                inner_total_time = time()-start_time
                outer_delay += inner_delay
                write_to_file(base_output_dir + "/" + init_pli_file + ".out",
                    "Outer loop " + str(outer_loop+1) + " - Inner loop " + 
                    str(inner_loop+1) + " - Generation " + str(gen+1) + " - Total: " +
                    str(inner_total_time - inner_delay) + "s, Raw: " +
                    str(inner_total_time) + "s, Delay: " +
                    str(inner_delay) + "s")
                if do_print:
                    write_to_file(base_output_dir + "/" + init_pli_file + ".out",
                             "len(population) < population_size from mutation")

                Encoder.chrom_to_pli(chrom=selection.get_best_chrom(),
                    gen=selection.get_best_chrom(),
                    num_evol_leaps=selection.get_num_evol_leaps(),
                    runtime=time()-start_time,
                    directory=inner_loop_dir, filename=init_pli_file,
                    extension="final")
                selection.get_best_chrom().save_graph(name=init_pli_file+'-final',
                    directory=inner_loop_dir)

                ################################
                #   PERFORMANCE EVALUATION
                ################################
                # Taken from Casauay et al.

                avg_runtime += time() - start_time - inner_delay
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

            Encoder.chrom_to_pli(chrom=overall_best_chrom, 
                gen=selection.get_best_gen(),
                num_evol_leaps=selection.get_num_evol_leaps(),
                directory=outer_loop_dir, filename=init_pli_file, 
                extension="final")
            overall_best_chrom.save_graph(name=init_pli_file + '-final',
                directory=outer_loop_dir)

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
        total_delay += outer_delay
        write_to_file(base_output_dir + "/" + init_pli_file + ".out",
                 "Run #" + str(run) + " total time: " + str(outer_loop_time - outer_delay) +
                 "s, raw: " + str(outer_loop_time) + "s, delay: " + str(outer_delay) + "s")

    write_to_file(base_output_dir + "/" + init_pli_file + ".out", "Total time: " + str(total_time - total_delay) +
        "s\nRaw: " + str(total_time) + "s\nDelay: " + str(total_delay))

print("Done!")
