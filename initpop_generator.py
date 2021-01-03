#######################################################
# Program for generating initial population for
# GA Framework
#
# Returns filename 'initial_population.py' containing
# dictionary of format:
#['operation-category': [[population],delay]]
#    - where operation-category is taken from available
#        choices,
#    - population is the generated initial population
#        by mutating the corresponding init System
#    - delay is the time the program took to copy data 
#        from device-host and v.v.
#
# Last modified by Rogelio Gungon and Katreen Hernandez
# on January 3, 2021
########################################################

from random import randint
from pli_parser import parse
import math, random
from time import time
import numpy
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import driver, compiler, gpuarray, tools

class Chromosome:
    # Represents an SNP System recognizable by the GA Function
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
class GPU:
    """
    Houses GPU related tasks for Mutation and Crossover
    of the GA function
    """
    def __init__(self, mod, population_size, mutation_rate, 
        crossover_rate, chrom_init):
        self.module = mod
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.chrom_init = chrom_init
        self.num_nrns = chrom_init.get_num_nrns()
        self.chrom_size = chrom_init.get_num_nrns() **2

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
        "Do mutate"
        pop, prepend_pop, tasks, syns = gpu.get_mutation_params(population=population)
        transfer_time_start = 0    # for subtracting time to copy files from host-data v.v.
        transfer_time = 0
        print("Mutated chroms: "+str(len(pop)))
        print("Non-mutated chroms: "+str(len(prepend_pop)))
        print("TASKS")
        print(tasks)
        print("SYNS")
        print(syns)
        
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
    #define _XT(t) (t + + blockIdx.x * blockDim.x)

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
""")
# population
population = {}

# read orig chroms per input type
chroms = {}
population_size = {}

chroms['add-adv'], population_size['add-adv'] = Encoder.parsed_pli_to_chrom(parse(filename='init_pli/adv_pli/add-adv.pli')), 40
chroms['sub-adv'], population_size['sub-adv'] = Encoder.parsed_pli_to_chrom(parse(filename='init_pli/adv_pli/sub-adv.pli')), 40
chroms['add-lg'], population_size['add-lg'] = Encoder.parsed_pli_to_chrom(parse(filename='init_pli/lg_pli/add-lg.pli')), 40
chroms['sub1-lg'], population_size['sub1-lg'] = Encoder.parsed_pli_to_chrom(parse(filename='init_pli/lg_pli/sub1-lg.pli')), 80
chroms['sub3-lg'], population_size['sub3-lg'] = Encoder.parsed_pli_to_chrom(parse(filename='init_pli/lg_pli/sub3-lg.pli')), 80
chroms['add-sm'], population_size['add-sm'] = Encoder.parsed_pli_to_chrom(parse(filename='init_pli/sm_pli/add-sm.pli')), 15
chroms['sub-sm'], population_size['sub-sm'] = Encoder.parsed_pli_to_chrom(parse(filename='init_pli/sm_pli/sub-sm.pli')), 20

mutation_rate = 0.5
crossover_rate = 0.3
selection_rate = 0.6

start_time = 0
run_time = 0

# generating initial population
for key in chroms:
    gpu = GPU(mod=mod, 
        population_size=population_size[key],
        mutation_rate=mutation_rate, 
        crossover_rate=crossover_rate,
        chrom_init=chroms[key])

    print("Generating initial population for "+key+"...")
    start_time = time()
    pop, gpu_delay = gpu.do_mutate(population=[chroms[key]])
    run_time = time() - start_time
    temp = []
    for chrom in pop:
        temp.append([Encoder.chrom_to_parsed_pli(chrom)])
    population[key] = [temp,run_time,gpu_delay]


f = open('initial_population.py','w')
f.write("init_population = "+str(population)+"\n")
f.close()
