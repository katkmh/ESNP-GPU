from getopt import getopt, GetoptError
import sys
import os

def get_number_str(str, to_int=0):
    number = ''
    for char in str:
        if char.isdigit() or char == '.':
            number += char
        else:
            break

    if to_int:
        return float(number)

    return number


if __name__ == "__main__":
    try:
        opts, args = getopt(sys.argv[1:], 'd:')
    except GetoptError:
        print("Error")
        sys.exit(2)

    if not opts:
        print("Error")
        sys.exit(2)

    directory = ''
    for opt, arg in opts:
        if opt == '-d':
            directory = arg

    f = open('results.py', 'a')
    f.write(directory + ' = ')
    ret_val = {}

    for directory1 in os.listdir(directory):
        # directory1: outputs_add-adv, outputs_add-lg, outputs_add-sm, etc.
        full_dir = os.path.join(directory, directory1)
        if os.path.isdir(full_dir):
            ret_val[directory1] = {}
            total_precision = 0
            total_overall_size = 0
            total_num_nrns = 0
            total_num_init_spikes = 0
            total_num_rules = 0
            total_num_synapses = 0
            for item in os.listdir(full_dir):
                if os.path.isfile(os.path.join(full_dir, item)):
                    f1 = open(os.path.join(full_dir, item), 'r')

                    prev_outer_loop = 1.0
                    prev_inner_loop = 1.0
                    total_num_generations = 0.0
                    total_runtime = 0.0

                    total_inner_loops = 0.0
                    total_ave_num_generations = 0.0
                    total_ave_runtime = 0.0

                    total_outer_loops = 0.0
                    total_ave_inner_loops = 0.0
                    ta_total_ave_num_generations = 0.0
                    ta_total_ave_runtime = 0.0
                    for line in f1.readlines():
                        if "Outer loop " in line:
                            outer_loop_index = len("Outer loop ")
                            outer_loop = get_number_str(line[outer_loop_index:])
                            inner_loop_index = outer_loop_index +  len(outer_loop) + len(" - Inner loop ")
                            inner_loop = get_number_str(line[inner_loop_index:])
                            generations_index = inner_loop_index + len(inner_loop) + len(" - Generation ")
                            num_generations = get_number_str(line[generations_index:])
                            runtime_index = generations_index + len(num_generations) + len(" - ")
                            runtime = get_number_str(line[runtime_index:])

                            outer_loop = float(outer_loop)
                            inner_loop = float(inner_loop)
                            num_generations = float(num_generations)
                            # print(num_generations)
                            runtime = float(runtime)

                            if outer_loop == prev_outer_loop:
                                total_num_generations += num_generations
                                total_runtime += runtime
                                prev_inner_loop = inner_loop
                            else:
                                total_inner_loops += prev_inner_loop
                                total_ave_num_generations += total_num_generations / prev_inner_loop
                                # print(total_num_generations / prev_inner_loop)
                                total_ave_runtime += total_runtime / prev_inner_loop
                                prev_outer_loop = outer_loop
                                prev_inner_loop = 1.0
                                total_num_generations = num_generations
                                total_runtime = runtime
                        elif "Run" in line:
                            total_inner_loops += prev_inner_loop
                            total_ave_num_generations += total_num_generations / prev_inner_loop
                            # print(total_ave_num_generations)
                            total_ave_runtime += total_runtime / prev_inner_loop

                            total_outer_loops += prev_outer_loop
                            total_ave_inner_loops += total_inner_loops / prev_outer_loop
                            ta_total_ave_runtime += total_ave_runtime / prev_outer_loop
                            ta_total_ave_num_generations += total_ave_num_generations / prev_outer_loop

                            prev_outer_loop = 1.0
                            prev_inner_loop = 1.0
                            total_num_generations = 0.0
                            total_runtime = 0.0

                            total_inner_loops = 0.0
                            total_ave_num_generations = 0.0
                            total_ave_runtime = 0.0

                    ave_total_outer_loops = total_outer_loops / 5.0
                    ave_ta_inner_loops = total_ave_inner_loops / 5.0
                    ave_ta_ta_runtime = ta_total_ave_runtime / 5.0
                    ave_ta_ta_num_generations = ta_total_ave_num_generations / 5.0

                    ret_val[directory1]['ave_outer_loops'] = ave_total_outer_loops
                    ret_val[directory1]['ave_inner_loops'] = ave_ta_inner_loops
                    ret_val[directory1]['ave_runtime'] = ave_ta_ta_runtime 
                    ret_val[directory1]['ave_num_generations'] = ave_ta_ta_num_generations

                    f1.close()

                elif os.path.isdir(os.path.join(full_dir, item)):
                    full_dir1 = os.path.join(full_dir, item)
                    for item1 in os.listdir(os.path.join(full_dir1)):
                        if '-final.pli' in item1:
                            f1 = open(os.path.join(full_dir1, item1), 'r')

                            for line in f1.readlines():
                                if 'precision' in line:
                                    index = len('precision: ')
                                    total_precision += get_number_str(line[index:], 1)
                                elif 'overall_size' in line:
                                    index = len('overall_size = ')
                                    total_overall_size += get_number_str(line[index:], 1)
                                elif 'num_nrns' in line:
                                    index = len('num_nrns = ')
                                    total_num_nrns += get_number_str(line[index:], 1)
                                elif 'num_init_spikes' in line:
                                    index = len('num_init_spikes = ')
                                    total_num_init_spikes += get_number_str(line[index:], 1)
                                elif 'num_rules' in line:
                                    index = len('num_rules = ')
                                    total_num_rules += get_number_str(line[index:], 1)
                                elif 'num_synapses' in line:
                                    index = len('num_synapses = ')
                                    total_num_synapses += get_number_str(line[index:], 1)

                            f1.close()

                    ret_val[directory1]['ave_precision'] = total_precision / 5
                    ret_val[directory1]['ave_overall_size'] = total_overall_size / 5
                    ret_val[directory1]['ave_num_nrns'] = total_num_nrns / 5
                    ret_val[directory1]['ave_num_init_spikes'] = total_num_init_spikes / 5
                    ret_val[directory1]['ave_num_rules'] = total_num_rules / 5
                    ret_val[directory1]['ave_synapses'] = total_num_synapses / 5


    f.write(str(ret_val) + '\n')
    f.close()
