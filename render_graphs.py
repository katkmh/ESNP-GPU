import os
import sys
from getopt import getopt, GetoptError
from pli_parser import parse
from ga_framework import Encoder


# Returns index of first and last letter
def find_substr(str, substr):
    if substr not in str:
        return -1, -1

    substr_index = 0
    first_index = -1
    for i in range(len(str)):
        if substr_index == len(substr):
            return first_index, i - 1
        elif str[i] == substr[substr_index]:
            substr_index += 1
        else:
            substr_index = 0
            first_index = -1


if __name__ == "__main__":
    try:
        opts, args = getopt(sys.argv[1:], 'hd:')
    except GetoptError:
        print("Error! Use format: render_graphs.py -d <directory-of-pli-files>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("render_graphs.py -d <directory-of-pli-files>")
        elif opt == '-d':
            directory = arg
            curr_dir = ''
            for root, dirs, files in os.walk(directory + '/'):
                for file in files:
                    if file.endswith(".pli"):
                        parsed_pli = parse(filename=root + '/' + file)
                        chrom = Encoder.parsed_pli_to_chrom(parsed_pli=parsed_pli)
                        chrom.save_graph(name=file[:-4], directory=root)


