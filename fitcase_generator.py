from random import randint
from fitness_cases import fit_cases

rng = 2
pli_files = ['sub']
# pli_files = ['add', 'and', 'or', 'xor']

# rng = 1
# pli_files = ['not']


fit_cases_add = {}
case_max_len = 50   # 50
num_cases = 100      # 100
"""
    x = {'in': ['0101', '0011'], 'out': '0000'}
    y = {'in': ['0101', '0011'], 'out': '0000'}
    z = {'in': ['0101', '0011'], 'out': '0000'}
    fit_cases_add = {'and': [x, y, z]}
"""

for pli in pli_files:
    fit_cases_add[pli] = []
    print("Generating fitness cases for " + pli + "...")
    for i in range(num_cases):
        next_case = False
        in_ = []
        while not next_case:
            in_ = []
            for j in range(rng):
                case_len = randint(1, case_max_len)
                case_in = ''
                for k in range(case_len):
                    case_in += str(randint(0, 1))
                in_.append(case_in)

            if pli == 'sub' and int(in_[0][::-1], 2) - int(in_[1][::-1], 2) < 0:
                # Reverse strings first with [::-1] above
                # in[0] must always be greater than or equal to in[1]
                temp = in_[0]
                in_[0] = in_[1]
                in_[1] = temp

            # Check if case already exists
            next_case = True
            for case in fit_cases_add[pli]:
                if set(in_) == set(case['in']):
                    next_case = False
                    break

        out_spt = ''
        if pli == 'add':
            out_spt = int(in_[0][::-1], 2) + int(in_[1][::-1], 2)
            out_spt = bin(out_spt)[2:][::-1]
        elif pli == 'sub':
            out_spt = int(in_[0][::-1], 2) - int(in_[1][::-1], 2)
            out_spt = bin(out_spt)[2:][::-1]
        elif pli == 'and':
            max_len = len(in_[0])
            if len(in_[1]) > max_len:
                max_len = len(in_[1])
            for j in range(max_len):
                bit = 0
                try:
                    if in_[0][j] and in_[1][j]:
                        bit = 1
                except IndexError:
                    pass    # bit still 0

                out_spt += bit
        elif pli == 'or':
            max_len = len(in_[0])
            if len(in_[1]) > max_len:
                max_len = len(in_[1])
            for j in range(max_len):
                bit = 0
                try:
                    if in_[0][j] or in_[1][j]:
                        bit = 1
                except IndexError:
                    if len(in_[0]) <= j and in_[1][j]:
                        bit = 1
                    elif in_[0]:
                        bit = 1

                out_spt += bit
        elif pli == 'xor':
            max_len = len(in_[0])
            if len(in_[1]) > max_len:
                max_len = len(in_[1])
            for j in range(max_len):
                bit = 0
                try:
                    if (in_[0][j] and not in_[1][j]) or (not in_[0][j] and in_[1][j]):
                        bit = 1
                except IndexError:
                    if len(in_[0]) <= j and in_[1][j]:
                        bit = 1
                    elif in_[0]:
                        bit = 1

                out_spt += bit
        elif pli == 'not':
            for j in in_[0]:
                if j:
                    out_spt += 0
                else:
                    out_spt += 1

        fit_cases_add[pli].append({'in': in_, 'out': out_spt})

# Add new fitness cases to existing cases
# Note: This will override the value of keys with the same name
fit_cases.update(fit_cases_add)
f = open('fitness_cases.py', 'w')
f.write("fit_cases = " + str(fit_cases) + "\n")
f.close()
