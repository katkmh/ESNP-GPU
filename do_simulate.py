from simulator import simulate

directory = input("Directory: ")
if directory:
    out_spt = simulate(directory=directory, print_computations=True)
else:
    out_spt = simulate(print_computations=True)
print(out_spt)
# print(int(out_spt[3:][::-1], 2))
