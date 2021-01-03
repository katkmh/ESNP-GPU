import re


def parse(filename):
	keywords = ['@mu', '@ms', '@marcs', '@min', '@mout', '-->']
	nrn_names = []
	spikes = {}
	synapses = []
	in_nrn_names = []
	out_nrn_names = []
	rules = {}
	f = open(filename, 'r')
	for line in f:
		a = ''.join(line.split()) # Remove all white spaces
		a = a.split(';')
		for b in a:
			for kw in keywords:
				if kw in b:
					if kw == '@mu':
						c = b.split('=')
						c = c[1]
						nrn_names.extend(c.split(','))
					elif kw == '@ms':
						c = b.split('=')
						nrn = re.search('\((.*)\)', c[0]).group(1)
						if '*' in c[1]:
							spk = int(re.search('\*(.*)', c[1]).group(1))
						else:
							spk = 1
						spikes[nrn] = spk
					elif kw == '@marcs':
						c = b.split('=')
						c = c[1]
						syn = re.search('\((.*)\)', c).group(1)
						syn = syn.split(',')
						synapses.append((syn[0], syn[1]))
					elif kw == '@min':
						c = b.split('=')
						c = c[1]
						in_nrn_names.extend(c.split(','))
					elif kw == '@mout':
						c = b.split('=')
						c = c[1]
						out_nrn_names.extend(c.split(','))
					else:
						c = b.split("'")

						con = str(re.search('\[(.*)-->', c[0]).group(1))
						if '*' in con:
							con = int(re.search('\*(.*)', con).group(1))
						else:
							con = 1

						prod = str(re.search('-->(.*)\]', c[0]).group(1))
						if '*' in prod:
							prod = int(re.search('\*(.*)', prod).group(1))
						elif '#' in prod:
							prod = 0
						else:
							prod = 1

						if '"' in c[1]:
							nrn = re.search('(.*)"a', c[1]).group(1)

							e = re.search('"(.*)"', c[1]).group(1)
							if '*' in e:
								x = int(re.search('\*(.*)', e).group(1))
							else:
								x = 1

							if ':' in c[1]:
								delay = int(re.search('::(.*)', c[1]).group(1))
							else:
								delay = 0
						elif ':' in c[1]:
							nrn = re.search('(.*)::', c[1]).group(1)
							x = con
							delay = int(re.search('::(.*)', c[1]).group(1))
						else:
							nrn = c[1]
							x = con
							delay = 0

						try:
							rules[nrn].append({'x': x, 'y': 0, 'c': con, 'p': prod, 'delay': delay})
						except KeyError:
							rules[nrn] = [{'x': x, 'y': 0, 'c': con, 'p': prod, 'delay': delay}]

	f.close()

	return nrn_names, spikes, synapses, in_nrn_names, out_nrn_names, rules

# 	print(nrn_names)
# 	print(spikes)
# 	print(synapses)
# 	print(in_nrn_names)
# 	print(out_nrn_names)
# 	print(rules)
#
#
# parse(input("Input filename: "))
