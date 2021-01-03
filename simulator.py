"""
Simulates SN P systems in the generative mode
"""
from pli_parser import parse
from beautifultable import BeautifulTable


class Nrn:
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
	def __init__(self, E, c, p, nrn, delay=0):
		"""
			E: (a-x) % y == 0 where 
			a is in the regex a^x(a^y)*
		"""
		self.x = E[0]
		self.y = E[1]
		self.c = c
		self.p = p
		self.nrn = nrn
		self.delay = delay

	def in_e(self, a):
		if self.y == 0:
			if a-self.x == 0:
				return True
		else:
			if (a-self.x) % self.y == 0:
				return True
		return False

	def consume(self):
		self.nrn.a = self.nrn.a - self.c

	def produce(self, neurons, syn):
		self.nrn.distrib['delay'] = self.delay
		self.nrn.distrib['a'] = self.p

		if self.p != 0:
			for i in range(len(neurons)):
				if syn[self.nrn.idn][i]:
					self.nrn.distrib['receivers'].append(neurons[i])

	def spike(self, neurons, syn):
		if self.in_e(self.nrn.a) and self.nrn.distrib['delay'] == -1:
			self.consume()
			self.produce(neurons, syn)
			return True
		return False


def traverse(neurons, out_nrn_names, syn):
	# Remove all spikes from the output neurons
	for out_name in out_nrn_names:
		for nrn in neurons:
			if nrn.name == out_name:
				nrn.a = 0

	# See which neurons can use a rule (i.e. spike)
	cont = False
	for nrn in neurons:
		for rule in nrn.rules:
			spiked = rule.spike(neurons, syn)
			if spiked:
				cont = True
				break

	if cont is False:
		return neurons, cont

	for nrn in neurons:
		if nrn.distrib['delay'] == 0:
			for receiver in nrn.distrib['receivers']:
				receiver.enter_spikes(nrn.distrib['a'])
			nrn.distrib['receivers'] = []
		if nrn.distrib['delay'] > -1:
			nrn.distrib['delay'] -= 1

	return neurons, cont


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


def simulate(filename="", in_spt=None, directory="sm_pli", parsed=(), print_computations=False):
	if parsed == ():
		if len(filename):
			nrn_names, spikes, synapses, in_nrn_names, out_nrn_names, rules = \
				parse(directory + '/' + filename + '.pli')
		else:
			nrn_names, spikes, synapses, in_nrn_names, out_nrn_names, rules = \
				parse(directory + '/' + input("Input filename: ") + '.pli')
	else:
		nrn_names, spikes, synapses, in_nrn_names, out_nrn_names, rules = parsed

	neurons = []
	for name in nrn_names:
		try:
			n = Nrn(idn=len(neurons), name=name, a=spikes[name])
		except KeyError:
			n = Nrn(idn=len(neurons), name=name)

		try:
			for rule in rules[name]:
				n.add_rule(E=(rule['x'], rule['y']), c=rule['c'], p=rule['p'], delay=rule['delay'])
		except KeyError:
			pass
		finally:
			neurons.append(n)

	syn_matrix = [[0]*len(nrn_names) for i in range(len(nrn_names))]
	for syn in synapses:
		i, j = None, None
		for nrn in neurons:
			if nrn.name == syn[0]:
				i = nrn.idn
			if nrn.name == syn[1]:
				j = nrn.idn
			if i and j:
				break
		syn_matrix[i][j] = 1

	# Get input spike trains
	if in_spt is None:
		in_spt = []
		while True:
			for i in range(len(in_nrn_names)):
				while True:
					spt = input('Input spike train #' + str(i + 1) + ': ')
					if is_binary_string(spt):
						in_spt.append(spt)
						break
					else:
						print("Please input spike trains in binary.")

			max_spt = get_max_spt(in_spt)
			if not max_spt:
				print("Please input at least one non-empty spike train.")
			else:
				break
	else:
		max_spt = get_max_spt(in_spt)

	# Add 0s to spike trains shorter than the maximum length
	for spt_index in range(len(in_spt)):
		add_0 = max_spt - len(in_spt[spt_index])
		in_spt[spt_index] += '0'*add_0

	# SIMULATE SN P SYSTEM
	num_neurons_in_tables = 10
	neurons, cont = traverse(neurons, out_nrn_names, syn_matrix)
	# Display computations
	if print_computations:
		tables = []
		index = 0
		for i in range(int(len(neurons)/num_neurons_in_tables)):
			table = BeautifulTable()
			tables.append(table)
			header = ['step']
			for j in range(index, index + num_neurons_in_tables):
				header.append(neurons[j].name)
			table.column_headers = header
			index = num_neurons_in_tables*(i+1)

		table = BeautifulTable()
		tables.append(table)
		header = ['step']
		for j in range(index, len(neurons)):
			header.append(neurons[j].name)
		table.column_headers = header

	out_spt = ''
	cont = True
	step = 0
	in_spt_copy = in_spt.copy()
	while (cont or get_max_spt(in_spt_copy)) and step < get_max_spt(in_spt) + len(neurons):
		step += 1
		if get_max_spt(in_spt_copy):
			for spt_index in range(len(in_spt_copy)):
				spt = in_spt_copy[spt_index]
				a = int(spt[0])
				in_spt_copy[spt_index] = spt[1:]
				for nrn in neurons:
					if nrn.name == in_nrn_names[spt_index]:
						nrn.enter_spikes(a)

		# Get spikes sent to the environment
		env_spikes = 0
		for out_name in out_nrn_names:
			for nrn in neurons:
				if nrn.name == out_name:
					env_spikes += nrn.a
		out_spt += str(env_spikes)

		if print_computations:
			index = 0
			for i in range(len(tables) - 1):
				row = [step]
				for j in range(index, index + num_neurons_in_tables):
					row.append(neurons[j].a)
				tables[i].append_row(row)
				index = num_neurons_in_tables*(i+1)

			row = [step]
			for j in range(index, len(neurons)):
				row.append(neurons[j].a)
			tables[-1].append_row(row)

		neurons, cont = traverse(neurons, out_nrn_names, syn_matrix)

	if print_computations:
		for table in tables:
			print(table)

	return out_spt
