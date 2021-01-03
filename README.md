# ESNP-GPU
GPU Implementation of Evolving SNP Systems

Prerequisites
=============
Python 3 and `pip` are required to be able to run the files. Create a virtual environment and set it up with the appropriate packages by executing:
```
pip install -r requirements.txt
```

Experiments that run partially parallel were made with the following environment modules:
* anaconda/3-5.3.1
* cuda/10.1_cudnn-7.6.5

Files
=====

The following files are taken verbatim from Casauay et al's repository:
* `do_simulate.py`
* `adv_pli_generator.py`
* `fitcase_generator.py`
* `get_results.py`
* `pli_parser.py`
* `render_graphs.py`
* `simulator.py`

Minor modification was made on the two GA framework code files in order to implement same initial population per evolution:
* `ga_framework-des1.py`
* `ga_framework-des3.py`

These files correspond to the Variant 1 and Variant 3 **in their paper**, respectively. To run these files, they use the same syntax as `ga_gpu.py`, further discussed in the next section.

GPU Implementation
------------------
The code responsible for the execution of the GA-GPU Framework is `ga_gpu.py` which is adapted from Casauay et al's main code with major modifications in order to apply parallel implementation to the GA operators, specifically Mutation, Crossover, and Selection. GPU Implementation of SNP System simulation is adapted from both Casauay et al and Aboy et al.

Given two variants of the GA framework, Variant 1 and Variant 2, where both delete inactive neurons during evolution but the former does not allow the generation of self loops while the latter allows it, `ga_gpu.py` is run with the following syntax:

```
python3 ga_gpu.py -d dir -f file -c op (-p pop | -s "True")?
```
* `-d dir`: selects category directory from `init_pli/`
* `-f file`: selects operation-category file within specified dir
* `-c op`: operation, used in selected corresponding input spike trains from `fitness_cases.py`
* `-p pop`: optional, takes integer input population size. Otherwise, it uses default population size value 80
* `-s "True"`: optional, if added implements Variant 2. Otherwise, it implements Variant 1.

An example run would be 
```
python3 ga_gpu.py -d "sm_pli/" -f "add-sm" -c "add" -p 15
```
where it executes GA-GPU Framework Variant 1 using input Binary Addition, category Baseline, with a population size of 15.

### Output directory structure
Output files generated from our experimentation can be accessed through [this GDrive folder](https://drive.google.com/drive/folders/1EuqsOHZYxcq6XQ1r2SzmU1TRfLs8Ivhx?usp=sharing). These files are compressed for easier transferring due to the large filesizes.

As a minor disclaimer, the output directories are named as follows: `outputs-gpu2-var1/` and `outputs-gpu2-var3/` which correspond to GA-GPU Framework **Variant 1** and **Variant 2** in our paper, respectively. Extra runs and runs made using the GA-CPU Framework are also included in the folder and named accordingly.

Initial Population
------------------
In order to have uniform initial population across all runs of each operation and category, `initpop_generator.py` is created. It outputs file `initial_population.py` which contains a dictionary of populations corresponding to different input types, with each input type as its key.

The available keys for Binary Addition are:
* `add-adv` (adversarial),
* `add-lg` (original), and 
* `add-sm` (baseline)

For Binary Subtraction, available keys are: 
* `sub-adv`, 
* `sub1-lg` (original 1), 
* `sub3-lg` (original 2), and 
* `sub-sm`
