# Takes the name of a python script in DeepMolecule, and param file,
# and a directory, and runs it as a job on Odyssey.

import os
import subprocess
from pprint import pformat

def run_jobs(job_generator, python_script, dir_prefix):
    for dir_name, params in job_generator():
        # Write params to a temp file
        outdir = os.path.join(dir_prefix, dir_name)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        params_file = os.path.join(outdir, 'params.txt')
        with open(params_file, 'w') as f:
            f.write(pformat(params))
        call_on_odyssey(python_script, params_file, dir_name)

def call_on_odyssey(python_script, params_file, dir_name):
    slurm_string = """
#!/bin/bash
#SBATCH -n 4 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 2:00 # Runtime in minutes
#SBATCH -p hips # Partition to submit to
#SBATCH --mem-per-cpu=4000 # Memory per cpu in MB (see also --mem)
#SBATCH -o /scratch/jobouts/jobname-%A.out # Standard out goes to this file
#SBATCH -e /scratch/jobouts/jobname-%A.err # Standard err goes to this filehostname

# Now the experiment code starts!
echo "Starting experiment from Odyssey"

outdir=~/repos/DeepMoleculesData/results
expname=odyssey-test
pyscript=~/repos/DeepMolecules/experiment_scripts/compare-fingerprints-to-convnet.py
#~/repos/DeepMolecules/experiment_scripts/trusty_scribe -d $outdir -n $expname python -u $pyscript
#~/repos/DeepMolecules/experiment_scripts/trusty_scribe -d $outdir -n $expname python -u $pyscript
#echo "Finished experiment on Odyssey"

~/repos/DeepMolecules/experiment_scripts/python -u """

    slurm_string += python_script + ' ' + params_file + ' ' + dir_name + '\n'
    with open('slurm_file.slurm', 'w') as f:
        f.write(slurm_string)
    subprocess.call('sbatch slurm_file.slurm')
    os.remove('slurm_file.slurm')


