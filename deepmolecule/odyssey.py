# Takes the name of a python script in DeepMolecule, and param file,
# and a directory, and runs it as a job on Odyssey.

import os
import subprocess
from pprint import pformat

def run_jobs(job_generator, python_script, dir_prefix):
    if os.path.exists(dir_prefix):
	raise Exception("Experiment directory already exists: " + dir_prefix)
    for jobname, params in job_generator():
        # Write params to a temp file
        outdir = os.path.join(dir_prefix, jobname)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        params_file = os.path.join(outdir, 'params.txt')
        with open(params_file, 'w') as f:
            f.write(pformat(params))
        call_on_odyssey(python_script, params_file, outdir, jobname)

def call_on_odyssey(python_script, params_file, outdir, jobname):
    slurm_string = \
"""#!/bin/bash
#SBATCH -n 1 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 16:00:00
#SBATCH -p hips
#SBATCH --mem-per-cpu=8000
#SBATCH -o """ + outdir + """/stdout.txt
#SBATCH -e """ + outdir + """/stderr.txt
#SBATCH -J """ + jobname + """
echo "Starting experiment from Odyssey"
echo
python -u """ + python_script + ' ' + params_file + ' ' + outdir + """
echo
echo "Finished experiment on Odyssey" """

    with open('slurm_file.slurm', 'w') as f:
        f.write(slurm_string)
    subprocess.call(['sbatch', 'slurm_file.slurm'])
    os.remove('slurm_file.slurm')


