#!/bin/bash

#SBATCH -n 1 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 2:00:00 # Runtime in minutes
#SBATCH -p hips # Partition to submit to
#SBATCH -a 1-40 # Number of copies of this job
#SBATCH --mem-per-cpu=4000 # Memory per cpu in MB (see also --mem)
#SBATCH -o jobname-%a.out # Standard out goes to this file
#SBATCH -e jobname-%a.err # Standard err goes to this filehostname


# Now the experiment code starts!

echo "Starting experiment from Odyssey"
python -u ~/repos/DeepMoleculesData/experiments/2015-05-25-delaney/8-odyssey/experiment.py
echo "Finished experiment on Odyssey"
