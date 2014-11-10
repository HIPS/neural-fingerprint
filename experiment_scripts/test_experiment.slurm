#!/bin/bash

#SBATCH -n 4 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 2:00 # Runtime in minutes
#SBATCH -p hips # Partition to submit to
#SBATCH --mem-per-cpu=4000 # Memory per cpu in MB (see also --mem)
#SBATCH -o jobname-%A.out # Standard out goes to this file
#SBATCH -e jobname-%A.err # Standard err goes to this filehostname


# Now the experiment code starts!

echo "Starting experiment from Odyssey"

outdir=~/repos/DeepMoleculesData/results
expname=odyssey-test
pyscript=~/repos/DeepMolecules/experiment_scripts/compare-fingerprints-to-convnet.py
~/repos/DeepMolecules/experiment_scripts/trusty_scribe -d $outdir -n $expname python -u $pyscript

echo "Finished experiment on Odyssey"
