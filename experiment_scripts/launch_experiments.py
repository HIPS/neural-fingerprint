import os
import sys
import json
import subprocess
from subprocess import Popen, PIPE
import numpy as np
import numpy.random as npr
from time import sleep
from tempfile import NamedTemporaryFile

from nfpexperiment import util

LAUNCH_USING_SLURM = True
DEBUG_JOB = False

def make_sbatch_args(outfile):
    return ['sbatch',
            '-n', '1',
            '-N', '1',
            '-t', '10:00:00',
            '-p', 'hips',
            '--mem-per-cpu=4000',
            '-o', outfile]

experiment_command = sys.argv[1]
if sys.argv[2] == "test":
    test_mode = True
elif sys.argv[2] == "validation":
    test_mode = False
else:
    raise Exception("Need to choose between validation and test mode using second argument.")

if __debug__:
    N_cores = 1
    N_jobs = 1
    num_iters = 1
    num_folds = 3
    #datasets = ['toxin', 'delaney', 'malaria', 'cep']
    #dataset_sizes = [100, 100,    100,     100]
    datasets = ['toxin']
    dataset_sizes = [7297]
else:
    N_cores = 8
    N_jobs = 50
    num_iters = 10000
    num_folds = 5
    datasets = ['delaney', 'toxin', 'malaria', 'cep']
    dataset_sizes = [1124, 7297,    10000,     20000]

morgan_bounds = dict(fp_length      = [16, 1024],
                     fp_depth       = [1, 4],
                     log_init_scale = [-2, -6],
                     log_step_size  = [-8, -4],
                     log_L2_reg     = [-6, 2],
                     h1_size        = [50, 100],
                     conv_width     = [5, 20])

neural_bounds = dict(fp_length      = [16, 128],   # Smaller upper range.
                     fp_depth       = [1, 4],
                     log_init_scale = [-2, -6],
                     log_step_size  = [-8, -4],
                     log_L2_reg     = [-6, 2],
                     h1_size        = [50, 100],
                     conv_width     = [5, 20])

model_bounds = dict(mean = morgan_bounds,
                    morgan_plus_linear = morgan_bounds,
                    morgan_plus_net = morgan_bounds,
                    conv_plus_linear = neural_bounds,
                    conv_plus_net = neural_bounds)

def generate_params(bounds, N_jobs):
    rs = npr.RandomState(0)
    for jobnum in xrange(N_jobs):
        yield jobnum, {param_name : rs.uniform(*param_bounds)
                       for param_name, param_bounds in bounds.iteritems()}

def outputfile(jobnum, dataset, model, fold):
    debug_text = '_DEBUG' if __debug__ else ''
    return 'results{}/{}_{}_{}_{}.json'.format(debug_text, fold, jobnum, dataset, model)

def outputfile_test(dataset, model, fold):
    debug_text = '_DEBUG' if __debug__ else ''
    return 'test_results{}/test_{}_{}_{}.json'.format(debug_text, fold, dataset, model)

def build_params_string(jobnum, varied_params, dataset, model, train_slices, test_slices, fold):
    params = dict(num_records = 20,
                  jobnum = jobnum,
                  model = dict(net_type   = model,
                               fp_length  = int(round(varied_params['fp_length'])),
                               fp_depth   = int(round(varied_params['fp_depth'])),
                               conv_width = int(round(varied_params['conv_width'])),
                               h1_size    = int(round(varied_params['h1_size'])),
                               L2_reg     = np.exp(varied_params['log_L2_reg'])),
                  train = dict(num_iters  = num_iters,
                               batch_size = 100,
                               init_scale = np.exp(varied_params['log_init_scale']),
                               step_size  = np.exp(varied_params['log_step_size']),
                               seed       = fold + 1),
                  task = dict(name = dataset,
                              train_slices = train_slices,
                              test_slices  = test_slices),
                  varied_params = varied_params)

    return json.dumps(params)

def runjob_debug(params_string, outfile):
    print params_string
    print outfile

def runjob_local(params_string, outfile):
    with open(outfile, 'w') as f:
        p = Popen(experiment_command, stdin=PIPE, stdout=f, shell=True)
        p.stdin.write(params_string)
        p.stdin.close()
        return p

def runjob_sbatch(params_string, outfile):
    infile_obj =  NamedTemporaryFile(dir='./tmp', delete=False)
    with infile_obj as f:
        f.write(params_string)
    infile = infile_obj.name
    logfile = outfile + '_log.txt'
    bash_script = '#!/bin/bash\n{} >{} <{}'.format(experiment_command, outfile, infile)
    p = subprocess.Popen(make_sbatch_args(logfile), stdin=subprocess.PIPE)
    p.communicate(bash_script)
    if p.returncode != 0:
        raise RuntimeError

def generate_slice_lists(num_folds, N_data):
    chunk_boundaries = map(int, np.linspace(0, N_data, num_folds + 1))
    chunk_slices = zip(chunk_boundaries[0:-1], chunk_boundaries[1:])

    for f_ix in range(num_folds):
        validation_chunk_ixs = [f_ix]
        test_chunk_ixs = [(f_ix + 1) % num_folds]
        train_chunk_ixs = [i for i in range(num_folds) if i not in validation_chunk_ixs + test_chunk_ixs]
        yield (map(chunk_slices.__getitem__, train_chunk_ixs),
               map(chunk_slices.__getitem__, validation_chunk_ixs),
               map(chunk_slices.__getitem__, test_chunk_ixs))

extract_test_loss = lambda job_data : job_data['test_loss']

if not test_mode:
    print "Starting validation experiments..."
    all_jobs = []
    still_running = lambda job : job.poll() is None
    for dataset, num_data in zip(datasets, dataset_sizes):
        for fold, (train_slices, validation_slices, test_slices)\
                in enumerate(generate_slice_lists(num_folds, num_data)):
            for model, bounds in model_bounds.iteritems():
                for jobnum, varied_params in generate_params(bounds, N_jobs):
                    print "\nRunning fold {} job {} on dataset {} with model {} " \
                          "and params {}".format(fold, jobnum, dataset, model, varied_params)
                    cur_outfile = outputfile(jobnum, dataset, model, fold)
                    if os.path.exists(cur_outfile) and os.path.getsize(cur_outfile) > 0:
                        print "SKIPPING because it's already done."
                        continue
                    params_string = build_params_string(jobnum, varied_params, dataset, model,
                                                        train_slices, validation_slices, fold)
                    if DEBUG_JOB:
                        runjob_debug(params_string, cur_outfile)
                    elif LAUNCH_USING_SLURM:
                        runjob_sbatch(params_string, cur_outfile)
                    else:
                        p = runjob_local(params_string, cur_outfile)
                        all_jobs.append(p)   # Only have a few jobs running at a time.
                        while len(all_jobs) >= N_cores:
                            sleep(1)
                            all_jobs = filter(still_running, all_jobs)
else:
    print "Starting test experiments..."
    all_jobs = []
    still_running = lambda job : job.poll() is None
    for dataset, num_data in zip(datasets, dataset_sizes):
        for fold, (train_slices, validation_slices, test_slices)\
                in enumerate(generate_slice_lists(num_folds, num_data)):
            for model in model_bounds:
                cur_outfile = outputfile_test(dataset, model, fold)
                if os.path.exists(cur_outfile) and os.path.getsize(cur_outfile) > 0:
                    print "SKIPPING because it's already done."
                    continue

                all_outfiles = [outputfile(jobnum, dataset, model, fold) for jobnum in range(N_jobs)]
                all_results = util.get_jobs_data(all_outfiles)
                print "Loaded {} results for fold {} dataset {} model {}"\
                    .format(len(all_results), fold, dataset, model)
                best_hypers = sorted(all_results, key=extract_test_loss)[0]['params']
                best_hypers['seed'] = 0
                best_hypers['task']['train_slices'] = train_slices + validation_slices
                best_hypers['task']['test_slices'] = test_slices
                params_string = json.dumps(best_hypers)
                if DEBUG_JOB:
                    runjob_debug(params_string, cur_outfile)
                elif LAUNCH_USING_SLURM:
                    runjob_sbatch(params_string, cur_outfile)
                else:
                    p = runjob_local(params_string, cur_outfile)
                    all_jobs.append(p)   # Only have a few jobs running at a time.
                    while len(all_jobs) >= N_cores:
                        sleep(1)
                        all_jobs = filter(still_running, all_jobs)
