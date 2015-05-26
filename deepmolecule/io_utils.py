import os
import csv
import autograd.numpy as np
from util import slicedict
from collections import defaultdict

def read_csv(filename, nrows, input_name, target_name):
    data = defaultdict(list)
    with open(filename) as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            if i > nrows: break
            data[input_name].append(row[input_name])
            data[target_name].append(float(row[target_name]))
    return data

def load_data(filename, sizes, input_name, target_name):
    nrows_total = sum(sizes)
    data = read_csv(filename, nrows_total, input_name, target_name)
    all_data = {colname : np.array(data[colname]) for colname in data}

    datasets = []
    start = 0
    for size in sizes:
        end = start + size
        datasets.append(slicedict(all_data, slice(start, end)))
        start = end
    return datasets

def get_output_file(rel_path):
    return os.path.join(output_dir(), rel_path)

def get_data_file(rel_path):
    return os.path.join(data_dir(), rel_path)

def output_dir():
    return os.path.expanduser(safe_get("OUTPUT_DIR"))

def data_dir():
    return os.path.expanduser(safe_get("DATA_DIR"))

def safe_get(varname):
    if varname in os.environ:
        return os.environ[varname]
    else:
        raise Exception("%s environment variable not set" % varname)
