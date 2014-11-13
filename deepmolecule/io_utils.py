import os
import pandas as pd
import numpy as np
from util import slicedict

def load_data(filename, sizes):
    nrows_total = sum(sizes)
    pd_data = pd.io.parsers.read_csv(filename, sep=',', nrows=nrows_total)
    all_data = {colname : np.array(pd_data[colname]) for colname in pd_data}

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
