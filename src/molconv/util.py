import os
import sys
import csv
import numpy as np

def load_csv(filename):
    filename = os.path.expanduser(filename)

    molecules = []
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        header = reader.next()
        for row in reader:

            # Make Nulls into a NaNs.
            # Note that these were previously treated as zeros.
            row = [np.NaN if r == 'Null' else r for r in row]

            molecules.append(dict(zip(header, row)))
    return molecules
