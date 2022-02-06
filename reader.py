import csv
import sys
from csrreader import read_mmio

def read_csv(filename, delim=None):
    delim = delim if delim is not None else ','
    with open(filename, newline='', delimiter=delim) as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

if  __name__ == "__main__":
    filename = sys.argv[1]
    
    if '.csv' in filename:
        read_csv(filename)
    elif '.tsv' in filename:
        read_csv(filename)
    else:
        I, J, val = [], [], []
        read_mmio(filename, I, J, val)