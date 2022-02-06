import csv
import sys
from numpy import dtype
import torch

from csrreader import read_mmio

def read_csv(filename, delim=None):
    rowptr, colptr, val = [], [], []
    delim = delim if delim is not None else ','
    with open(filename, newline='', delimiter=delim) as f:
        reader = csv.reader(f)
        for line in reader:
            val.append(line[0])
            colptr.append(line[1])
            rowptr.append(line[2])

    return torch.Tensor(val), torch.Tensor(colptr, dtype=torch.int16), torch.Tensor(rowptr, dtype=torch.int16)
    
if  __name__ == "__main__":
    filename = sys.argv[1]
    
    rowptr, colptr, val = torch.Tensor(), torch.Tensor(), torch.Tensor()
    if '.csv' in filename:
        rowptr, colptr, val = read_csv(filename)
    elif '.tsv' in filename:
        rowptr, colptr, val = read_csv(filename)
    else:
        read_mmio(filename, rowptr, colptr, val)