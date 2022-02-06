import csv
import sys
from csrreader import read_mmio

def read_csv(filename, delim=None):
    delim = delim if delim is not None else ','
    with open(filename, newline='', delimiter=delim) as f:
        reader = csv.reader(f)
        data = list(reader)

    rowptr, colptr, val = [], [], []
    for i in range(len(data), 3):
        rowptr.append(data[i])
        colptr.append(data[i+1])
        val.append(data[i+2])
    
if  __name__ == "__main__":
    filename = sys.argv[1]
    
    rowptr, colptr, val =  [], [], []
    if '.csv' in filename:
        rowptr, colptr, val = read_csv(filename)
    elif '.tsv' in filename:
        rowptr, colptr, val = read_csv(filename)
    else:
        read_mmio(filename, rowptr, colptr, val)