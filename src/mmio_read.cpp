#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include "mmio.h"

using namespace torch::indexing;

template <typename T_idx, typename T_val>
void read_mmio(std::string filename, T_idx* I, T_idx* J, T_val* val) {
    const char *c_filename = filename.c_str();
    FILE *f;
    if ((f = fopen(c_filename, "r")) == NULL) {
        throw std::runtime_error("Could not open file");
    }

    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0) {
        throw std::runtime_error("Could not process Matrix Market banner.\n");
    }
       
    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */
    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
            mm_is_sparse(matcode) )
    {
        throw std::runtime_error("Sorry, this application does not support this Market Market type\n");
    }
    
    int ret_code;
    T_idx M, N, nz; 
    /* find out size of sparse matrix .... */
    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0) {
        throw std::runtime_error("Issue with mm_read_mtx_crd_size");
    }   

    if (M != N) {
        throw std::runtime_error("Cannot read in because non-square matrix.");
    }
    
    /* reseve memory for matrices */
    // TODO: see if unsigned
    I = (T_idx*) malloc(nz * sizeof(T_idx));
    J = (T_idx*) malloc(nz * sizeof(T_idx));
    val = (T_val*) malloc(nz * sizeof(T_val));

    for (T_idx i = 0; i < nz; i++) {
        fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }
    
    if (f != stdin) {
        fclose(f);
    }
}

// TODO: read tsv/csv 

template <typename T_idx, typename T_val>
void coo_to_csr(T_idx nz, T_idx M, T_val v, T_idx* csr_rowptr, T_idx* csr_colptr, float* csr_val) {
    // careful of bidirectional edges and undirected/directed
    // TODO: typecheck if nz > max
    //

    
}

template <typename T_idx, typename T_val>
void torch_coo_to_csr(T_idx nz, T_idx M, T_val v, T* csr_rowptr, T* csr_colptr, float* csr_val) {
    // careful of bidirectional edges and undirected/directed
    // TODO: typecheck if nz > max
    //

    rowptr = torch::zeros(M+1, torch::TensorOptions().dtype(torch::kInt64)); 
    colptr = torch::zeros(nz, torch::TensorOptions().dtype(torch::kInt64));
    val = torch::zeros(nz, torch::TensorOptions().dtype(torch::kFloat32));
    
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("read_mmio", &read_mmio, "Reads Matrix Market file and returns CSR format.");
  m.def("coo_to_csr", &coo_to_csr, "");
  m.def("read_mmio_csr", &read_mmio_csr, "");
}
