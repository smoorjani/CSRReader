#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <format>

#include "mmio.h"
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

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
    std::cout << (sizeof(T_idx)) << std::endl;
    I = (T_idx*) malloc(nz * sizeof(T_idx));
    J = (T_idx*) malloc(nz * sizeof(T_idx));
    val = (T_val*) malloc(nz * sizeof(T_val));

    char* parse_string;

    if (delim == ',') {
        parse_string = "%d,%d,%lg\n";
    } else {
        parse_string = "%d\t%d\t%lg\n";
    }
    for (T_idx i = 0; i < nz; i++) {
        fscanf(f, parse_string, &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }
    
    if (f != stdin) {
        fclose(f);
    }
}

// TODO: read tsv/csv 
template <typename T_idx, T_val>
void read_csv(std::string filename, T_idx* I, T_idx* J, T_val* val, T_idx* _M, T_idx* _nnz, char delim=',') {
    const char *c_filename = filename.c_str();
    FILE *f;
    if ((f = fopen(c_filename, "r")) == NULL) {
        throw std::runtime_error("Could not open file");
    }

    T_idx M, nnz = 0;
    T_idx tmprow;
    T_idx tmpcol;
    T_idx tmpval;

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    // 2 indices, 1 value, 2 delimiters, 1 newline
    *nnz = size / (sizeof(T_idx) * 2 + sizeof(T_val) + sizeof(char) * 3);

    I = (T_idx*) malloc(nnz * sizeof(T_idx));
    J = (T_idx*) malloc(nnz * sizeof(T_idx));
    val = (T_val*) malloc(nnz * sizeof(T_val));

    for (T_idx i = 0; i < nnz; i++) {
        fscanf(f, "%d,%d,%lg\n", &I[i], &J[i], &val[i]);
        M = MAX(M, I[i]);
        M = MAX(M, J[i]);
    }

    if (f != stdin) {
        fclose(f);
    }

    *_M = M;
    *_nnz = nnz;
}

template <typename T_idx>
void coo_to_csr(T_idx nnz, T_idx M, T_idx* coo_rowptr, T_idx* csr_rowptr) {
    // careful of bidirectional edges and undirected/directed
    //
    if (nnz > M * M) {
        throw std::runtime_error("Too many nonzero values");
    }

    for (T_idx i = 0; i < nnz; i++) {
        csr_rowptr[coo_rowptr[i] + 1]++;
    }
    for (T_idx i = 0; i < M; i++) {
        csr_rowptr[i + 1] += csr_rowptr[i];
    }
}

template <typename T_idx>
void coo_to_csr(T_idx nnz, T_idx M, torch::Tensor coo_rowptr, torch::Tensor csr_rowptr) {
    // careful of bidirectional edges and undirected/directed
    // TODO: typecheck if nz > max
    //

    if (nnz > M * M) {
        throw std::runtime_error("Too many nonzero values");
    }

    T_idx* coo_rowdata = coo_rowptr.data_ptr<T_idx>();
    T_idx* csr_rowdata = csr_rowptr.data_ptr<T_idx>();

    for (T_idx i = 0; i < nnz; i++) {
        csr_rowdata[coo_rowdata[i] + 1]++;
    }

    for (T_idx i = 0; i < M; i++) {
        csr_rowdata[i + 1] += csr_rowdata[i];
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("read_mmio", &read_mmio<uint64_t, double>, "Reads Matrix Market file and returns CSR format.");
  m.def("coo_to_csr", &coo_to_csr<uint64_t>, "");
  m.def("read_mmio_csr", &read_mmio_csr, "");
}
