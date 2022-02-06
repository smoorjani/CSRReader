#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include "mmio.h"

using namespace torch::indexing;

void read_mmio(std::string filename, torch::Tensor I, torch::Tensor J, torch::Tensor val) {
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
    
    int ret_code, M, N, nz; 
    /* find out size of sparse matrix .... */
    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0) {
        throw std::runtime_error("Issue with mm_read_mtx_crd_size");
    }   
    
    /* reseve memory for matrices */
    I = torch::zeros(nz, torch::TensorOptions().dtype(torch::kInt16));
    J = torch::zeros(nz, torch::TensorOptions().dtype(torch::kInt16));
    val = torch::zeros(nz, torch::TensorOptions().dtype(torch::kFloat64));

    int i, j;
    double v;
    for (int idx = 0; idx < nz; idx++) {
        fscanf(f, "%d %d %lg\n", &i, &j, &v);
        i--;  /* adjust from 1-based to 0-based */
        j--;
        I.index_put_({idx}, i);
        J.index_put_({idx}, j);
        val.index_put_({idx}, v);
    }
    
    if (f != stdin) {
        fclose(f);
    }

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("read_mmio", &read_mmio, "Reads Matrix Market file and returns CSR format.");
}
