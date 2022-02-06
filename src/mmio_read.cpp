#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"

void read_mmio(std::string filename, py::list& pyI, py::list& pyJ, py::list& pyval) {
    FILE *f;
    if ((f = fopen(filename, "r")) == NULL) {
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
        throw std::runtime_error("Sorry, this application does not support Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    }
    
    int ret_code, M, N, nz; 
    /* find out size of sparse matrix .... */
    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0) {
        throw std::runtime_error("Issue with mm_read_mtx_crd_size");
    }   
    
    int i
    /* reseve memory for matrices */

    int *I = (int *) malloc(nz * sizeof(int));
    int *J = (int *) malloc(nz * sizeof(int));
    double *val = (double *) malloc(nz * sizeof(double));

    pyI = py::cast(I)
    pyJ = py::cast(J)
    pyval = py::cast(val)

    for (int i = 0; i < nz; i++) {
        fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }
    
    if (f != stdin) {
        fclose(f);
    }

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("read_mmio", &read_mmio, "Reads Matrix Market file and returns CSR format.");
}