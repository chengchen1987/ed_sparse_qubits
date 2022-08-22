#pragma once

#include "common.h"

template <class Type>
class SparseMat {
public:
    SparseMat();
    ~SparseMat();
    void SparseMat_Init(const l_int& m_dim, const l_int& m_len);
    void SparseMat_Copy(SparseMat& mat_b, int& m_dim, const l_int& m_len);
    void SparseMat_Copy_real2complex(SparseMat<double>& mat_b, int& m_dim, const l_int& m_len);
    void SparseMat_Write(char* fname);
    void SparseMat_Read(char* fname);
    void SparseMat_Clear();
    // size of matrix, Number of the nonzero elements (length of data)
    int mat_dim;
    int mat_nnz;
    // data, compressed sparse row format
    Type* vals;
    int* cols;
    int* Pointer_BE;
    // debug, get dense form 
    void SparseMat_ToDense();
    Type* denseform;
};

template <class Type>
SparseMat<Type>::SparseMat() {
}
template <class Type>
SparseMat<Type>::~SparseMat() {
}

template <class Type>
void SparseMat<Type>::SparseMat_Init(const l_int& m_dim, const l_int& m_len) {
    mat_dim = m_dim;
    mat_nnz = m_len;
    vals = new Type[mat_nnz];
    cols = new int[mat_nnz];
    Pointer_BE = new int[mat_dim + 1];
}

template <class Type>
void SparseMat<Type>::SparseMat_Copy(SparseMat& mat_b, int& m_dim, const l_int& m_len) {
    for (l_int i = 0; i < m_len; i++) {
        vals[i] = mat_b.vals[i];
        cols[i] = mat_b.cols[i];
    }
    for (int i = 0; i < m_dim + 1; i++) {
        Pointer_BE[i] = mat_b.Pointer_BE[i];
    }
}

template <class Type>
void SparseMat<Type>::SparseMat_Copy_real2complex(SparseMat<double>& mat_b, int& m_dim, const l_int& m_len) {
    for (l_int i = 0; i < m_len; i++) {
        vals[i] = mat_b.vals[i];
        cols[i] = mat_b.cols[i];
    }
    for (int i = 0; i < m_dim + 1; i++) {
        Pointer_BE[i] = mat_b.Pointer_BE[i];
    }
}

template <class Type>
void SparseMat<Type>::SparseMat_Clear() {
    delete[] vals;
    delete[] cols;
    delete[] Pointer_BE;
}

template <class Type>
void SparseMat<Type>::SparseMat_ToDense() {
    for (l_int i = 0; i < mat_dim; i++) {
        for (l_int j = Pointer_BE[i]; j < Pointer_BE[i + 1]; j++) {
            denseform[i * mat_dim + cols[j]] = vals[j];
        }
    }
}

// write and read ////////////////////////////////////////////////////
template <class Type>
void SparseMat<Type>::SparseMat_Write(char* fname) {
    FILE* fp = fopen(fname, "w");
    fwrite(&mat_dim, sizeof(l_int), 1, fp);
    fwrite(&mat_nnz, sizeof(l_int), 1, fp);
    fwrite(&vals, sizeof(Type), mat_nnz, fp);
    fwrite(&cols, sizeof(l_int), mat_nnz, fp);
    fwrite(&Pointer_BE, sizeof(l_int), mat_dim + 1, fp);
    fclose(fp);
}

template <class Type>
void SparseMat<Type>::SparseMat_Read(char* fname) {
    FILE* fr = fopen(fname, "w");
    fread(&mat_dim, sizeof(l_int), 1, fr);
    fread(&mat_nnz, sizeof(l_int), 1, fr);
    vals = new Type[mat_nnz];
    cols = new l_int[mat_nnz];
    Pointer_BE = new l_int[mat_dim + 1];
    fread(&vals, sizeof(Type), mat_nnz, fr);
    fread(&cols, sizeof(l_int), mat_nnz, fr);
    fread(&Pointer_BE, sizeof(l_int), mat_dim + 1, fr);
    fclose(fr);
}


// sparse matrix operations in CSR format 
double SparseMat_Inner(const SparseMat<double>& smat, double* wf0);
double SparseMat_Inner(const SparseMat<double>& smat, my_cmplx* wf0);


// real Hamiltonian, real tau: imaginary time evolution, exp(-beta*H) 
void RK4_onestep(const SparseMat<double>& smat, const double& tau, double* wf0, double* wf1);
// real Hamiltonian, complex tau: real time evolution, exp(-i*t*H) 
void RK4_onestep(const SparseMat<double>& smat, const my_cmplx& tau, my_cmplx* wf0, my_cmplx* wf1);
// complex Hamiltonian, complex tau: real/imaginary time evolution 
void RK4_onestep(const SparseMat<my_cmplx>& smat, const my_cmplx& tau, my_cmplx* wf0, my_cmplx* wf1);

// one-step of Lanczos time evolution
// |wf1> = exp(-1i*H*dt)|wf0>
void Lanczos_evo_onestep(const SparseMat<my_cmplx>& smat, const int& M, const my_cmplx& dt, my_cmplx* wf0, my_cmplx* wf1);
