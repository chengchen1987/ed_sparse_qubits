#pragma once

#pragma warning(disable : 4996)

#include <cstdint>
#include <complex>

// #define l_int long long
// I will set all integer related to "Dim" as l_int, in case of large systems
// do not use unsigned type because some intel MKL functions only support sigened type

#define MKL_Complex16 std::complex<double>
#include <mkl.h>
#define l_int MKL_INT
#define my_cmplx std::complex<double>

#define PI 3.14159265358979323846

void rd_wf(const l_int& dim, double* wf);
void rd_wf(const l_int& dim, my_cmplx* wf);
void uni_wf(const l_int& dim, double* wf);
void rd_wf_product(const l_int& dim, double* wf);

// random number
double ran_num();
// bit operation, default l_int, 'int' for length of int 
int numOfBit1(const l_int& a);
void findBit1(const l_int& a, const int& n, int* b);
void print_binary(const l_int& a, const int& n);
int HammingDis(const l_int& a, const l_int& b);
void bits_decomposition(const l_int& s, const int& n, const int& size_A, int* sites_A, l_int &a, l_int &b);
//
l_int nchoosek(const int& k, const int& n);
//
void NormalizedCopy(const double* f, double* g, const l_int& Dim);
void SetToZero(l_int* tmp, l_int length);
void SetToZero(double* f, const l_int dim);
//
void MatrixEvd(int matrix_layout, char jobz, char uplo, lapack_int n, double* a, lapack_int lda, double* w);
void DenseMatrixEigenSolver(int matrix_layout, char jobz, char uplo, lapack_int n, double* a, lapack_int lda, double* w);
void DenseMatrixEigenSolver_FInterface(int matrix_layout, char jobz, char uplo, lapack_int n, double* a, lapack_int lda, double* w);
void MatrixSvd(int matrix_layout, char jobu, char jobvt, lapack_int m, lapack_int n, double* a, lapack_int lda, double* s, double* u, lapack_int ldu, double* vt, lapack_int ldvt);
void MatrixSvd(int matrix_layout, char jobu, char jobvt, lapack_int m, lapack_int n, my_cmplx* a, lapack_int lda, double* s, my_cmplx* u, lapack_int ldu, my_cmplx* vt, lapack_int ldvt);
//

double VN_entropy(const l_int& dim, double* wf);

// parse input 
void GetParaFromInput_int(const char* fname, const char* string_match, int& para);
void GetParaFromInput_real(const char* fname, const char* string_match, double& para);
void GetParaFromInput_char(const char* fname, const char* string_match, char& para);

void Vec_fwrite_double(const char* fname, double* data, const int& dsize);
void Vec_fread_double(const char* fname, double* data, const int& dsize);

double my_vec_dot(const l_int& n, double* x, const l_int incx, double* y, const l_int incy);
my_cmplx my_vec_dot(const l_int& n, my_cmplx* x, const l_int incx, my_cmplx* y, const l_int incy);


void Lanczos(const l_int& Dim, double* vals, l_int* cols, l_int* PointerBE, const l_int& M, char job_vec,
	double* l_vals, double* l_vecs, double* l_eigvecs); // job_vec: ('N')'I' (no) Lanzcos vectors 
//void Lanczos(const l_int& Dim, my_cmplx* vals, l_int* cols, l_int* PointerBE, const l_int& M, char job_vec,
//	double* l_vals, my_cmplx* l_vecs, my_cmplx* l_eigvecs); // job_vec: ('N')'I' (no) Lanzcos vectors 
void Lanczos(const l_int& dim, my_cmplx* vals, l_int* cols, l_int* PointerBE, const l_int& M, char job_vec,
	double* alpha, my_cmplx* l_vecs, double* l_eigvecs);

// extra vector operations 
// y = a*x
void myvec_ax(const l_int& dim, const double& a, double* x, double* y); 
void myvec_ax(const l_int& dim, const my_cmplx &a, my_cmplx* x, my_cmplx*y);
// a = real(x)
void myvec_real(const l_int&dim, my_cmplx *x, double *a);
// a = imag(x)
void myvec_imag(const l_int& dim, my_cmplx* x, double* a);
// a = x + i*y, x and y are real vectors
void myvec_xpiy(const l_int& dim, double* x, double* y, my_cmplx* a);