#include <fstream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <ctime>
#include <sstream>

#include <iostream>
#include <iomanip>
#include <random>
#include <cstring>
#include "common.h"

std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
std::uniform_real_distribution<> uni_rd(0.0, 1.0); // call uni_rd to get a random number in [0,1)

using namespace std;

void rd_wf(const l_int& dim, double* wf)
{
	for (l_int i = 0; i < dim; i++)
	{
		wf[i] = (uni_rd(gen) - 0.5);
	}
	// normalize
	cblas_dscal(dim, 1.0 / cblas_dnrm2(dim, wf, 1), wf, 1);
}

void rd_wf(const l_int& dim, my_cmplx* wf)
{
	my_cmplx II(0, 1);
	for (l_int i = 0; i < dim; i++)
	{
		wf[i] = (uni_rd(gen) - 0.5) + (uni_rd(gen) - 0.5) * II;
	}
	// normalize
	cblas_zdscal(dim, 1.0 / cblas_dznrm2(dim, wf, 1), wf, 1);
}

void rd_wf_product(const l_int& dim, double* wf)
{
	for (l_int i = 0; i < dim; i++)
	{
		wf[i] = 0;
	}
	l_int idx = l_int(uni_rd(gen) * dim);
	wf[idx] = 1.0;
	//cout << "norm of random product states"  << cblas_dnrm2(dim, wf, 1) << endl;
}

void uni_wf(const l_int& dim, double* wf)
{
	for (l_int i = 0; i < dim; i++)
	{
		wf[i] = 1;
	}
	// normalize
	cblas_dscal(dim, 1.0 / cblas_dnrm2(dim, wf, 1), wf, 1);
}

// a random number function from Anders Sandvik
double ran_num() {
	double ran;
	unsigned long long ran64;
	ifstream ifseed("seed.in");
	ifseed >> ran64;
	ifseed.close();
	unsigned long long irmax, mul64, add64;
	double dmu64;
	irmax = 9223372036854775807;
	mul64 = 2862933555777941757;
	add64 = 1013904243;
	dmu64 = 0.5 / double(irmax);
	ran64 = ran64 * mul64 + add64;
	ran = dmu64 * double(ran64);
	// refresh seed
	ran64 = (ran64 * mul64) / 5 + 5265361;
	ofstream ofseed("seed.in");
	ofseed << ran64;
	ofseed.close();
	return ran;
}

// bitwise operations
int numOfBit1(const l_int& b)
{
	int a = b;
	int cnt = 0;
	while (a != 0)
	{
		++cnt;
		a &= (a - 1);
	}
	return cnt;
}
// find first n '1's of a 
void findBit1(const l_int& a, const int& n, int* b)
{
	int x = 0;
	int i = 0;
	while (x < n) {
		if (((a >> i) & 1) == 1) {
			b[x] = i;
			x++;
		}
		i++;
	}
}

void print_binary(const l_int& a, const int& n)
{
	cout << "  ";
	for (int ix = 0; ix < n; ix++) cout << ((a >> (n - ix - 1)) & 1);
	cout << "  ";
}

int HammingDis(const l_int& a, const l_int& b)
{
	return numOfBit1(a ^ b);
}

void bits_decomposition(const l_int& s, const int& n, const int& size_A, int* sites_A, l_int& a, l_int& b)
{
	// 
	int size_B = n - size_A;
	l_int aux_B = 0;
	//
	a = 0;
	b = 0;
	for (int ia = 0; ia < size_A; ia++)
	{
		a += (((s >> sites_A[ia]) & 1) << ia);
		aux_B += (1 << sites_A[ia]);
	}
	//
	int ib = 0;
	aux_B = (~aux_B);
	for (int i = 0; i < n; i++)
	{
		if ((aux_B >> i) & 1)
		{
			b += ((s >> i) & 1) << ib;
			ib++;
		}
	}
}

l_int nchoosek(const int& n, const int& _k)
{
	if (_k > n) return 0;
	int k = _k < (n - _k) ? _k : (n - _k);
	if (0 == k) return 1;
	if (1 == k) return n;
	double aux = 1.0;
	for (int i = 0; i < k; i++)
	{
		aux *= double(n - i) / double(k - i);
	}
	return (l_int)(aux + 1e-2);
}

//
void NormalizedCopy(const double* f, double* g, const l_int& Dim) {
	double res = 0;
	for (l_int i = 0; i < Dim; i++)
		res += f[i] * f[i];
	res = sqrt(res);
	for (l_int j = 0; j < Dim; j++)
		g[j] = f[j] / res;
}

//
void SetToZero(l_int* tmp, l_int length) {
	for (l_int x = 0; x < length; x++) tmp[x] = 0;
}
void SetToZero(double* f, const l_int dim) {
	for (l_int i = 0; i < dim; i++) f[i] = 0;
}

// ===============================================================================
// Matrix evd (use mkl lapacke function) 
void MatrixEvd(int matrix_layout, char jobz, char uplo, lapack_int n, double* a, lapack_int lda, double* w) {
	LAPACKE_dsyev(matrix_layout, jobz, uplo, n, a, lda, w);
}

// ===============================================================================
// Matrix evd (use mkl lapacke function) 
void DenseMatrixEigenSolver(int matrix_layout, char jobz, char uplo, lapack_int n, double* a, lapack_int lda, double* w)
{
	int eigtype = 1;	// 0, dsyevd; 1, dsyev; 2 dsyevr;
	if (0 == eigtype)
	{
		LAPACKE_dsyev(matrix_layout, jobz, uplo, n, a, lda, w);
	}
	if (1 == eigtype)
	{
		LAPACKE_dsyevd(matrix_layout, jobz, uplo, n, a, lda, w);
	}
	if (2 == eigtype)
	{
		char range = 'A'; // all eigenvalues
		lapack_int* isuppz = new lapack_int[2 * n];
		double abstol = 0;
		LAPACKE_dsyevr(matrix_layout, jobz, range, uplo,
			n, a, lda, NULL, NULL, NULL, NULL,
			abstol, &n, w, a, n, isuppz);
	}
}

void DenseMatrixEigenSolver_FInterface(int matrix_layout, char jobz, char uplo, lapack_int n, double* a, lapack_int lda, double* w)
{
	int eigtype = 0;	// 0, dsyevd; 1, dsyev; 2 dsyevr;
	lapack_int info;
	// dsyev
	if (0 == eigtype)
	{
		lapack_int lwork;
		double wkopt;
		double* work;
		lwork = -1;
		dsyev(&jobz, &uplo, &n, a, &lda, w, &wkopt, &lwork, &info);
		lwork = (int)wkopt;
		cout << "lwork = " << lwork;
		work = (double*)malloc(lwork * sizeof(double));
		// Solve eigenproblem
		dsyev(&jobz, &uplo, &n, a, &lda, w, work, &lwork, &info);
	}
	// dsyevd
	if (1 == eigtype)
	{
		lapack_int lwork, liwork;
		lapack_int* iwork;
		double* work;
		/*
			int iwkopt;
			double wkopt;
			lwork = -1;
			liwork = -1;
			dsyevd(&jobz, &uplo, &n, a, &lda, w, &wkopt, &lwork, &iwkopt, &liwork, &info );
			lwork = (int)wkopt;
			liwork = iwkopt;
		*/
		lwork = 2 * n * n + 6 * n + 1;
		work = (double*)malloc(lwork * sizeof(double));
		liwork = 5 * n + 3;
		iwork = (lapack_int*)malloc(liwork * sizeof(int));
		// Solve eigenproblem
		dsyevd(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, &info);
	}

	// dsyevr 
	if (2 == eigtype)
	{
		lapack_int il, iu, ldz, info, lwork, liwork;
		ldz = lda;
		double abstol, vl, vu;
		lapack_int iwkopt;
		lapack_int* iwork;
		double wkopt;
		double* work;
		lapack_int* isuppz = new lapack_int[n];

		abstol = -1;
		il = 1;
		iu = n;
		lwork = -1;
		liwork = -1;
		dsyevr("Vectors", "Indices", "Upper", &n, a, &lda, &vl, &vu, &il, &iu,
			&abstol, &n, w, a, &ldz, isuppz, &wkopt, &lwork, &iwkopt, &liwork,
			&info);
		lwork = (int)wkopt;
		work = (double*)malloc(lwork * sizeof(double));
		liwork = iwkopt;
		iwork = (lapack_int*)malloc(liwork * sizeof(int));
		cout << "lwork = " << lwork << ", liwork = " << liwork << endl;
		// Solve eigenproblem
		dsyevr("Vectors", "Indices", "Upper", &n, a, &lda, &vl, &vu, &il, &iu,
			&abstol, &n, w, a, &ldz, isuppz, work, &lwork, iwork, &liwork,
			&info);
	}

	/* Check for convergence */
	if (info > 0) {
		printf("The algorithm failed to compute eigenvalues.\n");
		exit(1);
	}
}

// Matrix svd (calling lapacke matrix svd)
void MatrixSvd(int matrix_layout, char jobu, char jobvt, lapack_int m, lapack_int n, double* a, lapack_int lda, double* s, double* u, lapack_int ldu, double* vt, lapack_int ldvt) 
{
	double* superb = new double[min(m, n)];
	LAPACKE_dgesvd(matrix_layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}

void MatrixSvd(int matrix_layout, char jobu, char jobvt, lapack_int m, lapack_int n, my_cmplx* a, lapack_int lda, double* s, my_cmplx* u, lapack_int ldu, my_cmplx* vt, lapack_int ldvt) 
{
	double* superb = new double[min(m, n)];
	LAPACKE_zgesvd(matrix_layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}

//
double VN_entropy(const l_int& dim, double* wf) {
	/*	double* log_wf = new double[dim];
		vdLn(dim, wf, log_wf);
		double aux = cblas_ddot(dim, wf, 1, log_wf, 1);
		delete[] log_wf;
		return -aux;
	*/
	// in case log(0) = nan might be a problem
	double aux = 0;
	for (l_int i = 0; i < dim; i++) {
		if (wf[i] > 1e-32) {
			aux += -wf[i] * log(wf[i]);
		}
	}
	return aux;
}

void GetParaFromInput_int(const char* fname, const char* string_match, int& para) {
	FILE* f_in = fopen(fname, "r");
	char testchar[40], line[80];
	sprintf(testchar, string_match);
	int len = strlen(testchar);
	while (fgets(line, 200, f_in) != NULL) {
		if (strncmp(line, testchar, len) == 0)
		{
			char* p = strtok(line, "=");
			stringstream ss;
			p = strtok(NULL, "=");
			ss << p;
			ss >> para;
			std::cout << "GetParaFromInput: " << string_match << " " << para << std::endl;
			break;
		}
	}
	fclose(f_in);
	f_in = NULL;
}

void GetParaFromInput_real(const char* fname, const char* string_match, double& para) {
	FILE* f_in = fopen(fname, "r");
	char testchar[40], line[80];
	sprintf(testchar, string_match);
	int len = strlen(testchar);
	while (fgets(line, 200, f_in) != NULL) {
		if (strncmp(line, testchar, len) == 0)
		{
			char* p = strtok(line, "=");
			stringstream ss;
			p = strtok(NULL, "=");
			ss << p;
			ss >> para;
			std::cout << "GetParaFromInput: " << string_match << " " << para << std::endl;
			break;
		}
	}
	fclose(f_in);
	f_in = NULL;
}

void GetParaFromInput_char(const char* fname, const char* string_match, char& para) {
	FILE* f_in = fopen(fname, "r");
	char testchar[40], line[80];
	sprintf(testchar, string_match);
	int len = strlen(testchar);
	while (fgets(line, 200, f_in) != NULL) {
		if (strncmp(line, testchar, len) == 0)
		{
			char* p = strtok(line, "=");
			stringstream ss;
			p = strtok(NULL, "=");
			ss << p;
			ss >> para;
			std::cout << "GetParaFromInput: " << string_match << " " << para << std::endl;
			break;
		}
	}
	fclose(f_in);
	f_in = NULL;
}

void Vec_fwrite_double(const char* fname, double* data, const int& dsize)
{
	FILE* f_out;
	f_out = fopen(fname, "wb");
	fwrite(data, sizeof(double), dsize, f_out);
	fclose(f_out);
}

void Vec_fread_double(const char* fname, double* data, const int& dsize)
{
	FILE* f_in;
	f_in = fopen(fname, "rb");
	fread(data, sizeof(double), dsize, f_in);
	fclose(f_in);
}

double my_vec_dot(const l_int& dim, double* x, const l_int incx, double* y, const l_int incy)
{
	return cblas_ddot(dim, x, incx, y, incy);
}
my_cmplx my_vec_dot(const l_int& dim, my_cmplx* x, const l_int incx, my_cmplx* y, const l_int incy)
{
	my_cmplx aux;
	cblas_zdotc_sub(dim, x, incx, y, incy, &aux);
	return aux;
}

void Lanczos(const l_int& dim, double* vals, l_int* cols, l_int* PointerBE, const l_int& M, char job_vec, double* alpha, double* l_vecs, double* l_eigvecs)
{
	sparse_matrix_t A;
	mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, dim, dim, PointerBE, PointerBE + 1, cols, vals);
	struct matrix_descr descrA;
	descrA.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
	descrA.mode = SPARSE_FILL_MODE_UPPER;
	descrA.diag = SPARSE_DIAG_NON_UNIT;
	mkl_sparse_optimize(A);
	// 
	int steps = M;
	// Lanczos matrix elements
	double* beta = new double[steps];		// beta[0] is unused

	// if we do not need lanzcos vecters, only 3 vectors are used
	if ('N' == job_vec)
	{
		double* phi0 = new double[dim];
		double* phi1 = new double[dim];
		double* phi2 = new double[dim];
		// copy initial vector form 1st dim elements form l_vecs 
		cblas_dcopy(dim, l_vecs, 1, phi0, 1);

		// alpha[0], beta[1] -- beta[0] is not required
		mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, A, descrA, phi0, 0, phi1);
		alpha[0] = cblas_ddot(dim, phi0, 1, phi1, 1);
		cblas_daxpy(dim, -alpha[0], phi0, 1, phi1, 1);
		beta[1] = cblas_dnrm2(dim, phi1, 1);
		cblas_dscal(dim, 1.0 / beta[1], phi1, 1);

		// alpha[m], beta[m+1], m = [0,steps-2]
		for (int m = 1; m < steps - 1; m++)
		{
			// alpha[m], m = [0,steps-2]
			mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, A, descrA, phi1, 0, phi2);
			alpha[m] = cblas_ddot(dim, phi1, 1, phi2, 1);
			// beta[m+1], m = [0,steps-2]
			cblas_daxpy(dim, -alpha[m], phi1, 1, phi2, 1);
			cblas_daxpy(dim, -beta[m], phi0, 1, phi2, 1);
			beta[m + 1] = cblas_dnrm2(dim, phi2, 1);
			cblas_dscal(dim, 1.0 / beta[m + 1], phi2, 1);
			cblas_dcopy(dim, phi1, 1, phi0, 1);
			cblas_dcopy(dim, phi2, 1, phi1, 1);
		}
		// alpha[m], m = steps-1
		int m = steps - 1;
		mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, A, descrA, phi1, 0, phi2);
		alpha[m] = cblas_ddot(dim, phi1, 1, phi2, 1);

		LAPACKE_dsteqr(LAPACK_ROW_MAJOR, 'N', steps, alpha, &beta[0] + 1, NULL, steps);

		delete[]phi0;
		delete[]phi1;
		delete[]phi2;
	}

	// lanzcos vecters are required 
	if ('V' == job_vec)
	{
		// alpha[0], beta[1] -- beta[0] is not required
		mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, A, descrA, &l_vecs[0], 0, &l_vecs[dim]);
		alpha[0] = cblas_ddot(dim, &l_vecs[0], 1, &l_vecs[dim], 1);
		cblas_daxpy(dim, -alpha[0], &l_vecs[0], 1, &l_vecs[dim], 1);
		beta[1] = cblas_dnrm2(dim, &l_vecs[dim], 1);
		cblas_dscal(dim, 1.0 / beta[1], &l_vecs[dim], 1);

		// alpha[0], beta[1] -- beta[0] is not required
		for (int m = 1; m < steps-1; m++)
		{
			// alpha[m], m = [0,steps-2]
			mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, A, descrA, &l_vecs[m * dim], 0, &l_vecs[(m + 1) * dim]);
			alpha[m] = cblas_ddot(dim, &l_vecs[m * dim], 1, &l_vecs[(m + 1) * dim], 1);
			// beta[m+1], m = [0,steps-2]
			cblas_daxpy(dim, -alpha[m], &l_vecs[m * dim], 1, &l_vecs[(m + 1) * dim], 1);
			cblas_daxpy(dim, -beta[m], &l_vecs[(m - 1) * dim], 1, &l_vecs[(m + 1) * dim], 1);
			beta[m + 1] = cblas_dnrm2(dim, &l_vecs[(m + 1) * dim], 1);
			cblas_dscal(dim, 1.0 / beta[m + 1], &l_vecs[(m + 1) * dim], 1);
		}
		// alpha[m], m = steps-1
		int m = steps - 1;
		double* tmpvec = new double[dim];
		mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, A, descrA, &l_vecs[m * dim], 0, tmpvec);
		alpha[m] = cblas_ddot(dim, &l_vecs[m * dim], 1, tmpvec, 1);
		delete[]tmpvec;

		LAPACKE_dsteqr(LAPACK_ROW_MAJOR, 'I', steps, alpha, &beta[0] + 1, l_eigvecs, steps);

		// test wf0
		/*
		double* wf0 = new double[dim];
		double* wf1 = new double[dim];
		for (l_int i = 0; i < dim; i++)
		{
			wf0[i] = my_vec_dot(M, &l_eigvecs[0], M, &l_vecs[i], dim);
		}
		double aux0 = cblas_dnrm2(dim, wf0, 1);
		cout << "aux0: " << aux0 << endl;
		cblas_dscal(dim, 1.0 / aux0, wf0, 1);
		mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, A, descrA, wf0, 0, wf1);
		cout << "<wf0|H|wf0> = " << setprecision(14) << cblas_ddot(dim, wf0, 1, wf1, 1) << endl;
		delete[]wf0;
		delete[]wf1;
		*/
	}
	delete[]beta;
	mkl_sparse_destroy(A);
}

void Lanczos(const l_int& dim, my_cmplx* vals, l_int* cols, l_int* PointerBE, const l_int& M, char job_vec,
	double* alpha, my_cmplx* l_vecs, double* l_eigvecs)
{
	sparse_matrix_t A;
	mkl_sparse_z_create_csr(&A, SPARSE_INDEX_BASE_ZERO, dim, dim, PointerBE, PointerBE + 1, cols, vals);
	struct matrix_descr descrA;
	descrA.type = SPARSE_MATRIX_TYPE_HERMITIAN;
	descrA.mode = SPARSE_FILL_MODE_UPPER;
	descrA.diag = SPARSE_DIAG_NON_UNIT;
	mkl_sparse_optimize(A);
	// 
	int steps = M;
	// Lanczos matrix elements
	double* beta = new double[steps];		// beta[0] is unused

	// if we do not need lanzcos vecters, only 3 vectors are used
	if ('N' == job_vec)
	{
		my_cmplx* phi0 = new my_cmplx[dim];
		my_cmplx* phi1 = new my_cmplx[dim];
		my_cmplx* phi2 = new my_cmplx[dim];
		// copy initial vector form 1st dim elements form l_vecs 
		cblas_zcopy(dim, l_vecs, 1, phi0, 1);

		// alpha[0], beta[1] -- beta[0] is not required
		mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, A, descrA, phi0, 0, phi1);
		my_cmplx aux;
		cblas_zdotc_sub(dim, phi0, 1, phi1, 1, &aux);
		alpha[0] = aux.real();
		my_cmplx minus_aux = -aux;
		cblas_zaxpy(dim, &minus_aux, phi0, 1, phi1, 1);
		beta[1] = cblas_dznrm2(dim, phi1, 1);
		cblas_zdscal(dim, 1.0 / beta[1], phi1, 1);

		// alpha[m], beta[m+1], m = [0,steps-2]
		for (int m = 1; m < steps - 1; m++)
		{
			// alpha[m], m = [0,steps-2]
			mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, A, descrA, phi1, 0, phi2);
			cblas_zdotc_sub(dim, phi1, 1, phi2, 1, &aux);
			alpha[m] = aux.real();
			// beta[m+1], m = [0,steps-2]
			minus_aux = -aux;
			cblas_zaxpy(dim, &minus_aux, phi1, 1, phi2, 1);
			my_cmplx minus_beta = -beta[m];
			cblas_zaxpy(dim, &minus_beta, phi0, 1, phi2, 1);
			beta[m + 1] = cblas_dznrm2(dim, phi2, 1);
			cblas_zdscal(dim, 1.0 / beta[m + 1], phi2, 1);
			cblas_zcopy(dim, phi1, 1, phi0, 1);
			cblas_zcopy(dim, phi2, 1, phi1, 1);
		}
		// alpha[m], m = steps-1
		int m = steps - 1;
		mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, A, descrA, phi1, 0, phi2);
		cblas_zdotc_sub(dim, phi1, 1, phi2, 1, &aux);
		alpha[m] = aux.real();
	
		LAPACKE_dsteqr(LAPACK_ROW_MAJOR, 'N', steps, alpha, &beta[0] + 1, NULL, steps);

		delete[]phi0;
		delete[]phi1;
		delete[]phi2;
	}

	// lanzcos vecters are required 
	//my_cmplx* alpha_cmplx = new my_cmplx[dim];
	if ('V' == job_vec)
	{
		// alpha[0], beta[1] -- beta[0] is not required
		my_cmplx aux;
		mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, A, descrA, &l_vecs[0], 0, &l_vecs[dim]);
		cblas_zdotc_sub(dim, &l_vecs[0], 1, &l_vecs[dim], 1, &aux);
		alpha[0] = aux.real();
		my_cmplx minus_aux = -aux;
		cblas_zaxpy(dim, &minus_aux, &l_vecs[0], 1, &l_vecs[dim], 1);
		beta[1] = cblas_dznrm2(dim, &l_vecs[dim], 1);
		cblas_zdscal(dim, 1.0 / beta[1], &l_vecs[dim], 1);

		// alpha[m], beta[m+1], m = [0,steps-2]
		for (int m = 1; m < steps - 1; m++)
		{
			// alpha[m], m = [0,steps-2]
			mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, A, descrA, &l_vecs[m * dim], 0, &l_vecs[(m + 1) * dim]);
			cblas_zdotc_sub(dim, &l_vecs[m * dim], 1, &l_vecs[(m + 1) * dim], 1, &aux);
			alpha[m] = aux.real();
			// alpha[m+1], m = [0,steps-2]
			minus_aux = -aux;
			cblas_zaxpy(dim, &minus_aux, &l_vecs[m * dim], 1, &l_vecs[(m + 1) * dim], 1);
			my_cmplx minus_beta = -beta[m];
			cblas_zaxpy(dim, &minus_beta, &l_vecs[(m - 1) * dim], 1, &l_vecs[(m + 1) * dim], 1);
			beta[m + 1] = cblas_dznrm2(dim, &l_vecs[(m + 1) * dim], 1);
			cblas_zdscal(dim, 1.0 / beta[m + 1], &l_vecs[(m + 1) * dim], 1);
		}
		// alpha[m], m = steps-1
		int m = steps - 1;
		my_cmplx* tmpvec = new my_cmplx[dim];
		mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, A, descrA, &l_vecs[m * dim], 0, tmpvec);
		cblas_zdotc_sub(dim, &l_vecs[m * dim], 1, tmpvec, 1, &aux);
		delete[]tmpvec;
		alpha[m] = aux.real();

		//LAPACKE_dsteqr(LAPACK_ROW_MAJOR, 'I', steps, alpha, &beta[0] + 1, l_eigvecs, steps);
		//for (int m = 0; m < steps; m++) alpha[m] = alpha_cmplx[m].real();
		//LAPACKE_zsteqr(LAPACK_ROW_MAJOR, 'I', steps, alpha, &beta[0] + 1, l_eigvecs, steps);
		LAPACKE_dsteqr(LAPACK_ROW_MAJOR, 'I', steps, alpha, &beta[0] + 1, l_eigvecs, steps);

		// test wf0
		/*
		my_cmplx* wf0 = new my_cmplx[dim];
		my_cmplx* wf1 = new my_cmplx[dim];
		for (l_int i = 0; i < dim; i++)
		{
			//wf0[i] = my_vec_dot(M, &l_eigvecs[0], M, &l_vecs[i], dim);
			wf0[i] = 0;
			for (int m = 0; m < M; m++)
			{
				wf0[i] += l_eigvecs[m * M] * l_vecs[m * dim + i];
			}
		}
		double aux0 = cblas_dznrm2(dim, wf0, 1);
		cout << "aux0: " << endl;
		cblas_zdscal(dim, 1.0 / aux0, wf0, 1);
		mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, A, descrA, wf0, 0, wf1);
		cout << "<wf0|H|wf0> = " << setprecision(14) << my_vec_dot(dim, wf0, 1, wf1, 1) << endl;
		delete[]wf0;
		delete[]wf1;
		*/
	}

	delete[]beta;
	mkl_sparse_destroy(A);
}


// extra vector operations 
// y = a*x
void myvec_ax(const l_int& dim, const my_cmplx& a, my_cmplx* x, my_cmplx* y)
{
	for (l_int i = 0; i < dim; i++)
	{
		y[i] = a * x[i];
	}
}
void myvec_ax(const l_int& dim, const double& a, double* x, double* y)
{
	for (l_int i = 0; i < dim; i++)
	{
		y[i] = a * x[i];
	}
}
// a = real(x)
void myvec_real(const l_int& dim, my_cmplx* x, double* a)
{
	for (l_int i = 0; i < dim; i++)
	{
		a[i] = x[i].real();
	}
}
// a = imag(x)
void myvec_imag(const l_int& dim, my_cmplx* x, double* a)
{
	for (l_int i = 0; i < dim; i++)
	{
		a[i] = x[i].imag();
	}
}
// a = x + i*y, x and y are real vectors
void myvec_xpiy(const l_int& dim, double* x, double* y, my_cmplx* a)
{
	my_cmplx II(0, 1);
	for (l_int i = 0; i < dim; i++)
	{
		a[i] = x[i] + II*y[i];
	}
}