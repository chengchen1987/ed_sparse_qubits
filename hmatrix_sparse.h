#pragma once

using namespace std;
//#include <omp.h>

template <class Type>
l_int Ham_Qubits<Type>::SparaseMat_Counts()
{
	l_int counts = 0;
	for (l_int k = 0; k < Dim; k++)
	{
		l_int State_k = basis->get_state(k);
		// diagonal
		counts++;
		// off-diagonal
		for (int ib = 0; ib < N_hops; ib++)
		{
			int i = std::get<0>(hops[ib]);
			int j = std::get<1>(hops[ib]);
			// check onsite states 
			int k_i = ((State_k >> i) & 1);
			int k_j = ((State_k >> j) & 1);
			if ((k_i) && (!k_j))
			{
				counts++;
			}
		}
	}
	return counts;
}

// build sparse matrix of H 
// be carefull, we only make hopping pairs with i < j
template <class Type>
void Ham_Qubits<Type>::SparseMat_Build()
{
	// get the number of nonzero elements
	nnz = SparaseMat_Counts();
	cout << "No. of nonzeros: nnz = " << nnz << endl;
	double mem = nnz * 8 / 1e9;
	cout << "Estimated memory cost of the Hamiltonian matrix in sparse form: " << mem << " GB." << endl;

	SMat.SparseMat_Init(Dim, nnz);
	
	/*
	SMat_vals = new Type[nnz];
	SMat_cols = new l_int[nnz];
	SMat_PointerBE = new l_int[Dim + 1];
	*/

	l_int counts = 0;
	for (l_int k = 0; k < Dim; k++)
	{
		l_int State_k = basis->get_state(k);

		// diagonal term
		// \sum_{i} mu[i] * sigma_i^+ * sigma_i^-
		double aux_diag = 0;
		for (int i = 0; i < LatticeSize; i++)
		{
			int k_i = ((State_k >> i) & 1);
			aux_diag += mu[i] * k_i;
		}
		SMat.vals[counts] = aux_diag;
		SMat.cols[counts] = k;
		SMat.Pointer_BE[k] = counts;
		counts++;
		// off-diagonal term
		// \sum_{i<j} Jij[i,j] * sigma_i^+ * sigma_j^- 
		for (int ib = 0; ib < N_hops; ib++)
		{
			int i = std::get<0>(hops[ib]);
			int j = std::get<1>(hops[ib]);

			// sigma_i^+ sigma_j^- 
			int k_i = ((State_k >> i) & 1);
			int k_j = ((State_k >> j) & 1);
			if ((k_i) && (!k_j))
			{
				l_int State_l = State_k ^ (1 << i) ^ (1 << j);
				l_int l = basis->get_index(State_l);
				SMat.cols[counts] = l;
				SMat.vals[counts] = std::get<2>(hops[ib]);
				counts++;
			}
		}
		SMat.Pointer_BE[k + 1] = counts;
	}
	cout << "counts: " << counts << endl;
}

template <class Type>
void Ham_Qubits<Type>::SparseMat_Eigs()
{
	// 'V'/'I' with/without eigenvectors
	char job_vec = 'V';

	int M = 200;

	if ('N' == job_vec)
	{
		Type* l_vecs = new Type[Dim];
		rd_wf(Dim, l_vecs);
		
		double* alpha = new double[M];
		Lanczos(Dim, SMat.vals, SMat.cols, SMat.Pointer_BE, M, job_vec, alpha, l_vecs, NULL);
		E_min = alpha[0];
		cout << "M = " << M << ", E_min = " << setprecision(14) << E_min << endl;

		std::ofstream ofe("E0.dat");
		ofe << std::setprecision(14) << alpha[0];
		ofe << std::endl;
		ofe.close();

		delete[]alpha;
		delete[]l_vecs;
	}
	else if ('V' == job_vec)
	{
		Type* l_vecs = new Type[Dim * M]; 
		rd_wf(Dim, l_vecs);
		
		Type* l_eigvecs = new Type[M * M];			// eigenvectors in Lanczos basis
		double* alpha = new double[M];
		Lanczos(Dim, SMat.vals, SMat.cols, SMat.Pointer_BE, M, job_vec, alpha, l_vecs, l_eigvecs);
		E_min = alpha[0];
		cout << "M = " << M << ", E_min = " << setprecision(14) << E_min << endl;

		std::ofstream ofe("E_min.dat");
		ofe << std::setprecision(14) << alpha[0];
		ofe << std::endl;
		ofe.close();

		// groundstate wavefunction
		Type* wf0 = new Type[Dim];
		for (int i = 0; i < Dim; i++)
		{
			wf0[i] = my_vec_dot(M, &l_eigvecs[0], M, &l_vecs[i], Dim);
		}
		// check wf0
		//cout << "<wf0|U[:,0]> = " << cblas_ddot(Dim, wf0, 1, &DMat[0], Dim) << endl;
		delete[]wf0;

		delete[]alpha;
		delete[]l_vecs;
		delete[]l_eigvecs;
	}
	else
	{
		cout << "Error! job_vec for Lanczos method must be either \'N\' or \'V\'!" << endl;
		exit(666);
	}
}

template <class Type>
void Ham_Qubits<Type>::SparseMat_E_max()
{
	// 'V'/'I' with/without eigenvectors
	char job_vec = 'N';
	int M = 200;

	// compute E_max by 
	// compute E_0 of -H
	SparseMat<Type> smat_oppo;
	smat_oppo.SparseMat_Init(Dim, SMat.mat_nnz);
	for (l_int i = 0; i < SMat.mat_nnz; i++) {
		smat_oppo.vals[i] = - SMat.vals[i];
		smat_oppo.cols[i] = SMat.cols[i];
	}
	for (int i = 0; i < SMat.mat_dim + 1; i++) {
		smat_oppo.Pointer_BE[i] = SMat.Pointer_BE[i];
	}
	//

	if ('N' == job_vec)
	{
		Type* l_vecs = new Type[Dim];
		rd_wf(Dim, l_vecs);

		double* alpha = new double[M];
		Lanczos(Dim, smat_oppo.vals, smat_oppo.cols, smat_oppo.Pointer_BE, M, job_vec, alpha, l_vecs, NULL);
		E_max = -alpha[0];
		cout << "M = " << M << ", E_max = " << setprecision(14) << E_max << endl;

		std::ofstream ofe("E_max.dat");
		ofe << std::setprecision(14) << alpha[0];
		ofe << std::endl;
		ofe.close();

		delete[]alpha;
		delete[]l_vecs;
	}
	smat_oppo.SparseMat_Clear();
}