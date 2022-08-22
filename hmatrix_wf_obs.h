template <class Type>
void Ham_Qubits<Type>::Cal_ni_wfsq(double* wfsq, double* ni)
{
	for (int i = 0; i < LatticeSize; i++) ni[i] = 0;
	for (l_int k = 0; k < Dim; k++)
	{
		l_int State_k = basis->get_state(k);
		for (int i = 0; i < LatticeSize; i++)
		{
			ni[i] += ((State_k >> i) & 1) ? wfsq[k] : 0;
		}
	}
}

template <class Type>
void Ham_Qubits<Type>::Cal_ninj_wfsq(double* wfsq, double* ninj)
{
	for (int i = 0; i < LatticeSize * LatticeSize; i++) ninj[i] = 0;
	for (l_int k = 0; k < Dim; k++) {
		l_int State_k = basis->get_state(k);
		for (int i = 0; i < LatticeSize; i++)
			for (int j = i + 1; j < LatticeSize; j++)
			{
				ninj[i * LatticeSize + j] += ((State_k >> i) & (State_k >> i) & 1) ? wfsq[k] : 0;
				ninj[j * LatticeSize + i] = ninj[i * LatticeSize + j];
			}
	}
}

template <class Type>
void Ham_Qubits<Type>::Cal_PIr_wfsq(double* wfsq, double* HamD_pk, double* PI_r)
{
	for (int r = 0; r < LatticeSize + 1; r++)
	{
		PI_r[r] = 0;
		for (l_int k = 0; k < Dim; k++)
		{
			if (1e-10 > abs(r - HamD_pk[k]))
			{
				PI_r[r] += wfsq[k];
			}
		}
	}
}

template <class Type>
double Ham_Qubits<Type>::Cal_Pmax_wfsq(double* wfsq)
{
	double aux = *max_element(wfsq, wfsq + Dim);
	return aux;
}

template <class Type>
double Ham_Qubits<Type>::Cal_IPR_wfsq(double* wfsq)
{
	//double* auxvec = new double[Dim];
	double IPR = 1.0 / cblas_dnrm2(Dim, wfsq, 1);	// dnrm2 = sqrt(sum(wfsq^2))
	IPR *= IPR;
	//delete[]auxvec;
	return IPR;
}

template <class Type>
double Ham_Qubits<Type>::Cal_SE_wfsq(double* wfsq)
{
	return VN_entropy(Dim, wfsq);
}

template <class Type>
double Ham_Qubits<Type>::Cal_EE_wf(double* wf, const int& size_A, int* sites_A)
{
	int size_B = LatticeSize - size_A;
	l_int dim_A = 1 << size_A;
	l_int dim_B = 1 << size_B;
	l_int dim_2d = dim_A * dim_B;
	double* wf_2d = new double[dim_2d];
	for (l_int k = 0; k < Dim; k++)
	{
		l_int s = basis->get_state(k);
		int ind_a, ind_b;
		bits_decomposition(s, LatticeSize, size_A, sites_A, ind_a, ind_b);
		//
		wf_2d[ind_a + ind_b * dim_A] = wf[k];
	}
	double* lambda = new double[max(dim_A, dim_B)];
	MatrixSvd(LAPACK_ROW_MAJOR, 'N', 'N', dim_B, dim_A, wf_2d, dim_A, lambda, NULL, dim_A, NULL, dim_B);
	// singular value is the sqrt of eigenvalues of reduced density martrix
	vdSqr(dim_A, lambda, lambda);
	double ee = VN_entropy(dim_A, lambda);
	delete[]wf_2d;
	delete[]lambda;
	return ee;
}

template <class Type>
double Ham_Qubits<Type>::Cal_EE_wf(my_cmplx* wf, const int& size_A, int* sites_A)
{
	int size_B = LatticeSize - size_A;
	l_int dim_A = 1 << size_A;
	l_int dim_B = 1 << size_B;
	l_int dim_2d = dim_A * dim_B;
	my_cmplx* wf_2d = new my_cmplx[dim_2d];
	for (l_int k = 0; k < Dim; k++)
	{
		l_int s = basis->get_state(k);
		int ind_a, ind_b;
		bits_decomposition(s, LatticeSize, size_A, sites_A, ind_a, ind_b);
		//
		wf_2d[ind_a + ind_b*dim_A] = wf[k];
	}
	double* lambda = new double[max(dim_A, dim_B)];
	MatrixSvd(LAPACK_ROW_MAJOR, 'N', 'N', dim_B, dim_A, wf_2d, dim_A, lambda, NULL, dim_A, NULL, dim_B);

	// singular value is the sqrt of eigenvalues of reduced density martrix
	vdSqr(dim_A, lambda, lambda);
	double ee = VN_entropy(dim_A, lambda);
	delete[]wf_2d;
	delete[]lambda;
	return ee;
}