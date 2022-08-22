#pragma once

template <class Type>
void Ham_Qubits<Type>::SparseMat_Evolution()
{
	my_cmplx II(0, 1);

	int prt_ind = 0;
	l_int Index_p;

	double epsilon = 0.5;
	double E_band = E_max - E_min;
	// read/generate initial product state  
	char fname[80];
	sprintf(fname, "input_IniStateInfo.in");
	ifstream fin(fname, ios::in);
	l_int State_p;
	// Calculate product-state spectrum 
	double* pspec = new double[Dim];
	Cal_product_spectrum(pspec);
	// read initial state from input file
	if (fin)
	{
		double epsilon_read;
		fin >> Index_p;
		//fin >> State_p;
		//Index_p = basis->get_index(State_p);
		fin.close();
	}
	// generate a random initial state near the target energy
	else
	{
		vector <pair<double, l_int> > Fock_E_n;  // stores E_fock, and index of this Fock state
		for (l_int p = 0; p < Dim; p++)
		{
			pair<double, l_int> aux = make_pair(pspec[p], p);
			Fock_E_n.push_back(aux);
		}
		std::sort(Fock_E_n.begin(), Fock_E_n.end());
		// product-state spectrum in energy density
		double* sorted_Fock_epsi = new double[Dim];
		for (l_int k = 0; k < Dim; k++) {
			sorted_Fock_epsi[k] = (Fock_E_n[k].first - E_min) / E_band;
		}
		//
		int p0 = Get_TargetFock_left((epsilon - 0.02) * E_band + E_min, Fock_E_n);
		int p1 = Get_TargetFock_right((epsilon + 0.02) * E_band + E_min, Fock_E_n);
		int ps;
		if (p0 < p1)
		{
			ps = rand() % (p1 - p0) + p0;
		}
		else
		{
			ps = rand() % Dim;
			
		}
		Index_p = Fock_E_n[ps].second;
	}

	// print intial state information
	ofstream fout(fname);
	State_p = basis->get_state(Index_p);
	fout << Index_p << endl;
	fout << State_p << endl;	// this is a binary number
	fout << setprecision(14) << (pspec[Index_p] - E_min) / E_band << endl;
	fout.close();
	
	cout << "initial state: " << endl;
	print_binary(basis->get_state(Index_p), LatticeSize);
	
	/*
	double epsilon = 0.5;
	char fini[80];
	sprintf(fini, "IniStateInfo_state_ind%d.in", prt_ind);
	ofstream ofini(fini);
	ofini << p_state << endl;	// this is a binary number
	ofini << setprecision(14) << (E_real - E_min) / E_band << endl;
	ofini.close();
	*/

	// print time vec to file
	Vec_fwrite_double("evo_time_vec.bin", Params.time_vec, Params.evo_time_steps);

	//
	Dyn_DataStruc dyn_data;
	dyn_data.Initialize(Params, Dim, prt_ind);
	// once State_p is determined, get hamming distances <p|k> for all k in product basis
	if ('y' == Params.evo_PI_r)
	{
		for (l_int k = 0; k < Dim; k++)
		{
			dyn_data.HamD_k[k] = HammingDis(basis->get_state(Index_p), basis->get_state(k));
		}
	}

	// perform real time evolution
	{
		cout << "Real time evolution (real H):" << endl;
		if (1 == Params.evo_type)
		{
			cout << "evolution type: RK4" << endl;
		}
		if (0 == Params.evo_type)
		{
			cout << "evolution type: Lanczos, " << "No. of Lanczos vectors: " << Params.evo_Lanczos_M << endl;
		}

		int nts = Params.evo_time_steps;

		my_cmplx* wf0 = new my_cmplx[Dim];
		my_cmplx* wf1 = new my_cmplx[Dim];

		// set initial state
		for (l_int i = 0; i < Dim; i++) wf0[i] = 0;
		wf0[Index_p] = 1;

		for (int it = 0; it < nts; it++)
		{
			double tau = Params.time_vec[it + 1] - Params.time_vec[it];
			// print <wf1|H|wf1>
			double engt = SparseMat_Inner(SMat, wf0);
			cout << "it = " << it << ", (eng[it] - eng[0]) = " << setprecision(14) << (engt - pspec[Index_p]) << endl;
			// compute obs
			double* wf_sq = new double[Dim];
			vzAbs(Dim, wf0, wf_sq);
			vdPowx(Dim, wf_sq, 2, wf_sq);
			//
			if ('y' == Params.evo_ni)
			{
				Cal_ni_wfsq(wf_sq, &dyn_data.ni_t[it * LatticeSize]);
			}
			if ('y' == Params.evo_ninj)
			{
				Cal_ninj_wfsq(wf_sq, &dyn_data.ninj_t[it * LatticeSize * LatticeSize]);
			}
			if ('y' == Params.evo_IPR)
			{
				dyn_data.IPR_t[it] = Cal_IPR_wfsq(wf_sq);
			}
			if ('y' == Params.evo_SE)
			{
				dyn_data.SE_t[it] = Cal_SE_wfsq(wf_sq);
			}
			if ('y' == Params.evo_PI_r)
			{
				Cal_PIr_wfsq(wf_sq, dyn_data.HamD_k, &dyn_data.PI_r_t[it * (LatticeSize + 1)]);
			}
			if ('y' == Params.evo_EE)
			{
				dyn_data.EE_t[it] = Cal_EE_wf(wf0, Params.EE_size_A, Params.EE_sites_A);
			}
			if ('y' == Params.evo_Pmax)
			{
				dyn_data.Pmax_t[it] = Cal_Pmax_wfsq(wf_sq);
			}

			// compute |wf0> = exp(-i*H*(tau))|wf0>
			if (it < (nts - 1))
			{
				if (1 == Params.evo_type)
				{
					RK4_onestep(SMat, -II * tau, wf0, wf1);
				}
				if (0 == Params.evo_type)
				{
					int M = Params.evo_Lanczos_M;
					SparseMat<my_cmplx> smat_cmplx;
					smat_cmplx.SparseMat_Init(Dim, SMat.mat_nnz);
					smat_cmplx.SparseMat_Copy_real2complex(SMat, Dim, SMat.mat_nnz);
					// be careful! use tau (not -1i*tau) when calling this function
					Lanczos_evo_onestep(smat_cmplx, M, tau, wf0, wf1);
					smat_cmplx.SparseMat_Clear();
				}

				cblas_zcopy(SMat.mat_dim, wf1, 1, wf0, 1);
			}
			delete[]wf_sq;
		}
		delete[]wf0;
		delete[]wf1;

		// write obs
		dyn_data.PrintDynResults(Params);
		//
		dyn_data.ReleaseSpace(Params);
	}

}

template <class Type>
void Ham_Qubits<Type>::DenseMat_Evolution()
{
	// get dense form
	l_int dlen = Dim * Dim;
	double mem = dlen * 8 / 1e9;
	cout << "Estimated memory cost of the Hamiltonian matrix in dense form: " << mem << " GB." << endl;

	SMat.denseform = new double[dlen];
	for (l_int i = 0; i < dlen; i++) SMat.denseform[i] = 0;
	SMat.SparseMat_ToDense();
	double* spec = new double[Dim];

	char evd_jobz = 'V';
	//DenseMatrixEigenSolver_FInterface(LAPACK_ROW_MAJOR, evd_jobz, 'U', Dim, SMat.denseform, Dim, spec);
	MatrixEvd(LAPACK_ROW_MAJOR, evd_jobz, 'U', Dim, SMat.denseform, Dim, spec);
	cout << "E[0] from Dense Matrix EVD: " << spec[0] << endl;

	int nts = 400;
	double evo_time_max = 200;
	double* time_vec = new double[nts];
	double dt = evo_time_max / nts;
	for (int i = 0; i < nts; i++)
	{
		time_vec[i] = i * evo_time_max / nts;
	}
	double* nit = new double[nts * LatticeSize];

	l_int p = 5278;
	int t_len = nts;
	//#pragma omp parallel for schedule(dynamic)
	for (int t_slice = 0; t_slice < t_len; t_slice++) {
		double* ket_real = new double[Dim];
		double* ket_imag = new double[Dim];
		for (int k = 0; k < Dim; k++) {
			ket_real[k] = cos(-time_vec[t_slice] * spec[k]);
			ket_imag[k] = sin(-time_vec[t_slice] * spec[k]);
		}
		vdMul(Dim, ket_real, &SMat.denseform[p * Dim], ket_real);
		vdMul(Dim, ket_imag, &SMat.denseform[p * Dim], ket_imag);
		double num_1 = 1.0;
		double num_0 = 0.0;
		double* ket_real_1 = new double[Dim];
		double* ket_imag_1 = new double[Dim];
		cblas_dgemv(CblasRowMajor, CblasNoTrans, Dim, Dim, 1, SMat.denseform, Dim, ket_real, 1, 0, ket_real_1, 1);
		cblas_dgemv(CblasRowMajor, CblasNoTrans, Dim, Dim, 1, SMat.denseform, Dim, ket_imag, 1, 0, ket_imag_1, 1);
		//vdPowx(Dim, ket_real_1, 2, ket_real_1);
		//vdPowx(Dim, ket_imag_1, 2, ket_imag_1);
		vdSqr(Dim, ket_real_1, ket_real_1);
		vdSqr(Dim, ket_imag_1, ket_imag_1);

		// print <wf1|H|wf1>
		double engt = SparseMat_Inner(SMat, ket_real_1);
		cout << "it = " << t_slice << ", eng[it] = " << setprecision(14) << engt << endl;

		vdAdd(Dim, ket_real_1, ket_imag_1, ket_real_1);

		// obs

		// compute obs
		Get_ni(ket_real_1, &nit[t_slice * LatticeSize]);

		delete[]ket_real;
		delete[]ket_imag;
		delete[]ket_real_1;
		delete[]ket_imag_1;
	}
	// write obs 
	Vec_fwrite_double("ed_nit.bin", nit, LatticeSize * nts);
}