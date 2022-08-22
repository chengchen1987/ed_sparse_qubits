#pragma once

#include <cstdlib>
#include <cstdio>
#include <iostream> 
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <vector>
#include <tuple>
#include <algorithm>
using namespace std;

#include "basis.h" 
#include "sparsemat.h"

class Parameters
{
public:
	// Basis
	int LatticeSize, N_up;
	double* para_mu;
	// choose a model, do it later
	void SetDefault();
	void Read_input();

	// evolution
//	double evo_time_min;
	double evo_time_max;
	int evo_time_steps;
	int evo_time_IncStyle;	// 0/1 for linear/exponetial time increase
	double* time_vec;
	void GetTimeVec();

	int evo_type; // 0/1 for Lanczos/RK4 evolution
	int evo_Lanczos_M;

	char evo_ni;
	char evo_ninj;
	char evo_IPR;
	char evo_SE;
	char evo_PI_r;
	char evo_EE;
	char evo_Pmax;
	// entanglement entropy <- information of subsystem:
	int EE_size_A;
	int* EE_sites_A;

	// disorder
	double Rand_V;
};

class Dyn_DataStruc
{
public:

	void Initialize(Parameters& para, const l_int& _dim, const int& _prt_ind);
	void ReleaseSpace(Parameters& para);
	void PrintDynResults(Parameters& para);

	int prt_ind;

	int dim;

	double* ni_t;
	double* ninj_t;
	double* IPR_t;
	double* PI_r_t;
	double* HamD_k; // used to accelerate calculating Pi_r_t
	double* SE_t;
	double* EE_t;
	double* Pmax_t;
};

template <class Type>
class Ham_Qubits
{
private:
	Basis* basis;

public:
	Ham_Qubits(Basis* _basis);
	Parameters Params;
	~Ham_Qubits();

	int LatticeSize;
	l_int Dim;

	// numerical parameters 
	double* Jij;
	double* mu;

	// 
	double E_max;
	double E_min;

	// lattice 
	int N_hops;
	std::vector<std::tuple<int, int, Type> > hops;
	void Qubits_MakeLatticeHops(std::vector<std::tuple<int, int, Type> >& hops);

	// sparse matrix 
	l_int SparaseMat_Counts();
	void SparseMat_Build();

	SparseMat<Type> SMat;
	l_int nnz;

	void SparseMat_Eigs();
	void SparseMat_E_max();	// compute E_max

	// time evolution 
	void SparseMat_Evolution();

	// Dense Matrix evolution, as a benchmark
	void DenseMat_Evolution();

	// hmaitrx, obs from wavefunctions 
	void Cal_ni_wfsq(double* wfsq, double* ni);
	void Cal_ninj_wfsq(double* wfsq, double* ninj);
	void Cal_PIr_wfsq(double* wfsq, double* HamD_pk, double* PI_r);
	double Cal_SE_wfsq(double* wfsq);
	double Cal_IPR_wfsq(double* wfsq);
	double Cal_EE_wf(double* wf, const int& size_A, int* sites_A);
	double Cal_EE_wf(my_cmplx* wf, const int& size_A, int* sites_A);
	double Cal_Pmax_wfsq(double* wfsq);
	// 
	void Cal_product_spectrum(double* pspec);
	l_int Get_TargetFock_left(const double& Target_E, std::vector <std::pair<double, l_int> >& Fock_E_n);
	l_int Get_TargetFock_right(const double& Target_E, std::vector <std::pair<double, l_int> >& Fock_E_n);
};

template <class Type>
Ham_Qubits<Type>::Ham_Qubits(Basis* _basis) :
	basis(_basis),
	LatticeSize(basis->get_L()),
	Dim(basis->get_Dim())
{
	cout << "Dim = " << Dim << endl;

	Params.SetDefault();
	Params.Read_input();

	// parse Hamiltonian parameters and 
	// generate hopping bonds for Creutz lattice
	Qubits_MakeLatticeHops(hops);
	N_hops = hops.size();
	cout << "N_hops = " << N_hops << endl;

	// read/generate chemical potentials here
	Params.para_mu = new double[LatticeSize];
	ifstream model_dis("input_site_disorders.in", ios::in);
	if (model_dis) {
		std::cout << "Random onsite disorder read from input_site_disorder.dat" << endl;
		for (int i = 0; i < LatticeSize; i++) {
			model_dis >> Params.para_mu[i];
		}
	}
	else {
		for (int i = 0; i < LatticeSize; i++) { Params.para_mu[i] = Params.Rand_V * 2 * (ran_num() - 0.5); }
		//for (int i = 0; i < LatticeSize; i++) { Params.para_mu[i] = Params.Rand_V * 2 * (uni_rd() - 0.5); }
	}
	std::cout << "input Random onsite potential (MHz/2/pi):" << endl;
	ofstream of_disorder("input_site_disorders.in");
	for (int i = 0; i < LatticeSize; i++) {
		std::cout << setprecision(8) << Params.para_mu[i] << endl;
		of_disorder << setprecision(8) << Params.para_mu[i] << endl;
	}
	of_disorder.close();
	std::cout << endl;

	// rescale couplings to experimental-friendly style 
	double rescale_coeff = 2 * PI / 1000;			// initial unit: MHz/2/pi, after: GHz, corresponding time scale: ns
	mu = new double[LatticeSize];
	for (int i = 0; i < LatticeSize; i++) {
		mu[i] = rescale_coeff * Params.para_mu[i];
	}
	std::cout << "onsite potential in code (GHz):" << endl;
	for (int i = 0; i < LatticeSize; i++) {
		std::cout << setprecision(8) << mu[i] << endl;
	}
	std::cout << endl;
}

template <class Type>
Ham_Qubits<Type>::~Ham_Qubits()
{
	/*
	delete[]SMat_cols;
	delete[]SMat_vals;
	delete[]SMat_PointerBE;
	*/
}

template <class Type>
void Ham_Qubits<Type>::Cal_product_spectrum(double* pspec)
{
	for (l_int k = 0; k < Dim; k++)
	{
		double aux_diag = 0;
		for (int i = 0; i < LatticeSize; i++)
		{
			int k_i = ((basis->get_state(k) >> i) & 1);
			aux_diag += mu[i] * k_i;
		}
		pspec[k] = aux_diag;
	}
}

template <class Type>
l_int Ham_Qubits<Type>::Get_TargetFock_left(const double& Target_E, vector <pair<double, l_int> >& Fock_E_n) {
	for (l_int p = 0; p < Dim; p++) {
		if (Fock_E_n[p].first > Target_E) {
			return p;
		}
	}
	return -1;
}

template <class Type>
l_int Ham_Qubits<Type>::Get_TargetFock_right(const double& Target_E, vector <pair<double, l_int> >& Fock_E_n) {
	for (l_int p = Dim-1; p > 0; p--) {
		if (Fock_E_n[p].first < Target_E) {
			return p;
		}
	}
	return -1;
}

#include "hmatrix_sparse.h"

#include "hmatrix_sparse_dynamic.h"

#include "hmatrix_wf_obs.h"