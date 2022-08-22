#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdio>
#include <ctime>
#include <cmath>
using namespace std;
#include "common.h"
#include "basis.h"
#include "hmatrix.h"

int main() {

	// model information
	//cout << "Full exact Diadonalization  for many-body localization problems" << endl << endl;

	size_t time0 = time(0);
	// find states
	// read lattice informationi from input files
	int LatticeSize, nup;
	GetParaFromInput_int("input.in", "LatticeSize", LatticeSize);
	GetParaFromInput_int("input.in", "N_up", nup);
	
	cout << "LatticeSize: " << LatticeSize << endl;
	cout << "No. of up-spins: " << nup << endl;

	// generate basis
	Basis basis(LatticeSize, nup);
	size_t time1 = time(0);
	cout << "Time(s) to Build Basis: " << time1 - time0 << endl << endl;

	Ham_Qubits<double> ham(&basis);
	//Ham_Qubits<complex<double> > ham(&basis);
	ham.SparseMat_Build();
	int time2 = time(0);
	cout << "Time(s) to Build SparseMat: " << time2 - time1 << endl << endl;

	ham.SparseMat_Eigs();
	ham.SparseMat_E_max();
	int time3 = time(0);
	cout << "Time(s) to  Diagonalize SparseMat: " << time3 - time2 << endl << endl;

	ham.SparseMat_Evolution();
	int time4 = time(0);
	cout << "Time(s) for SparseMat evolution: " << time4 - time3 << endl << endl;

	/*
	ham.DenseMat_Evolution();
	int time5 = time(0);
	cout << "Time(s) for DenseMat evolution: " << time5 - time4 << endl << endl;
	*/

	return 0;
}
