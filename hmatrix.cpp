#include "hmatrix.h"

// lattice type: Qubits 
void Ham_Qubits<double>::Qubits_MakeLatticeHops(std::vector<std::tuple<int, int, double> >& hops)
{
	// int N_hops = LatticeSize * (LatticeSize - 1);
	// read in couplings 
	Jij = new double[LatticeSize * LatticeSize];
	ifstream model_Jij("Jij.in", ios::in);
	if (model_Jij)
	{
		for (int i = 0; i < LatticeSize; i++)
		{
			for (int j = 0; j < LatticeSize; j++)
			{
				model_Jij >> Jij[i * LatticeSize + j];
			}
		}
		model_Jij.close();
	}
	else
	{
		cout << "Error: no Jij.in found! Exit!" << endl;
		exit(-1);
	}

	// rescale couplings to experimental-friendly style 
	double rescale_coeff = 2 * PI / 1000;			// initial unit: MHz/2/pi, after: GHz, corresponding time scale: ns

	// save only nonzero couplings 
	int N_hops = 0;
	for (int i = 0; i < LatticeSize; i++)
	{
		for (int j = i + 1; j < LatticeSize; j++)
		{
			if (0 != Jij[i * LatticeSize + j])
			{
				auto bar = std::make_tuple(i, j, Jij[i * LatticeSize + j] * rescale_coeff);
				hops.push_back(bar);
				N_hops++;
			}
		}
	}
	hops.resize(N_hops);
}