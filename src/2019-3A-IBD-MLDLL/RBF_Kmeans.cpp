#include "RBF_Kmeans.h"
#include "Kmeans.h"

extern "C" {

	double get_distanceKRBF(double* Xpredict, double* Xn, int inputCountPerSample) {

		// l2-norm
		double accum = 0.;

		for (int i = 0; i < inputCountPerSample; ++i) {
			double res = Xpredict[i] - Xn[i];
			accum += res * res;
		}

		double norm = sqrt(accum);
		return norm;
	}

	SUPEREXPORT double* fit_regRBF_Kmeans(double* Kmeans, int K, double* X, double* YTrain, int sampleCount, int inputCountPerSample, double gamma) {

		Eigen::MatrixXd phi(sampleCount, K);
		Eigen::MatrixXd Y(sampleCount, 1);
		double* Kn = new double[inputCountPerSample];
		double* Xn = new double[inputCountPerSample]; // Init with big int?
		
		//convert array YTrain to matrix 
		for (int x = 0; x < sampleCount; x++)
			Y(x, 0) = YTrain[x];

		std::cout << K << " Hello \n";
		for (int k = 0; k < K; k++)
		{
			// get one K
			for (int i = 0; i < inputCountPerSample; i++)
				Kn[i] = Kmeans[(inputCountPerSample * k) + i];

			// for Kn get all distances. If distance smaller, replace
			for (int n = 0; n < sampleCount; n++)
			{
				for (int i = 0; i < inputCountPerSample; i++)
					Xn[i] = X[(inputCountPerSample * n) + i];

				double distance = get_distanceKRBF(Kn, Xn, inputCountPerSample);

				//get Phi
				phi(n, k) = std::exp(-gamma * std::pow(distance, 2));

			}


		}

		//Formula RBF + Kmeans
		auto phi_t = phi.transpose();
		auto mult = phi_t * phi;
		auto inv = mult.completeOrthogonalDecomposition().pseudoInverse();
		auto mult2 = inv * phi_t;
		auto matW = mult2 * Y;
		double* arrW = new double[sampleCount];

		for (int i = 0; i < K; i++)
			arrW[i] = matW(i);

		return arrW;
	}
}