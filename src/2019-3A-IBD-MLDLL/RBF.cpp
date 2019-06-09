#include "RBF.h"



extern "C" {

	SUPEREXPORT double get_distance(double* Xpredict, double* Xn, int inputCountPerSample) {
		
		// l2-norm
		double accum = 0.;

		for (int i = 0; i < inputCountPerSample; ++i) {
			double res = Xpredict[i] - Xn[i];
			accum += res * res;
		}

		double norm = sqrt(accum);
		std::cout << norm << "\n";
		return norm;
	}

	SUPEREXPORT double predict_reg_RBF_naive(double* W, double* X, double* Xpredict, int inputCountPerSample, double gamma, int N) {
		double* Xn = new double[inputCountPerSample];
		double w_sum = 0;
		for (int n = 1; n < N; n = n + inputCountPerSample)
		{
			for (int i = n; i < inputCountPerSample; i++)
				Xn[i - 1] = X[n + i];

			double dist = get_distance(Xpredict, Xn, inputCountPerSample);
			double pow = std::pow(dist, 2);
			double gam_pow = -gamma * pow;
			double exp = std::exp(gam_pow);
			w_sum += W[n] * exp;
		}

		return w_sum;

	}
}