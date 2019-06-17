#include "RBF.h"

extern "C" {

	double get_distance(double* Xpredict, double* Xn, int inputCountPerSample) {

		// l2-norm
		double accum = 0.;

		for (int i = 0; i < inputCountPerSample; ++i) {
			double res = Xpredict[i] - Xn[i];
			accum += res * res;
		}

		double norm = sqrt(accum);
		return norm;
	}

	double gauss(double* Xpredict, double* Xn, double gamma, int inputCountPerSample) {
		double dist = get_distance(Xpredict, Xn, inputCountPerSample);
		double pow = std::pow(dist, 2);
		double gam_pow = -gamma * pow;
		double exp = std::exp(gam_pow);
		return exp;
	}

	SUPEREXPORT int predict_class_RBF_naive(double* W, double* X, double* Xpredict, int inputCountPerSample, double gamma, int N) {
		if (predict_reg_RBF_naive(W, X, Xpredict, inputCountPerSample, gamma, N) >= 0)
			return 1;
		return -1;
	}

	SUPEREXPORT double predict_reg_RBF_naive(double* W, double* X, double* Xpredict, int inputCountPerSample, double gamma, int N) {
		double* Xn = new double[inputCountPerSample];
		double w_sum = 0;

		for (int n = 0; n < N; n++)
		{
			for (int i = 0; i < inputCountPerSample; i++)
			{
				Xn[i] = X[(inputCountPerSample * n) + i];
			}
			w_sum += W[n] * gauss(Xpredict, Xn, gamma, inputCountPerSample);
		}
		return w_sum;
	}

	SUPEREXPORT double* fit_reg_RBF_naive(double* XTrain, double gamma, double* YTrain, int sampleCount, int inputCountPerSample) {
		Eigen::MatrixXd phi(sampleCount, sampleCount);
		Eigen::MatrixXd Y(sampleCount, 1);

		double* Xn1 = new double[inputCountPerSample];
		double* Xn2 = new double[inputCountPerSample];
		for (int x = 0; x < sampleCount; x++)
		{
			for (int y = 0; y < sampleCount; y++)
			{
				for (int i = 0; i < inputCountPerSample; i++)
				{
					Xn1[i] = XTrain[(inputCountPerSample * x) + i];
				}
				for (int i = 0; i < inputCountPerSample; i++)
				{
					Xn2[i] = XTrain[(inputCountPerSample * y) + i];
				}
				phi(x, y) = gauss(Xn1, Xn2, gamma, inputCountPerSample);
			}
		}

		for (int x = 0; x < sampleCount; x++)
			Y(x, 0) = YTrain[x];

		Eigen::MatrixXd W(inputCountPerSample, 1);
		auto inv = phi.inverse();
		W = inv * Y;


		double* Wmat = new double[sampleCount];



		for (int i = 0; i < sampleCount; i++)
			Wmat[i] = W(i);

		return Wmat;
	}
}